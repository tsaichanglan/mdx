#!/usr/bin/env python3
"""Train the deep-learned ARA channel estimator.

The ARA network (utils/ara_estimator.py) is trained in a supervised fashion to
denoise the classical LS + linear-interpolation channel estimate towards the
true channel frequency response, following the three-stage scheme of
  Y. Wei et al., "Deep-Learned Channel Estimation for MIMO-OFDM System by
  Exploiting Frequency-Space Correlation", Electronics Letters, 2026.

Per training step:
  * a batch of PUSCH slots is transmitted through the configured channel over a
    range of SNRs,
  * the LS+linear estimate is used as the network input and the true channel as
    the target (both mapped to the frequency-space plane),
  * the ARA network is updated with an MSE loss.

The trained ARA weights are stored to
  ../weights/<label>_ara_weights            (pickled get_weights list)
and are picked up automatically by ``evaluate.py`` for the
``baseline_ara_lmmse`` / ``baseline_ara_kbest`` methods.

Example
-------
    TF_USE_LEGACY_KERAS=1 python train_ara.py -config_name ara_cdl.cfg \
        -channel_type CDL-C -num_steps 3000 -batch_size 32
"""
import os
import importlib.util
# Route tf.keras to legacy Keras 2 only when tf-keras is installed (the
# Python 3.12 / TF 2.16 / Sionna 0.19 setup). On the README stack
# (Python 3.11 / TF 2.15 / Sionna 0.18) tensorflow.keras is already Keras 2 and
# setting this var would break the import ("No module named 'tensorflow.keras'").
if importlib.util.find_spec("tf_keras") is not None:
    os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-config_name", type=str, default="ara_cdl.cfg")
parser.add_argument("-gpu", type=int, default=0)
parser.add_argument("-channel_type", type=str, default="CDL-C",
                    help="Evaluation/training channel, e.g. CDL-A..CDL-E, "
                         "TDL-B100, DoubleTDLlow, NTDLlow")
parser.add_argument("-tdl_models", type=str, nargs="+", default=["A"])
parser.add_argument("-num_tx", type=int, default=1)
parser.add_argument("-n_size_bwp", type=int, default=4)
parser.add_argument("-num_steps", type=int, default=3000)
parser.add_argument("-batch_size", type=int, default=32)
parser.add_argument("-learning_rate", type=float, default=1e-3)
parser.add_argument("-snr_db_min", type=float, default=-5.0)
parser.add_argument("-snr_db_max", type=float, default=15.0)
parser.add_argument("-max_ut_velocity", type=float, default=3.0)
parser.add_argument("-eval_every", type=int, default=250)
parser.add_argument("-seed", type=int, default=1)
parser.add_argument("-weights_dir", type=str, default="../weights/")
parser.add_argument("-name_suffix", type=str, default="")
args = parser.parse_args()

import sys
sys.path.append("../")
import numpy as np
import tensorflow as tf
try:
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        tf.config.set_visible_devices(gpus[args.gpu], "GPU")
        tf.config.experimental.set_memory_growth(gpus[args.gpu], True)
except Exception as e:
    print(f"GPU config note: {e}")

import sionna as sn
from sionna.nr import PUSCHLSChannelEstimator
from sionna.utils import BinarySource, ebnodb2no

from utils import Parameters, E2E_Model, save_weights, channel_to_freqspace

tf.random.set_seed(args.seed)


def build_params():
    p = Parameters(args.config_name, training=False, num_tx_eval=args.num_tx,
                   system="baseline_ara_lmmse")
    p.re_init(n_size_bwp_eval=args.n_size_bwp,
              batch_size_eval=args.batch_size,
              batch_size_eval_small=args.batch_size,
              max_ut_velocity_eval=args.max_ut_velocity,
              channel_type_eval=args.channel_type,
              tdl_models=args.tdl_models)
    return p


print("Building model on channel", args.channel_type, "...")
p = build_params()
model = E2E_Model(p, training=False, mcs_arr_eval_idx=0)
# build once so all variables (incl. ARA) exist
model(args.batch_size, tf.constant(5.0), num_tx=args.num_tx)

net = model._receiver._est._ara           # the ARA network
tx = p.transmitters[0]
src = BinarySource()

# a plain LS + linear-interpolation estimator provides the *input* estimate
pc = p.pusch_configs[0][0]
lin_est = PUSCHLSChannelEstimator(
    resource_grid=tx._resource_grid, dmrs_length=pc.dmrs.length,
    dmrs_additional_position=pc.dmrs.additional_position,
    num_cdm_groups_without_data=pc.dmrs.num_cdm_groups_without_data,
    interpolation_type="lin")

opt = tf.keras.optimizers.Adam(args.learning_rate)
n_params = int(np.sum([np.prod(v.shape) for v in net.trainable_variables]))
print(f"ARA trainable parameters: {n_params}")


def sample_batch(bs):
    """Draw a batch of (input estimate, target channel) in frequency-space."""
    snr = tf.random.uniform([], args.snr_db_min, args.snr_db_max)
    no = ebnodb2no(snr, tx._num_bits_per_symbol, tx._target_coderate,
                   tx._resource_grid)
    bits = src([bs, p.max_num_tx, tx._tb_size])
    x = tx(bits)
    y, h = p.channel([x, no])
    h_ls, _ = lin_est([y, no])
    return channel_to_freqspace(h_ls), channel_to_freqspace(h)


def nmse(pred, tgt):
    return float(tf.reduce_sum(tf.square(pred - tgt)) /
                 tf.reduce_sum(tf.square(tgt)))


@tf.function
def train_step(x, tgt):
    with tf.GradientTape() as tape:
        pred = net(x, training=True)
        loss = tf.reduce_mean(tf.square(pred - tgt))
    grads = tape.gradient(loss, net.trainable_variables)
    opt.apply_gradients(zip(grads, net.trainable_variables))
    return loss


# fixed validation batch for a stable NMSE read-out
xv, tv = sample_batch(max(64, args.batch_size))
base_nmse = nmse(xv, tv)
print(f"LS+linear baseline NMSE (val): {base_nmse:.4f}")
print(f"ARA NMSE before training      : {nmse(net(xv, training=False), tv):.4f} "
      "(identity init == baseline)\n")

best = float("inf")
label = p.label + args.name_suffix
os.makedirs(args.weights_dir, exist_ok=True)
weights_path = os.path.join(args.weights_dir, f"{label}_ara_weights")

for step in range(1, args.num_steps + 1):
    xb, tb = sample_batch(args.batch_size)
    loss = train_step(xb, tb)
    if step % args.eval_every == 0 or step == args.num_steps:
        val = nmse(net(xv, training=False), tv)
        gain = 10.0 * np.log10(base_nmse / max(val, 1e-9))
        flag = ""
        if val < best:
            best = val
            save_weights(net, weights_path, save_format="pkl")
            flag = "  <- saved"
        print(f"step {step:5d}/{args.num_steps}  train MSE {float(loss):.5f}  "
              f"val NMSE {val:.4f}  gain {gain:+.2f} dB{flag}")

print(f"\nDone. Best val NMSE {best:.4f} "
      f"({10.0*np.log10(base_nmse/max(best,1e-9)):+.2f} dB over LS+linear).")
print(f"ARA weights saved to: {weights_path}")
print("Evaluate with:")
print(f"  python evaluate.py -config_name {args.config_name} "
      f"-methods baseline_ara_lmmse -channel_type_eval {args.channel_type} "
      f"-num_tx_eval {args.num_tx} -n_size_bwp_eval {args.n_size_bwp} "
      "-mcs_arr_eval_idx 0")
