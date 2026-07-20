#!/usr/bin/env python3
"""Train the 1D delay-domain CNN channel estimator (pyAerial-style port).

Same supervised-denoising setup as train_ara.py: the classical LS + linear
interpolation estimate is the network input and the true channel the target,
sampled over an SNR range through the configured channel. Following the
pyAerial recipe, the CNN is trained with AdamW and a (complex) NMSE loss.

Trained weights are stored to  ../weights/<label>_cnn_weights  and picked up
automatically by evaluate.py for the baseline_cnn_lmmse method.

Example
-------
    python train_cnn.py -config_name ara_cdl_16x4.cfg -channel_type CDL-C \
        -num_steps 1500 -batch_size 16
"""
import os
import importlib.util
if importlib.util.find_spec("tf_keras") is not None:
    os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-config_name", type=str, default="ara_cdl_16x4.cfg")
parser.add_argument("-system", type=str, default="baseline_cnn_lmmse",
                    choices=["baseline_cnn_lmmse", "baseline_cnn_nofft_lmmse"],
                    help="CNN variant: with (default) or without delay-domain FFT")
parser.add_argument("-gpu", type=int, default=0)
parser.add_argument("-channel_type", type=str, default="CDL-C")
parser.add_argument("-tdl_models", type=str, nargs="+", default=["A"])
parser.add_argument("-num_tx", type=int, default=1)
parser.add_argument("-n_size_bwp", type=int, default=4)
parser.add_argument("-num_steps", type=int, default=1500)
parser.add_argument("-batch_size", type=int, default=16)
parser.add_argument("-learning_rate", type=float, default=1e-3)
parser.add_argument("-weight_decay", type=float, default=1e-4)
parser.add_argument("-snr_db_min", type=float, default=-12.0)
parser.add_argument("-snr_db_max", type=float, default=2.0)
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

from sionna.nr import PUSCHLSChannelEstimator
from sionna.utils import BinarySource, ebnodb2no
from utils import Parameters, E2E_Model, save_weights, channel_to_subcarrier

tf.random.set_seed(args.seed)


WEIGHTS_TAG = "cnn_nofft" if "nofft" in args.system else "cnn"


def build_params():
    p = Parameters(args.config_name, training=False, num_tx_eval=args.num_tx,
                   system=args.system)
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
model(args.batch_size, tf.constant(5.0), num_tx=args.num_tx)

net = model._receiver._est._cnn           # the CNN network
tx = p.transmitters[0]
src = BinarySource()

pc = p.pusch_configs[0][0]
lin_est = PUSCHLSChannelEstimator(
    resource_grid=tx._resource_grid, dmrs_length=pc.dmrs.length,
    dmrs_additional_position=pc.dmrs.additional_position,
    num_cdm_groups_without_data=pc.dmrs.num_cdm_groups_without_data,
    interpolation_type="lin")

# pyAerial recipe: AdamW + complex NMSE loss
try:
    opt = tf.keras.optimizers.AdamW(learning_rate=args.learning_rate,
                                    weight_decay=args.weight_decay)
except Exception:
    opt = tf.keras.optimizers.experimental.AdamW(learning_rate=args.learning_rate,
                                                 weight_decay=args.weight_decay)
n_params = int(np.sum([np.prod(v.shape) for v in net.trainable_variables]))
print(f"CNN trainable parameters: {n_params}")


def sample_batch(bs):
    snr = tf.random.uniform([], args.snr_db_min, args.snr_db_max)
    no = ebnodb2no(snr, tx._num_bits_per_symbol, tx._target_coderate,
                   tx._resource_grid)
    bits = src([bs, p.max_num_tx, tx._tb_size])
    x = tx(bits)
    y, h = p.channel([x, no])
    h_ls, _ = lin_est([y, no])
    return channel_to_subcarrier(h_ls), channel_to_subcarrier(h)


def nmse(pred, tgt):
    return float(tf.reduce_sum(tf.square(pred - tgt)) /
                 tf.reduce_sum(tf.square(tgt)))


@tf.function
def train_step(x, tgt):
    with tf.GradientTape() as tape:
        pred = net(x, training=True)
        # normalised MSE (ComplexMSELoss equivalent)
        loss = tf.reduce_sum(tf.square(pred - tgt)) / tf.reduce_sum(tf.square(tgt))
    grads = tape.gradient(loss, net.trainable_variables)
    opt.apply_gradients(zip(grads, net.trainable_variables))
    return loss


xv, tv = sample_batch(max(64, args.batch_size))
base_nmse = nmse(xv, tv)
print(f"LS+linear baseline NMSE (val): {base_nmse:.4f}")
print(f"CNN NMSE before training      : {nmse(net(xv, training=False), tv):.4f} "
      "(identity init == baseline)\n")

best = float("inf")
label = p.label + args.name_suffix
os.makedirs(args.weights_dir, exist_ok=True)
weights_path = os.path.join(args.weights_dir, f"{label}_{WEIGHTS_TAG}_weights")

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
        print(f"step {step:5d}/{args.num_steps}  train NMSE {float(loss):.5f}  "
              f"val NMSE {val:.4f}  gain {gain:+.2f} dB{flag}")

print(f"\nDone. Best val NMSE {best:.4f} "
      f"({10.0*np.log10(base_nmse/max(best,1e-9)):+.2f} dB over LS+linear).")
print(f"CNN weights saved to: {weights_path}")
