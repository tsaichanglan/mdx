#!/usr/bin/env python3
"""Channel-estimation NMSE vs SNR for LS, CNN and ARA estimators (16x4, CDL-C).

For each SNR, transmits PUSCH slots through the multi-user CDL-C channel,
computes each estimator's channel estimate and its NMSE against the true
channel:
  * LS   = LS despreading + linear interpolation (the classical baseline),
  * CNN  = LS+lin refined by the delay-domain CNN (pyAerial-style),
  * ARA  = LS+lin refined by the attentive residual autoencoder.

NMSE is measured on the SAME estimate the LMMSE detector consumes, so it lines
up with the BLER comparison. Results are pickled and plotted by
plot_nmse_16x4_cdlc.py.

Run from the scripts/ directory:
    PYTHON=/home/alan/sionna-env-018/bin/python \
        ../eval_16x4_cdlc/compute_nmse_16x4_cdlc.py
"""
import os
import importlib.util
if importlib.util.find_spec("tf_keras") is not None:
    os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import argparse
import pickle

import numpy as np
import tensorflow as tf

HERE = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument("-config_name", type=str, default="ara_cdl_16x4_mu.cfg")
parser.add_argument("-channel_type", type=str, default="CDL-C")
parser.add_argument("-num_tx", type=int, default=4)
parser.add_argument("-n_size_bwp", type=int, default=4)
parser.add_argument("-batch_size", type=int, default=16)
parser.add_argument("-num_batches", type=int, default=20)
parser.add_argument("-snr_min", type=float, default=-12.0)
parser.add_argument("-snr_max", type=float, default=12.0)
parser.add_argument("-snr_step", type=float, default=2.0)
parser.add_argument("-ara_weights", type=str,
                    default="../weights/ara_cdl_16x4_mu_ara_weights")
parser.add_argument("-cnn_weights", type=str,
                    default="../weights/ara_cdl_16x4_mu_cnn_weights")
parser.add_argument("-cnn_nofft_weights", type=str,
                    default="../weights/ara_cdl_16x4_mu_cnn_nofft_weights")
parser.add_argument("-out", type=str,
                    default=os.path.join(HERE, "nmse_16x4_cdlc_results"))
parser.add_argument("-seed", type=int, default=2)
args = parser.parse_args()

import sys
sys.path.append("../")
from sionna.nr import PUSCHLSChannelEstimator
from sionna.utils import BinarySource, ebnodb2no
from utils import Parameters, E2E_Model, load_weights

tf.random.set_seed(args.seed)


def build(system):
    p = Parameters(args.config_name, training=False, num_tx_eval=args.num_tx,
                   system=system)
    p.re_init(n_size_bwp_eval=args.n_size_bwp, batch_size_eval=args.batch_size,
              batch_size_eval_small=args.batch_size, max_ut_velocity_eval=3.,
              channel_type_eval=args.channel_type)
    return p


# One E2E per learned estimator (shares the same channel realisation drawing).
p_cnn = build("baseline_cnn_lmmse")
m_cnn = E2E_Model(p_cnn, training=False, mcs_arr_eval_idx=0)
m_cnn(1, 1.)
if os.path.exists(args.cnn_weights):
    load_weights(m_cnn._receiver._est._cnn, args.cnn_weights)
    print(f"CNN weights: {args.cnn_weights}")
else:
    print("WARNING: CNN weights not found -> CNN == LS baseline")

m_cnnf = None
if os.path.exists(args.cnn_nofft_weights):
    p_cnnf = build("baseline_cnn_nofft_lmmse")
    m_cnnf = E2E_Model(p_cnnf, training=False, mcs_arr_eval_idx=0)
    m_cnnf(1, 1.)
    load_weights(m_cnnf._receiver._est._cnn, args.cnn_nofft_weights)
    print(f"CNN(no FFT) weights: {args.cnn_nofft_weights}")

p_ara = build("baseline_ara_lmmse")
m_ara = E2E_Model(p_ara, training=False, mcs_arr_eval_idx=0)
m_ara(1, 1.)
if os.path.exists(args.ara_weights):
    load_weights(m_ara._receiver._est._ara, args.ara_weights)
    print(f"ARA weights: {args.ara_weights}")
else:
    print("WARNING: ARA weights not found -> ARA == LS baseline")

tx = p_cnn.transmitters[0]
src = BinarySource()
pc = p_cnn.pusch_configs[0][0]
ls_est = PUSCHLSChannelEstimator(
    resource_grid=tx._resource_grid, dmrs_length=pc.dmrs.length,
    dmrs_additional_position=pc.dmrs.additional_position,
    num_cdm_groups_without_data=pc.dmrs.num_cdm_groups_without_data,
    interpolation_type="lin")
cnn_est = m_cnn._receiver._est
ara_est = m_ara._receiver._est
cnnf_est = m_cnnf._receiver._est if m_cnnf is not None else None


def nmse_db(h_hat, h_true):
    num = tf.reduce_sum(tf.abs(h_hat - h_true) ** 2)
    den = tf.reduce_sum(tf.abs(h_true) ** 2)
    return 10.0 * np.log10(float(num / den))


snrs = np.arange(args.snr_min, args.snr_max + 1e-6, args.snr_step)
keys = ["LS", "CNN", "ARA"] + (["CNN_noFFT"] if cnnf_est is not None else [])
acc = {k: {s: [0.0, 0.0] for s in snrs} for k in keys}


def accum(store, s, h_hat, h_true):
    store[s][0] += float(tf.reduce_sum(tf.abs(h_hat - h_true) ** 2))
    store[s][1] += float(tf.reduce_sum(tf.abs(h_true) ** 2))


for s in snrs:
    no = ebnodb2no(float(s), tx._num_bits_per_symbol, tx._target_coderate,
                   tx._resource_grid)
    for _ in range(args.num_batches):
        bits = src([args.batch_size, p_cnn.max_num_tx, tx._tb_size])
        x = tx(bits)
        y, h = p_cnn.channel([x, no])          # true channel + rx grid
        h_ls, _ = ls_est([y, no])              # [B,R,A,T,S,O,F]
        h_cnn, _ = cnn_est([y, no])
        h_ara, _ = ara_est([y, no])
        # true channel already matches the estimate layout (num_tx_ant = 1)
        h_true = h
        accum(acc["LS"], s, h_ls, h_true)
        accum(acc["CNN"], s, h_cnn, h_true)
        accum(acc["ARA"], s, h_ara, h_true)
        if cnnf_est is not None:
            h_cnnf, _ = cnnf_est([y, no])
            accum(acc["CNN_noFFT"], s, h_cnnf, h_true)
    line = {k: 10.0 * np.log10(acc[k][s][0] / acc[k][s][1]) for k in acc}
    extra = f"   CNN_noFFT {line['CNN_noFFT']:6.2f}" if "CNN_noFFT" in line else ""
    print(f"SNR {s:+5.1f} dB   LS {line['LS']:6.2f}   "
          f"CNN {line['CNN']:6.2f}{extra}   ARA {line['ARA']:6.2f}  [dB]", flush=True)

nmse = {k: [10.0 * np.log10(acc[k][s][0] / acc[k][s][1]) for s in snrs]
        for k in acc}
with open(args.out, "wb") as f:
    pickle.dump({"snrs": snrs, "nmse_db": nmse}, f)
print(f"saved {args.out}")
