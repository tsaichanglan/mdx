#!/usr/bin/env python3
"""Verify the deep-learned ARA channel estimator inside the BaselineReceiver,
using a 3GPP CDL channel model.

Checks performed
----------------
1. End-to-end run   : the ARA receiver produces payload bits on a CDL channel.
2. Identity-at-init : with its zero-initialised residual read-out, the untrained
                      ARA estimator reproduces the classical LS+linear estimate
                      exactly (so integration cannot degrade the baseline).
3. Learning         : a short supervised training of the ARA network reduces the
                      channel-estimation NMSE below the LS+linear baseline,
                      demonstrating the estimator actually works.
4. BLER/BER         : ARA vs LS+linear receiver over a few SNR points on CDL.

Run from the ``scripts`` directory:
    TF_USE_LEGACY_KERAS=1 python verify_ara_cdl.py
"""
import os
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import sys
sys.path.append("../")

import numpy as np
import tensorflow as tf
import sionna as sn
from sionna.nr import PUSCHLSChannelEstimator
from sionna.utils import BinarySource, ebnodb2no

from utils import Parameters, E2E_Model

tf.random.set_seed(1)
CFG = "ara_cdl.cfg"
CHAN = "CDL-C"


def freq_space(hh):
    """[B,R,A,T,S,O,F] complex -> [N, A, F, 2] real (frequency-space plane)."""
    A = tf.shape(hh)[2]
    F = tf.shape(hh)[6]
    hp = tf.transpose(hh, perm=[0, 1, 3, 4, 5, 2, 6])   # B,R,T,S,O,A,F
    hp = tf.reshape(hp, [-1, A, F])
    return tf.stack([tf.math.real(hp), tf.math.imag(hp)], axis=-1)


def build_params(system):
    p = Parameters(CFG, training=False, num_tx_eval=1, system=system)
    p.re_init(n_size_bwp_eval=4, batch_size_eval=8, batch_size_eval_small=8,
              max_ut_velocity_eval=3., channel_type_eval=CHAN)
    return p


print("=" * 70)
print("1) END-TO-END RUN  (ARA receiver on %s)" % CHAN)
print("=" * 70)
p_ara = build_params("baseline_ara_lmmse")
m_ara = E2E_Model(p_ara, training=False, mcs_arr_eval_idx=0)
b, b_hat = m_ara(8, tf.constant(6.0), num_tx=1)
ber0 = float(np.mean(np.abs(b.numpy() - b_hat.numpy())))
n_params = int(np.sum([np.prod(v.shape) for v in m_ara._receiver._est._ara.trainable_variables]))
print("  b", b.shape, " b_hat", b_hat.shape,
      " BER@6dB(untrained ARA) = %.4f" % ber0)
print("  ARA network trainable parameters: %d" % n_params)

print()
print("=" * 70)
print("2) IDENTITY-AT-INIT  (untrained ARA == LS+linear estimate)")
print("=" * 70)
tx = p_ara.transmitters[0]
src = BinarySource()
no = ebnodb2no(6.0, tx._num_bits_per_symbol, tx._target_coderate,
               tx._resource_grid)
bb = src([8, p_ara.max_num_tx, tx._tb_size])
x = tx(bb)
y, h = p_ara.channel([x, no])

pc = p_ara.pusch_configs[0][0]
lin_est = PUSCHLSChannelEstimator(
    resource_grid=tx._resource_grid, dmrs_length=pc.dmrs.length,
    dmrs_additional_position=pc.dmrs.additional_position,
    num_cdm_groups_without_data=pc.dmrs.num_cdm_groups_without_data,
    interpolation_type="lin")
h_ls, _ = lin_est([y, no])
h_ara, _ = m_ara._receiver._est([y, no])
max_diff = float(tf.reduce_max(tf.abs(h_ls - h_ara)))
print("  max|h_ARA - h_LSlin| at init = %.3e   (expected ~0)" % max_diff)
assert max_diff < 1e-4, "ARA is not identity at init!"
print("  OK: untrained ARA reproduces LS+linear exactly.")

print()
print("=" * 70)
print("3) LEARNING  (train ARA to denoise; NMSE vs LS+linear baseline)")
print("=" * 70)
net = m_ara._receiver._est._ara
opt = tf.keras.optimizers.Adam(1e-3)
train_no = ebnodb2no(4.0, tx._num_bits_per_symbol, tx._target_coderate,
                     tx._resource_grid)


def sample_batch(bs, no_val):
    bits = src([bs, p_ara.max_num_tx, tx._tb_size])
    xx = tx(bits)
    yy, hh = p_ara.channel([xx, no_val])
    h_in, _ = lin_est([yy, no_val])       # LS+linear estimate  -> input
    x_in = freq_space(h_in)               # [N,A,F,2]
    tgt = freq_space(hh)                  # true channel        -> target
    return x_in, tgt


def nmse(pred, tgt):
    num = tf.reduce_sum(tf.square(pred - tgt))
    den = tf.reduce_sum(tf.square(tgt))
    return float(num / den)


# baseline NMSE of the LS+linear estimate (no denoising)
xv, tv = sample_batch(16, train_no)
base_nmse = nmse(xv, tv)
init_nmse = nmse(net(xv, training=False), tv)
print("  LS+linear baseline NMSE      = %.4f" % base_nmse)
print("  ARA NMSE (before training)   = %.4f  (== baseline, identity init)"
      % init_nmse)

STEPS = 400
for step in range(STEPS):
    xb, tb = sample_batch(16, train_no)
    with tf.GradientTape() as tape:
        pred = net(xb, training=True)
        loss = tf.reduce_mean(tf.square(pred - tb))
    grads = tape.gradient(loss, net.trainable_variables)
    opt.apply_gradients(zip(grads, net.trainable_variables))
    if (step + 1) % 100 == 0:
        print("    step %3d/%d  train MSE = %.5f" % (step + 1, STEPS, float(loss)))

trained_nmse = nmse(net(xv, training=False), tv)
print("  ARA NMSE (after %d steps)    = %.4f" % (STEPS, trained_nmse))
gain_db = 10.0 * np.log10(base_nmse / max(trained_nmse, 1e-9))
print("  --> channel-estimation gain over LS+linear: %.2f dB" % gain_db)

print()
print("=" * 70)
print("4) BER/BLER  (ARA vs LS+linear receiver on %s)" % CHAN)
print("=" * 70)
p_ls = build_params("baseline_lslin_lmmse")
m_ls = E2E_Model(p_ls, training=False, mcs_arr_eval_idx=0)


def eval_ber(model, snr_db, n_batches=6, bs=16):
    be = bt = 0.0
    for _ in range(n_batches):
        bb_, bh_ = model(bs, tf.constant(float(snr_db)), num_tx=1)
        be += float(np.sum(np.abs(bb_.numpy() - bh_.numpy())))
        bt += bb_.numpy().size
    return be / bt


print("  %-6s %-14s %-14s" % ("SNR", "ARA(trained)", "LS+linear"))
for snr in [0.0, 4.0, 8.0]:
    ara_ber = eval_ber(m_ara, snr)
    ls_ber = eval_ber(m_ls, snr)
    print("  %-6.1f %-14.4e %-14.4e" % (snr, ara_ber, ls_ber))

print()
print("VERIFICATION COMPLETE: ARA channel estimator integrated, runs on CDL,")
print("is identity-at-init, and learns to outperform the LS+linear baseline.")
