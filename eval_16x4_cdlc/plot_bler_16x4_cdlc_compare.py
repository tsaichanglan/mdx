#!/usr/bin/env python3
"""BLER vs SNR for LS vs CNN vs ARA channel estimators (16x4 MU-MIMO, CDL-C).

One panel per MCS (9/14/19), 4 active users, all three estimators feeding the
same LMMSE equalizer. Reads the merged results file written by the LS/ARA and
CNN evaluation runs.
"""
import argparse
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument("-results", type=str,
                    default=os.path.join(HERE, "ara_cdl_16x4_mu_results"))
parser.add_argument("-out", type=str,
                    default=os.path.join(HERE, "bler_16x4_cdlc_compare.png"))
parser.add_argument("-num_tx", type=int, default=4)
args = parser.parse_args()

MCS_LABEL = {0: 9, 1: 14, 2: 19}
STYLE = {
    "Baseline - LS/lin+LMMSE":     dict(color="#7030a0", marker="o", ls="--",
                                        label="LS + linear interp"),
    "Baseline - CNN(no FFT)+LMMSE":dict(color="#f0a500", marker="v", ls="-",
                                        label="CNN, no FFT (freq domain)"),
    "Baseline - CNN+LMMSE":        dict(color="#1f9e46", marker="^", ls="-",
                                        label="CNN, delay domain (pyAerial)"),
    "Baseline - ARA+LMMSE":        dict(color="#e8288c", marker="s", ls="-",
                                        label="ARA (attentive autoencoder)"),
}
ORDER = ["Baseline - LS/lin+LMMSE", "Baseline - CNN(no FFT)+LMMSE",
         "Baseline - CNN+LMMSE", "Baseline - ARA+LMMSE"]

with open(args.results, "rb") as f:
    data = pickle.load(f)
ebno, BLERs, SNRs = data[0], data[2], (data[7] if len(data) == 8 else {})
for k in BLERs:
    SNRs.setdefault(k, ebno)
print(f"loaded {args.results}")
for k in BLERs:
    print("  curve:", k)

mcs_idxs = sorted({k[2] for k in BLERs})
fig, axes = plt.subplots(1, len(mcs_idxs), figsize=(5 * len(mcs_idxs), 4.2),
                         squeeze=False)

for col, mcs_idx in enumerate(mcs_idxs):
    ax = axes[0][col]
    for sys_name in ORDER:
        key = (sys_name, args.num_tx, mcs_idx)
        if key not in BLERs:
            continue
        y = np.asarray(BLERs[key], dtype=float)
        x = np.asarray(SNRs[key], dtype=float)[:len(y)]
        mask = y > 0
        if not mask.any():
            continue
        st = dict(STYLE[sys_name])
        label = st.pop("label")
        ax.semilogy(x[mask], y[mask], label=label, markersize=5, **st)
    ax.grid(True, which="both", alpha=0.3)
    ax.set_ylim(1e-3, 1.1)
    ax.set_xlabel(r"$E_b/N_0$ [dB]")
    if col == 0:
        ax.set_ylabel("TBLER")
    ax.set_title(f"MCS {MCS_LABEL.get(mcs_idx, mcs_idx)}  "
                 f"({args.num_tx} active users)")

handles, labels = axes[0][0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=len(labels),
           bbox_to_anchor=(0.5, 1.07), frameon=True)
fig.suptitle("16x4 MU-MIMO, CDL-C — LS vs CNN vs ARA channel estimator "
             "(+ LMMSE equalizer)", y=1.15, fontsize=11)
fig.tight_layout()
fig.savefig(args.out, dpi=150, bbox_inches="tight")
print(f"saved {args.out}")
