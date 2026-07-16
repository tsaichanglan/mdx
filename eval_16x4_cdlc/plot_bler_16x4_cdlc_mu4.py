#!/usr/bin/env python3
"""Plot BLER vs SNR for the 16x4 MU-MIMO / CDL-C evaluation (4 active users).

Produces a one-row panel figure (one subplot per MCS), in the style of
eval_16x4_tdla/bler_16x4_tdla.png: TBLER vs Eb/N0 with one curve per receiver.

Usage (from this directory):
    python3 plot_bler_16x4_cdlc_mu4.py
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
                    default=os.path.join(HERE, "bler_16x4_cdlc_mu4.png"))
parser.add_argument("-num_tx", type=int, default=4)
args = parser.parse_args()

# mcs array index -> actual MCS index in the config
MCS_LABEL = {0: 9, 1: 14, 2: 19}

STYLE = {
    "Baseline - LS/lin+LMMSE": dict(color="#7030a0", marker="o", ls="-",
                                    label="Baseline: LSlin+LMMSE"),
    "Baseline - ARA+LMMSE":    dict(color="#e8288c", marker="s", ls="-",
                                    label="ARA+LMMSE (deep-learned chest)"),
}


def load(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    if len(data) == 8:
        ebno, BERs, BLERs, _, _, _, _, SNRs = data
    elif len(data) == 7:
        ebno, BERs, BLERs, _, _, _, _ = data
        SNRs = {}
    else:
        ebno, BERs, BLERs = data
        SNRs = {}
    for k in BLERs:
        SNRs.setdefault(k, ebno)
    return BLERs, SNRs


BLERs, SNRs = load(args.results)
print(f"loaded {args.results}")
for k in BLERs:
    print("  curve:", k)

mcs_idxs = sorted({k[2] for k in BLERs})
fig, axes = plt.subplots(1, len(mcs_idxs), figsize=(5 * len(mcs_idxs), 4.2),
                         squeeze=False)

for col, mcs_idx in enumerate(mcs_idxs):
    ax = axes[0][col]
    for key in sorted(BLERs, key=lambda k: str(k)):
        sys_name, num_tx, m_idx = key
        if m_idx != mcs_idx or num_tx != args.num_tx:
            continue
        y = np.asarray(BLERs[key], dtype=float)
        x = np.asarray(SNRs[key], dtype=float)[:len(y)]
        mask = y > 0
        if not mask.any():
            continue
        st = dict(STYLE.get(sys_name, {}))
        label = st.pop("label", sys_name)
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
           bbox_to_anchor=(0.5, 1.06), frameon=True)
fig.suptitle("16x4 MU-MIMO, CDL-C — classical baseline vs deep-learned ARA "
             "channel estimator", y=1.14, fontsize=11)
fig.tight_layout()
fig.savefig(args.out, dpi=150, bbox_inches="tight")
print(f"saved {args.out}")
