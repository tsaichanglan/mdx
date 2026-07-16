#!/usr/bin/env python3
"""Plot BLER (and BER) vs SNR for the 16x4 / CDL-C evaluation.

Reads the results file produced by run_eval_16x4_cdlc.sh and plots one curve
per receiver (classical LS/lin+LMMSE baseline vs deep-learned ARA+LMMSE).

Usage (from this directory):
    python3 plot_bler_16x4_cdlc.py
    python3 plot_bler_16x4_cdlc.py -results ara_cdl_16x4_results
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
                    default=os.path.join(HERE, "ara_cdl_16x4_results"),
                    help="path to the pickled results file")
parser.add_argument("-out_prefix", type=str,
                    default=os.path.join(HERE, "16x4_cdlc"),
                    help="prefix for the generated .png files")
args = parser.parse_args()


def load_results(path):
    """Return (BERs, BLERs, SNRs) dicts keyed by (sys_name, num_tx, mcs_idx)."""
    with open(path, "rb") as f:
        data = pickle.load(f)
    if len(data) == 3:
        ebno_db, BERs, BLERs = data
        SNRs = {}
    elif len(data) == 7:
        ebno_db, BERs, BLERs, _, _, _, _ = data
        SNRs = {}
    elif len(data) == 8:
        ebno_db, BERs, BLERs, _, _, _, _, SNRs = data
    else:
        raise ValueError(f"Unexpected results format (len={len(data)})")
    # fall back to the global SNR axis when per-key SNRs are missing
    for k in BLERs:
        SNRs.setdefault(k, ebno_db)
    return BERs, BLERs, SNRs


# nicer, stable styling per receiver
STYLE = {
    "Baseline - LS/lin+LMMSE": dict(color="#1f77b4", marker="o", ls="--"),
    "Baseline - ARA+LMMSE":    dict(color="#d62728", marker="s", ls="-"),
}


def plot(metric, values, SNRs, ylabel, title, fname):
    plt.figure(figsize=(7, 5))
    plotted = 0
    for key in sorted(values, key=lambda k: str(k)):
        sys_name, num_tx, mcs_idx = key
        y = np.asarray(values[key], dtype=float)
        x = np.asarray(SNRs[key], dtype=float)[:len(y)]
        # drop trailing zeros (simulation early-stopped => no errors observed)
        mask = y > 0
        if not mask.any():
            continue
        style = STYLE.get(sys_name, {})
        label = f"{sys_name} ({num_tx} UE, MCS idx {mcs_idx})"
        plt.semilogy(x[mask], y[mask], label=label, markersize=5, **style)
        plotted += 1
    if plotted == 0:
        print(f"[warn] no non-zero {metric} values to plot")
        return
    plt.grid(True, which="both", alpha=0.3)
    plt.xlabel("Eb/N0 [dB]")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    print(f"saved {fname}")


BERs, BLERs, SNRs = load_results(args.results)
print(f"loaded {args.results}")
for k in BLERs:
    print("  curve:", k)

plot("BLER", BLERs, SNRs, "BLER",
     "16x4 CDL-C: BLER vs SNR — classical baseline vs deep-learned ARA",
     f"{args.out_prefix}_bler.png")
plot("BER", BERs, SNRs, "BER",
     "16x4 CDL-C: BER vs SNR — classical baseline vs deep-learned ARA",
     f"{args.out_prefix}_ber.png")
