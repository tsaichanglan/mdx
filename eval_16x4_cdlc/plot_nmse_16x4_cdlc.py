#!/usr/bin/env python3
"""Plot channel-estimation NMSE vs SNR for LS, CNN and ARA estimators."""
import argparse
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument("-results", type=str,
                    default=os.path.join(HERE, "nmse_16x4_cdlc_results"))
parser.add_argument("-out", type=str,
                    default=os.path.join(HERE, "nmse_16x4_cdlc.png"))
args = parser.parse_args()

with open(args.results, "rb") as f:
    d = pickle.load(f)
snrs, nmse = d["snrs"], d["nmse_db"]

STYLE = {
    "LS":        dict(color="#7030a0", marker="o", ls="--", label="LS + linear interp"),
    "CNN_noFFT": dict(color="#f0a500", marker="v", ls="-",  label="CNN, no FFT (freq domain)"),
    "CNN":       dict(color="#1f9e46", marker="^", ls="-",  label="CNN, delay domain (pyAerial)"),
    "ARA":       dict(color="#e8288c", marker="s", ls="-",  label="ARA (attentive autoencoder)"),
}

plt.figure(figsize=(7, 5))
for k in ["LS", "CNN_noFFT", "CNN", "ARA"]:
    if k in nmse:
        plt.plot(snrs, nmse[k], markersize=5, **STYLE[k])
plt.grid(True, which="both", alpha=0.3)
plt.xlabel("SNR [dB]")
plt.ylabel("Channel estimation NMSE [dB]")
plt.title("16x4 MU-MIMO, CDL-C — channel-estimation NMSE\n"
          "LS vs CNN vs ARA (all feeding the LMMSE equalizer)")
plt.legend()
plt.tight_layout()
plt.savefig(args.out, dpi=150)
print(f"saved {args.out}")
