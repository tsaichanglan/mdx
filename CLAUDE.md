# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

MDX is a compute/memory-efficient, model-driven neural 5G NR PUSCH receiver for MU-MIMO
(paper: arXiv:2508.12892). It is built on **NVIDIA Sionna** (link-level 5G simulation) and
**TensorFlow**, forked from NVIDIA's Neural-Rx (NRX). It also hosts two deep-learned channel
estimators added on top of the classical baseline receiver: **ARA** (Attentive Residual
Autoencoder) and a **1D delay-domain CNN**.

Everything runs through Sionna's `sim_ber` and must be compatible with TensorFlow **graph
execution** — plain eager (`-debug`) breaks grouped/1D convs on CPU.

## Environment

Two supported stacks (do not mix):
- **Recommended:** Sionna 0.18, TensorFlow 2.15, Python 3.11, Ubuntu 24.04 (GPU).
- **aarch64 / Python 3.12 fallback:** TensorFlow 2.16.2 + `tf-keras` with
  `TF_USE_LEGACY_KERAS=1`, Sionna 0.19.2. **CPU-only** on this arch (no CUDA TF wheel exists for
  GB10/Blackwell aarch64). `train_cnn.py` auto-sets `TF_USE_LEGACY_KERAS=1` when `tf_keras` is
  importable; the other scripts need it exported manually.

There is no build/lint/test suite. The closest thing to a test is `scripts/verify_ara_cdl.py`
(self-contained end-to-end sanity check: run → identity-at-init → short train → BER).

## Commands

All commands run from `scripts/`. Prefix with `TF_USE_LEGACY_KERAS=1` on the aarch64 fallback stack.

```bash
# Train MDX (or nrx) for a config
python3 train_neural_rx.py -system mdx -config_name mdx_res_blocks2_var_mcs_it1_ext.cfg -gpu 0

# Evaluate BLER/BER; picks method(s) via -methods, auto-loads matching weights
python3 evaluate.py -config_name ara_cdl_16x4.cfg -gpu 0 -methods mdx \
    -channel_type_eval CDL-C -num_tx_eval 1 -n_size_bwp_eval 4 -mcs_arr_eval_idx 0
python3 evaluate.py --help          # full flag list

# Train the ARA / CNN channel estimators (supervised denoising toward true channel)
python3 train_ara.py -config_name ara_cdl_16x4.cfg -channel_type CDL-C -num_steps 1500 -batch_size 16
python3 train_cnn.py -config_name ara_cdl_16x4.cfg -channel_type CDL-C -num_steps 1500 -batch_size 16

# One-shot sanity check
python3 verify_ara_cdl.py
```

The `eval_*/` directories hold ready-made `.sh` / `.sbatch` driver scripts (per-GPU, per-MCS SNR
sweeps) plus their `*_results` output files and Jupyter notebooks for plotting. Drivers accept a
`PYTHON=/path/to/venv/bin/python` override and are meant to be launched from `scripts/`, e.g.
`../eval_4x2_tdla/run_mdx_ext.sh`.

## Architecture

**Config-driven.** A run is defined entirely by a `.cfg` file in `config/` (parsed by
`configparser` with `ast.literal_eval` values) plus a chosen `system` string. `utils/parameters.py`
`Parameters` reads the config, builds the Sionna PUSCH transmitter/channel, and branches on
`self.system` to wire up the right receiver. `re_init()` re-parameterizes an existing `Parameters`
for evaluation (different channel type, bandwidth, num users) **without rebuilding the graph** —
this is why one trained model evaluates across configs.

**The `system` string is the central dispatch.** It selects which receiver `E2E_Model`
(`utils/e2e_model.py`) instantiates and how it routes the `[y, no]` call. Values include:
`mdx`, `nrx`, and the `baseline_*` family — `baseline_perf_csi_{lmmse,kbest}`,
`baseline_lmmse_{lmmse,kbest}`, `baseline_ls{lin,nn}_lmmse`, `baseline_lslin_kbest`, and the
learned-estimator baselines `baseline_ara_{lmmse,kbest}`, `baseline_cnn[_nofft]_{lmmse,kbest}`.
When adding a receiver/estimator you must touch three places consistently: the `system` branches in
`e2e_model.py` (construction + call routing), the branch in `parameters.py`, and the receiver class.

**Receiver implementations:**
- `utils/md_rx.py` — `MDNeuralPUSCHReceiver`, the MDX model (the main contribution; largest file).
- `utils/neural_rx.py` — `NeuralPUSCHReceiver`, the upstream NRX comparison model.
- `utils/baseline_rx.py` — `BaselineReceiver`; classical LS/LMMSE/K-best chains, and the
  insertion point where a learned estimator (ARA/CNN) replaces the classical channel estimator.
- `utils/ara_estimator.py` / `utils/cnn_estimator.py` — the two learned estimators, each a
  Sionna-compatible `*ChannelEstimator` wrapping a Keras network. Both are **identity-at-init**
  (final read-out conv zero-initialised) so an untrained estimator reproduces the LS+linear
  estimate exactly and never degrades the baseline before training.

**Channel models:** `utils/parameters.py` selects TDL (`utils/tdl.py`) or 3GPP CDL-A…E
(`utils/channel_models.py`) via `channel_type` / `channel_type_eval`.

**Shared infra in `utils/utils.py`:** `training_loop()` (the actual training step / XLA eval loop
used by the trainers), `save_weights`/`load_weights`, and the plotting/CSV/goodput helpers the eval
notebooks call. `channel_to_freqspace()` (in `ara_estimator.py`) is the layout helper mapping a
channel tensor to the `(space = num_rx_ant, freq = num_subcarriers)` plane the estimators denoise.

## Weights convention

Trained weights live in `weights/` and are **discovered by naming convention**, not passed
explicitly. The label comes from `label` in the `[global]` config section (optionally plus a
`-name_suffix`):
- NRX/MDX → `weights/<label>_weights` (h5).
- ARA → `weights/<label>_ara_weights`, CNN → `weights/<label>_cnn_weights` (pickled
  `get_weights()` list). `_nofft` variants get their own suffix.

`evaluate.py` auto-loads the file matching `<label>` + method. If no learned-estimator weights are
found it silently falls back to the identity (LS+linear) estimator, so evaluation always runs — a
"baseline-looking" result may mean weights were simply missing. **Train the estimator before
evaluating its `baseline_*` method.**

## Provenance

`scripts/` and `utils/{neural_rx,e2e_model,parameters,utils}.py` are modified NVIDIA NRX code
(NvidiaProprietary license headers). MDX (`md_rx.py`) and the ARA/CNN estimators are the additions
in this repo. Keep the license headers intact when editing NRX-derived files.
