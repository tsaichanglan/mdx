# Deep-Learned ARA Channel Estimator — Results & Workflow

Implementation of the Attentive Residual Autoencoder (ARA) channel estimator
(Y. Wei *et al.*, "Deep-Learned Channel Estimation for MIMO-OFDM System by
Exploiting Frequency-Space Correlation", *Electronics Letters*, 2026) as a
drop-in replacement for the classical channel estimator in `BaselineReceiver`,
verified on 3GPP CDL channels.

## What was added

| File | Purpose |
|------|---------|
| `utils/ara_estimator.py` | ARA network (`ARANetwork`) + Sionna-compatible `ARAChannelEstimator`; shared `channel_to_freqspace()` layout helper |
| `utils/baseline_rx.py` | `baseline_ara_lmmse` / `baseline_ara_kbest` systems wired into the receiver |
| `utils/e2e_model.py` | receiver branches + `[y, no]` call routing for the ARA systems |
| `utils/parameters.py` | 3GPP `CDL-A`…`CDL-E` channel models |
| `config/ara_cdl.cfg` | single-user, single-port CDL config (ground-truth aligns with estimator layout) |
| `scripts/train_ara.py` | supervised training of the ARA network; saves `weights/<label>_ara_weights` |
| `scripts/evaluate.py` | `baseline_ara_lmmse` block auto-loads ARA weights and runs `sim_ber`; CPU-safe GPU guard |
| `scripts/verify_ara_cdl.py` | self-contained sanity check (run / identity-init / train / BER) |

## Design notes

- **Three-stage scheme (paper Fig. 2):** LS despreading → linear interpolation
  (inherited from `PUSCHLSChannelEstimator`) → ARA denoising over the
  frequency-space plane `(num_rx_ant = space, num_subcarriers = freq)`.
- **ARA network (paper Figs. 3–4):** autoencoder with Res / Res-down (FIR +
  strided conv) / Res-up (upsample + FIR + conv) modules, 2D self-attention
  (encoder) and cross-attention (decoder), skip connections, multi-scale
  fusion, GroupNorm-per-4-channels, residual read-out. ~305k trainable params.
- **Identity-at-init:** the residual read-out conv is zero-initialised, so an
  untrained ARA reproduces the LS+linear estimate exactly — evaluation can
  never degrade below the LS+linear baseline before training.

## Measured results (CPU, `~/sionna-env`)

Environment: Python 3.12 / aarch64, TensorFlow 2.16.2 + `tf-keras`
(`TF_USE_LEGACY_KERAS=1`), Sionna 0.19.2. Channel: CDL-C, 1 user, QPSK,
`n_size_bwp = 4`.

### Training (`train_ara.py`, 300-step demo)

| Step | train MSE | val NMSE | gain over LS+linear |
|-----:|----------:|---------:|--------------------:|
| init | — | 0.2111 | 0.00 dB (identity) |
| 100  | 0.0256 | 0.1339 | +1.98 dB |
| 200  | 0.0205 | 0.0881 | +3.80 dB |
| 300  | 0.0275 | 0.0647 | **+5.14 dB** |

(A separate 400-step run inside `verify_ara_cdl.py` gave NMSE 0.0374 → 0.0120,
+4.92 dB, and BER better than LS+linear at 0/4 dB.)

### Evaluation (`evaluate.py -methods baseline_ara_lmmse`, weights auto-loaded)

| Eb/N0 [dB] | BER | BLER |
|-----------:|----:|-----:|
| −2.0 | 2.16e-02 | 1.41e-01 |
| 2.0 | 4.26e-03 | 3.13e-02 |
| 6.0 | 0.00e+00 | 0.00e+00 |

> These are from a lightly-trained (300-step, 4-PRB) demo model. Retrain with
> more steps and larger `n_size_bwp` for publication-quality curves.

## Reproduce

```bash
cd scripts

# Train
TF_USE_LEGACY_KERAS=1 python3 train_ara.py -config_name ara_cdl.cfg \
    -channel_type CDL-C -num_steps 3000 -batch_size 32

# Evaluate (auto-loads ../weights/ara_cdl_ara_weights)
TF_USE_LEGACY_KERAS=1 python3 evaluate.py -config_name ara_cdl.cfg \
    -methods baseline_ara_lmmse -channel_type_eval CDL-C \
    -num_tx_eval 1 -n_size_bwp_eval 4 -mcs_arr_eval_idx 0

# One-shot sanity check
TF_USE_LEGACY_KERAS=1 python3 verify_ara_cdl.py
```

## GPU status

GPU-accelerated TensorFlow is **not available** on this DGX Spark (GB10 /
Blackwell / aarch64): PyPI TensorFlow aarch64 wheels are CPU-only
(`is_cuda_build=False`) and no CUDA TensorFlow build exists for this arch
(NVIDIA's `sbsa/cu130` index builds CUDA PyTorch but only proxies PyPI's CPU
TensorFlow). All results above are CPU. Decision: stay on CPU.
