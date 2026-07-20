# 16x4 CDL-C evaluation — classical baseline vs deep-learned ARA estimator

Evaluates two PUSCH receivers on a **3GPP CDL-C** channel with **16 receive
antennas** (`config/ara_cdl_16x4.cfg`), and plots BLER/BER vs SNR:

| System | `-methods` | Channel estimator |
|---|---|---|
| Baseline receiver | `baseline_lslin_lmmse` | LS + linear interpolation |
| ARA receiver | `baseline_ara_lmmse` | LS + linear interpolation + **ARA** denoising |

Both use the same LMMSE MIMO detector, so the curves isolate the effect of the
deep-learned [ARA channel estimator](../utils/ara_estimator.py).

## Files

| File | Purpose |
|---|---|
| `train_ara_16x4_cdlc.sh` | trains the ARA network → `../weights/ara_cdl_16x4_ara_weights` |
| `run_eval_16x4_cdlc.sh` | single-user run, both receivers → `ara_cdl_16x4_results` (pickle) |
| `plot_bler_16x4_cdlc.py` | plots `16x4_cdlc_bler.png` and `16x4_cdlc_ber.png` |
| `run_eval_16x4_cdlc_mu4.sh` | **4 active users**, MCS 9/14/19 → `ara_cdl_16x4_mu_results` |
| `plot_bler_16x4_cdlc_mu4.py` | plots `bler_16x4_cdlc_mu4.png` (one panel per MCS) |

## Multi-user (4 active users, MCS 9 / 14 / 19)

Sionna's `CDL` models a **single link**, so multi-user CDL is provided by
`utils/channel_models.py:MultiUserCDLChannel`, which stacks one `CDL` per user
and concatenates along the `num_tx` axis (mirroring `NTDLChannel`). It is used
automatically for `channel_type = CDL-*` whenever `max_num_tx > 1`.

Config `config/ara_cdl_16x4_mu.cfg`: 16 rx antennas, 4 single-antenna UEs,
`mcs_index = [9, 14, 19]`.

```bash
cd mdx/scripts
PYTHON=/home/alan/sionna-env-018/bin/python ../eval_16x4_cdlc/run_eval_16x4_cdlc_mu4.sh
cd ../eval_16x4_cdlc
/home/alan/sionna-env-018/bin/python plot_bler_16x4_cdlc_mu4.py
```

The ARA network is fully convolutional/attention-based over the frequency-space
plane, so the weights trained for this setup are independent of the MCS and of
the number of users — `weights/ara_cdl_16x4_mu_ara_weights` reuses the trained
ARA directly.

## LS vs CNN vs ARA comparison (channel-estimation NMSE + BLER)

Compares three channel estimators, all feeding the **same LMMSE equalizer**:

| Estimator | System | Description |
|---|---|---|
| **LS** | `baseline_lslin_lmmse` | LS despreading + linear interpolation (classical) |
| **CNN** | `baseline_cnn_lmmse` | LS+lin refined by a 1D delay-domain residual CNN (TF port of NVIDIA pyAerial `channel_estimation`) |
| **ARA** | `baseline_ara_lmmse` | LS+lin refined by the attentive residual autoencoder |

Both learned estimators are residual refinements of the same LS+lin input
(zero-initialised read-out ⇒ identity at init), so the comparison is fair.

Files:

| File | Purpose |
|---|---|
| `../scripts/train_cnn.py` | trains the CNN (AdamW + NMSE loss, pyAerial recipe) |
| `compute_nmse_16x4_cdlc.py` | channel-estimation NMSE vs SNR for LS/CNN/ARA → `nmse_16x4_cdlc_results` |
| `plot_nmse_16x4_cdlc.py` | plots `nmse_16x4_cdlc.png` |
| `run_eval_16x4_cdlc_cnn_mu4.sh` | adds CNN BLER curves (4 users, MCS 9/14/19) into `ara_cdl_16x4_mu_results` |
| `plot_bler_16x4_cdlc_compare.py` | plots `bler_16x4_cdlc_compare.png` (LS/CNN/ARA per MCS) |

```bash
cd mdx/scripts
PY=/home/alan/sionna-env-018/bin/python

# 1) Train the CNN (LS/ARA already trained). Same SNR range as ARA.
$PY train_cnn.py -config_name ara_cdl_16x4.cfg -channel_type CDL-C \
    -num_steps 1500 -batch_size 16 -snr_db_min -12 -snr_db_max 2
# reuse the weights for the 4-user (mu) label:
cp ../weights/ara_cdl_16x4_cnn_weights ../weights/ara_cdl_16x4_mu_cnn_weights

# 2) Channel-estimation NMSE vs SNR (LS/CNN/ARA)
PYTHON=$PY $PY ../eval_16x4_cdlc/compute_nmse_16x4_cdlc.py
$PY ../eval_16x4_cdlc/plot_nmse_16x4_cdlc.py

# 3) Add CNN BLER curves (LS + ARA already in the results file)
PYTHON=$PY ../eval_16x4_cdlc/run_eval_16x4_cdlc_cnn_mu4.sh
$PY ../eval_16x4_cdlc/plot_bler_16x4_cdlc_compare.py
```

## Usage

Run the shell scripts **from the `scripts/` directory** (they call `evaluate.py`
/ `train_ara.py`, which resolve `../config` and `../weights` relatively). Set
`PYTHON` to your venv interpreter:

```bash
cd mdx/scripts

# 1) Train the ARA estimator (required for a meaningful ARA curve)
PYTHON=/home/alan/sionna-env-018/bin/python ../eval_16x4_cdlc/train_ara_16x4_cdlc.sh

# 2) Evaluate both receivers; results are written into this folder
PYTHON=/home/alan/sionna-env-018/bin/python ../eval_16x4_cdlc/run_eval_16x4_cdlc.sh

# 3) Plot BLER / BER vs SNR
cd ../eval_16x4_cdlc
/home/alan/sionna-env-018/bin/python plot_bler_16x4_cdlc.py
```

Both methods are evaluated in a single `evaluate.py` call, so their curves land
in one results file (`ara_cdl_16x4_results`) keyed by system name.

## Notes

- If the ARA weights are missing, the ARA estimator falls back to its
  **identity initialisation** and reproduces the LS+linear baseline exactly —
  the two curves would then overlap. Train first.
- **Do not pass `-debug`** on CPU: it enables eager execution, which does not
  support the grouped convolutions used by the neural receivers.
- Settings here (`n_size_bwp_eval=4`, modest `max_mc_iter`) are sized for a
  CPU-only run. Increase `n_size_bwp_eval`, `batch_size_eval`,
  `max_mc_iter` and `num_target_block_errors` for smoother, lower-BLER curves.
- CDL in Sionna is single-UT, so this setup evaluates 1 active UE
  (`-num_tx_eval 1`) with 16 BS antennas. For multi-user runs use a multi-link
  channel (`NTDLlow`, `DoubleTDLlow`, `UMi`/`UMa`).
