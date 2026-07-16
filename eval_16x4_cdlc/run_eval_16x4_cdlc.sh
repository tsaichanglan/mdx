#!/bin/bash

# Evaluate the classical baseline receiver (LS/lin + LMMSE) and the
# deep-learned ARA estimator receiver (ARA + LMMSE) on a 3GPP CDL-C channel
# with 16 receive antennas.
#
# Both methods share the same config, so their curves are stored in a single
# results file:  <this dir>/ara_cdl_16x4_results
# keyed by system name, and can be plotted with plot_bler_16x4_cdlc.py.
#
# Train the ARA first (train_ara_16x4_cdlc.sh), otherwise the ARA estimator
# falls back to its identity initialisation (== LS/lin baseline).
#
# Run from the scripts/ directory, e.g.
#   cd mdx/scripts
#   PYTHON=/home/alan/sionna-env-018/bin/python ../eval_16x4_cdlc/run_eval_16x4_cdlc.sh
#
# NOTE: do NOT add -debug on CPU: eager mode does not support the grouped
# convolutions used by the neural receivers.

PYTHON="${PYTHON:-python3}"

config_name="ara_cdl_16x4.cfg"   # 16 rx antennas, CDL, single UE port, QPSK
gpu=0

max_mc_iter=200
num_target_block_errors=200
target_bler=0.0001

snr_db_eval_min=-12
snr_db_eval_max=2
snr_db_eval_stepsize=1
max_ut_velocity_eval=3.
channel_type_eval="CDL-C"
n_size_bwp_eval=4
batch_size_eval=32
batch_size_eval_small=8

dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)/
echo "eval@: ${dir}"

name_suffix=""

# -----------------------------------------------------------------------
# Baseline receiver (LS/lin + LMMSE) and ARA estimator receiver (ARA + LMMSE)
# Both are evaluated in one call so they land in the same results file.
# -----------------------------------------------------------------------
$PYTHON evaluate.py -config_name="${config_name}" -gpu="${gpu}" \
    -mcs_arr_eval_idx 0 -num_tx_eval 1 \
    -num_target_block_errors="${num_target_block_errors}" \
    -max_mc_iter="${max_mc_iter}" -target_bler="${target_bler}" \
    -snr_db_eval_min="${snr_db_eval_min}" -snr_db_eval_max="${snr_db_eval_max}" \
    -snr_db_eval_stepsize="${snr_db_eval_stepsize}" \
    -max_ut_velocity_eval="${max_ut_velocity_eval}" \
    -channel_type_eval="${channel_type_eval}" \
    -n_size_bwp_eval="${n_size_bwp_eval}" \
    -batch_size_eval="${batch_size_eval}" \
    -batch_size_eval_small="${batch_size_eval_small}" \
    -dir="${dir}" -name_suffix="${name_suffix}" \
    -methods baseline_lslin_lmmse baseline_ara_lmmse

echo "Finished evaluation. Results in ${dir}"
exit 0
