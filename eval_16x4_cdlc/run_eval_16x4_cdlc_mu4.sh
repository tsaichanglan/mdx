#!/bin/bash

# 16x4 MU-MIMO on CDL-C: 4 active users, MCS 9 / 14 / 19.
# Compares the classical baseline receiver (LS/lin + LMMSE) against the
# deep-learned ARA estimator receiver (ARA + LMMSE), using the ARA weights
# trained by train_ara_16x4_cdlc.sh.
#
# Sionna's CDL models a single link, so the multi-user case is served by
# utils/channel_models.py:MultiUserCDLChannel, which stacks one CDL per user.
#
# Results (both methods, all MCS) land in a single file:
#   <this dir>/ara_cdl_16x4_mu_results     keyed by (system, num_tx, mcs_idx)
# Plot with plot_bler_16x4_cdlc_mu4.py
#
# Run from the scripts/ directory:
#   cd mdx/scripts
#   PYTHON=/home/alan/sionna-env-018/bin/python ../eval_16x4_cdlc/run_eval_16x4_cdlc_mu4.sh
#
# NOTE: do NOT add -debug on CPU (eager mode breaks grouped convolutions).

PYTHON="${PYTHON:-python3}"

config_name="ara_cdl_16x4_mu.cfg"   # 16 rx ant, 4 single-antenna UEs, MCS [9,14,19]
gpu=0

num_tx_eval=4
max_mc_iter=40
num_target_block_errors=150
target_bler=0.001

max_ut_velocity_eval=3.
channel_type_eval="CDL-C"
n_size_bwp_eval=4
batch_size_eval=16
batch_size_eval_small=8
snr_db_eval_stepsize=1

dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)/
echo "eval@: ${dir}"

methods="baseline_lslin_lmmse baseline_ara_lmmse"

# Per-MCS SNR ranges (chosen to bracket the BLER transition for 4 users)
#            MCS idx:  0 (MCS 9)   1 (MCS 14)   2 (MCS 19)
snr_min=(   -13        -12          -8 )
snr_max=(     5          5          10 )

for mcs_idx in 0 1 2; do
    echo "=============== MCS array index ${mcs_idx} (4 users) ==============="
    $PYTHON evaluate.py -config_name="${config_name}" -gpu="${gpu}" \
        -mcs_arr_eval_idx "${mcs_idx}" -num_tx_eval "${num_tx_eval}" \
        -num_target_block_errors="${num_target_block_errors}" \
        -max_mc_iter="${max_mc_iter}" -target_bler="${target_bler}" \
        -snr_db_eval_min="${snr_min[$mcs_idx]}" \
        -snr_db_eval_max="${snr_max[$mcs_idx]}" \
        -snr_db_eval_stepsize="${snr_db_eval_stepsize}" \
        -max_ut_velocity_eval="${max_ut_velocity_eval}" \
        -channel_type_eval="${channel_type_eval}" \
        -n_size_bwp_eval="${n_size_bwp_eval}" \
        -batch_size_eval="${batch_size_eval}" \
        -batch_size_eval_small="${batch_size_eval_small}" \
        -dir="${dir}" -methods ${methods}
done

echo "Finished 4-user MCS sweep. Results in ${dir}"
exit 0
