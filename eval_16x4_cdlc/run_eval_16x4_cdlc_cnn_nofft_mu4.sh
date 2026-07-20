#!/bin/bash

# Adds the CNN estimator curves (4 users, MCS 9/14/19) to the existing
# ara_cdl_16x4_mu_results file (which already holds LS/lin and ARA), so the
# LS / CNN / ARA comparison lives in a single results file.
#
# Requires trained CNN weights at ../weights/ara_cdl_16x4_mu_cnn_weights
# (see train_cnn.py). Run from the scripts/ directory:
#   cd mdx/scripts
#   PYTHON=/home/alan/sionna-env-018/bin/python ../eval_16x4_cdlc/run_eval_16x4_cdlc_cnn_mu4.sh
#
# NOTE: no -dont_load -> results are merged into the existing file.
#       no -debug on CPU (grouped/1D convs need graph mode).

PYTHON="${PYTHON:-python3}"

config_name="ara_cdl_16x4_mu.cfg"
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

# same per-MCS SNR ranges used for LS/ARA
snr_min=( -13 -12  -8 )
snr_max=(   5   5  10 )

for mcs_idx in 0 1 2; do
    echo "=============== CNN(no FFT), MCS array index ${mcs_idx} (4 users) ==============="
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
        -dir="${dir}" -methods baseline_cnn_nofft_lmmse
done

echo "Finished CNN(no FFT) 4-user MCS sweep. Merged into ${dir}ara_cdl_16x4_mu_results"
exit 0
