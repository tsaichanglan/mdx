#!/bin/bash

# Train the deep-learned ARA channel estimator for the 16x4 / CDL-C setup.
# Must be run BEFORE run_eval_16x4_cdlc.sh, which loads the resulting weights
# from ../weights/ara_cdl_16x4_ara_weights.
#
# Run from the scripts/ directory, e.g.
#   cd mdx/scripts
#   PYTHON=/home/alan/sionna-env-018/bin/python ../eval_16x4_cdlc/train_ara_16x4_cdlc.sh

PYTHON="${PYTHON:-python3}"

config_name="ara_cdl_16x4.cfg"   # 16 rx antennas, CDL, single UE port
channel_type="CDL-C"
gpu=0

num_steps=1500
batch_size=16
learning_rate=0.001
n_size_bwp=4
num_tx=1
snr_db_min=-12.0
snr_db_max=2.0
max_ut_velocity=3.0

echo "Training ARA on ${channel_type} (16 rx antennas) ..."

$PYTHON train_ara.py -config_name="${config_name}" -gpu="${gpu}" \
    -channel_type="${channel_type}" -num_tx="${num_tx}" \
    -n_size_bwp="${n_size_bwp}" -num_steps="${num_steps}" \
    -batch_size="${batch_size}" -learning_rate="${learning_rate}" \
    -snr_db_min="${snr_db_min}" -snr_db_max="${snr_db_max}" \
    -max_ut_velocity="${max_ut_velocity}" -eval_every=250

echo "Finished ARA training."
exit 0
