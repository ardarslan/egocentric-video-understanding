#!/bin/bash

sleep_if_necessary() {
    if [ $(expr $(($NUM_JOBS+1)) % 100) == "0" ]; then
        read -p "Number of submitted jobs: $NUM_JOBS. Sleeping for $SLEEP_TIME seconds..." -t $SLEEP_TIME
    fi
}

for NUM_NONLINEAR_LAYERS in (0 1 2); do
    for CONFIG_FILE_NAME_WO_EXT in ("ego4d_asl_features" "proposed_features_v2" "proposed_features_v5"); do
        sbatch --time 720 --gres=gpu:1 --cpus-per-task=1 --mem 60G  01_reproduce_per_frame_fcnn_experiments.sh --num_nonlinear_layers $NUM_NONLINEAR_LAYERS --config_file_name_wo_ext $CONFIG_FILE_NAME_WO_EXT
