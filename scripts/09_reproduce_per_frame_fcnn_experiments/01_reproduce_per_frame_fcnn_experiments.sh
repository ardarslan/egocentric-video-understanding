#!/bin/bash

helpFunction()
{
   echo ""
   echo "Usage: $0 -n NUM_NONLINEAR_LAYERS -c CONFIG_FILE_NAME_WO_EXT"
   echo -e "\t-n One of 0,1,2."
   echo -e "\t-c One of ego4d_asl_features,proposed_features_v1,proposed_features_v2,proposed_featues_v3,proposed_features_v4,proposed_features_v5,proposed_features_v6."
   exit 1 # Exit script after printing help
}

while getopts "n:c:" opt
do
   case "$opt" in
      n ) NUM_NONLINEAR_LAYERS="$OPTARG" ;;
      c ) CONFIG_FILE_NAME_WO_EXT="$OPTARG" ;;
      ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
   esac
done

# Print helpFunction in case parameters are empty
if [ -z "$NUM_NONLINEAR_LAYERS" ] || [ -z "$CONFIG_FILE_NAME_WO_EXT" ]
then
   echo "Some or all of the parameters are empty";
   helpFunction
fi

python3 01_reproduce_per_frame_fcnn_experiments.py --num_nonlinear_layers $NUM_NONLINEAR_LAYERS --config_file_name_wo_ext $CONFIG_FILE_NAME_WO_EXT
