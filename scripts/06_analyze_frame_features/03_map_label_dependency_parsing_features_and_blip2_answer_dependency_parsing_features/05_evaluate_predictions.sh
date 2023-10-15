#!/bin/bash

helpFunction()
{
   echo ""
   echo "Usage: $0 -p PREDICTIONS_FOLDER_NAME"
   echo -e "\t-p One of
   'blip2_dictionary_matching_max_per_label_predictions',
   'blip2_sbert_matching_max_per_label_predictions',
   'asl_max_per_label_predictions',
   'blip2_dictionary_matching_max_per_label_transfusion_predictions',
   'blip2_sbert_matching_max_per_label_transfusion_predictions',
   'asl_max_per_label_transfusion_predictions',
   'blip2_dictionary_matching_max_per_label_mode_filter_predictions',
   'blip2_sbert_matching_max_per_label_mode_filter_predictions',
   'asl_max_per_label_mode_filter_predictions',
   'blip2_dictionary_matching_max_per_label_median_filter_predictions',
   'blip2_sbert_matching_max_per_label_median_filter_predictions',
   'asl_max_per_label_median_filter_predictions'"
   exit 1 # Exit script after printing help
}

while getopts "p:" opt
do
   case "$opt" in
      p ) PREDICTIONS_FOLDER_NAME="$OPTARG" ;;
      ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
   esac
done

# Print helpFunction in case parameters are empty
if [ -z "$PREDICTIONS_FOLDER_NAME" ]
then
   echo "Predictions folder name parameter -p is empty";
   helpFunction
fi

python3 05_evaluate_predictions.py --predictions_folder_name $PREDICTIONS_FOLDER_NAME
