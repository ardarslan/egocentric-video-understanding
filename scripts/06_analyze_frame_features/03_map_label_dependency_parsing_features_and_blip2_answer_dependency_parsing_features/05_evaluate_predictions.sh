#!/bin/bash

helpFunction()
{
   echo ""
   echo "Usage: $0 -p PREDICTIONS_FOLDER_NAME -t TEMPORAL_AGGREGATION"
   echo -e "\t-t One of no_temporal_aggregation, transfusion_temporal_aggregation, median_temporal_aggregation."
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

while getopts "p:t:" opt
do
   case "$opt" in
      p ) PREDICTIONS_FOLDER_NAME="$OPTARG" ;;
      t ) TEMPORAL_AGGREGATION="$OPTARG" ;;
      ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
   esac
done

# Print helpFunction in case parameters are empty
if [ -z "$PREDICTIONS_FOLDER_NAME" ] || [ -z "$TEMPORAL_AGGREGATION" ]
then
   echo "Some or all of the parameters are empty";
   helpFunction
fi

python3 05_evaluate_predictions.py --predictions_folder_name $PREDICTIONS_FOLDER_NAME --temporal_aggregation $TEMPORAL_AGGREGATION
