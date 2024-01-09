#!/bin/bash

helpFunction()
{
   echo ""
   echo "Usage: $0 -p PREDICTIONS_FOLDER_NAME"
   echo "Usage: $0 -t TEMPORAL_AGGREGATION"
   echo "Usage: $0 -h THRESHOLD"
   echo "Usage: $0 -s SPLIT"
   echo -e "\t-p One of
   'asl_max_per_label_predictions',
   'proposed_features_v2_max_per_label_predictions',
   'proposed_features_v5_max_per_label_predictions',
   'blip2_dictionary_matching_max_per_label_predictions',
   'blip2_sbert_matching_all-distilroberta-v1_max_per_label_predictions',
   'blip2_sbert_matching_paraphrase-MiniLM-L6-v2_max_per_label_predictions'"
   echo -e "\t-t One of
   'no_temporal_aggregation',
   'transfusion_temporal_aggregation',
   'median_temporal_aggregation'"
   echo -e "\t-h e.g.
   'e.g. 0.2, 0.4, 0.6, 0.8, 1.0'"
   echo -e "\t-s e.g.
   'One of train, val, test'"
   exit 1 # Exit script after printing help
}

while getopts "p:t:h:s:" opt
do
   case "$opt" in
      p ) PREDICTIONS_FOLDER_NAME="$OPTARG" ;;
      t ) TEMPORAL_AGGREGATION="$OPTARG" ;;
      h ) THRESHOLD="$OPTARG" ;;
      s ) SPLIT="$OPTARG" ;;
      ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
   esac
done

# Print helpFunction in case parameters are empty
if [ -z "$PREDICTIONS_FOLDER_NAME" ] || [ -z "$TEMPORAL_AGGREGATION" ] || [ -z "$THRESHOLD" ] || [ -z "$SPLIT" ]
then
   echo "Some or all of the parameters are empty";
   helpFunction
fi

python3 05_evaluate_predictions.py --predictions_folder_name $PREDICTIONS_FOLDER_NAME --temporal_aggregation $TEMPORAL_AGGREGATION --threshold $THRESHOLD --split $SPLIT
