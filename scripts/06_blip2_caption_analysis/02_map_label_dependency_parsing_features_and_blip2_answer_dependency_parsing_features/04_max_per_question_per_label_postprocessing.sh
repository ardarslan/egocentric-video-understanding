#!/bin/bash

helpFunction()
{
   echo ""
   echo "Usage: $0 -p PREDICTIONS_FOLDER_NAME"
   echo -e "\t-p One of asl_predictions, asl_ego4d_features, proposed_features_v2, proposed_features_v5, blip2_dictionary_matching_predictions, blip2_sbert_matching_all-distilroberta-v1_predictions, blip2_sbert_matching_paraphrase-MiniLM-L6-v2_predictions"
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

python3 04_max_per_question_per_label_postprocessing.py --predictions_folder_name $PREDICTIONS_FOLDER_NAME
