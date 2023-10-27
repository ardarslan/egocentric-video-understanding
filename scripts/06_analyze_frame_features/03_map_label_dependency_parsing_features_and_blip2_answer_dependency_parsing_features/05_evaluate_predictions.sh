#!/bin/bash

helpFunction()
{
   echo ""
   echo "Usage: $0 -p PREDICTIONS_FOLDER_NAME"
   echo -e "\t-p One of
   'blip2_dictionary_matching_max_per_label_predictions',
   'blip2_sbert_matching_all-distilroberta-v1_max_per_label_predictions',
   'blip2_sbert_matching_paraphrase-MiniLM-L6-v2_max_per_label_predictions'"
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
   echo "Some or all of the parameters are empty";
   helpFunction
fi

python3 05_evaluate_predictions.py --predictions_folder_name $PREDICTIONS_FOLDER_NAME
