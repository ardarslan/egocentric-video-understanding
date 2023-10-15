#!/bin/bash

helpFunction()
{
   echo ""
   echo "Usage: $0 -p PREDICTION_METHOD"
   echo -e "\t-p One of asl, blip2_dictionary_matching, blip2_sbert_matching"
   exit 1 # Exit script after printing help
}

while getopts "p:" opt
do
   case "$opt" in
      p ) PREDICTION_METHOD="$OPTARG" ;;
      ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
   esac
done

# Print helpFunction in case parameters are empty
if [ -z "$PREDICTION_METHOD" ]
then
   echo "Prediction method parameter -p is empty";
   helpFunction
fi

python3 05_evaluate_predictions.py --prediction_method $PREDICTION_METHOD
