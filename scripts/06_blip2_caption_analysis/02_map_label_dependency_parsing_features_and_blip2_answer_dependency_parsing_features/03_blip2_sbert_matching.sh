#!/bin/bash


helpFunction()
{
   echo ""
   echo "Usage: $0 -q QUARTER_INDEX"
   echo "Usage: $0 -c CUDA_VISIBLE_DEVICES"
   echo "Usage: $0 -b BACKBONE"
   echo -e "\t-q One of 0, 1, 2, 3."
   echo -e "\t-c e.g. 4,5,6,7"
   echo -e "\t-b One of sentence-transformers/paraphrase-MiniLM-L6-v2, sentence-transformers/all-distilroberta-v1"
   exit 1 # Exit script after printing help
}

while getopts "q:c:b:" opt
do
   case "$opt" in
      q ) QUARTER_INDEX="$OPTARG" ;;
      c ) CUDA_VISIBLE_DEVICES="$OPTARG" ;;
      b ) BACKBONE="$OPTARG" ;;
      ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
   esac
done

# Print helpFunction in case parameters are empty
if [ -z "$QUARTER_INDEX" ] || [ -z "$CUDA_VISIBLE_DEVICES" ] || [ -z "$BACKBONE" ]
then
   echo "Some or all of the parameters are empty";
   helpFunction
fi

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python3 03_blip2_sbert_matching.py --quarter_index $QUARTER_INDEX --backbone $BACKBONE
