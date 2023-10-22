#!/bin/bash


helpFunction()
{
   echo ""
   echo "Usage: $0 -q QUARTER_INDEX"
   echo -e "\t-q One of 0, 1, 2, 3."
   exit 1 # Exit script after printing help
}

while getopts "q:" opt
do
   case "$opt" in
      q ) QUARTER_INDEX="$OPTARG" ;;
      ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
   esac
done

# Print helpFunction in case parameters are empty
if [ -z "$QUARTER_INDEX" ]
then
   echo "Quarter index parameter -q is empty";
   helpFunction
fi

python3 03_blip2_sbert_matching.py --quarter_index $QUARTER_INDEX