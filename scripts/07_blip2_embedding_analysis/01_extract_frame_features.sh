#!/bin/bash

helpFunction()
{
   echo ""
   echo "Usage: $0 -f FRAME_FEATURE_NAME -q QUARTER_INDEX"
   echo -e "\t-f One of blip2_vqa, video_blip."
   echo -e "\t-q One of 0, 1, 2, 3, 4, 5"
   exit 1 # Exit script after printing help
}

while getopts "f:q:" opt
do
   case "$opt" in
      f ) FRAME_FEATURE_NAME="$OPTARG" ;;
      q ) QUARTER_INDEX="$OPTARG" ;;
      ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
   esac
done

# Print helpFunction in case parameters are empty
if [ -z "$FRAME_FEATURE_NAME" ] || [ -z "$QUARTER_INDEX" ]
then
   echo "Some or all of the parameters are empty";
   helpFunction
fi

python3 01_extract_frame_features.py --frame_feature_name $FRAME_FEATURE_NAME --quarter_index $QUARTER_INDEX
