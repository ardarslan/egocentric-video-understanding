#!/bin/bash

helpFunction()
{
   echo ""
   echo "Usage: $0 -f FRAME_FEATURE_NAME -q QUARTER_INDEX -c CUDA_VISIBLE_DEVICES "
   echo -e "\t-f One of unidet, ofa, visor_hos, ego_hos, blip_vqa, blip_captioning or gsam."
   echo -e "\t-q One of 0, 1, 2 or 3."
   echo -e "\t-c e.g. 0,1,2,3"
   exit 1 # Exit script after printing help
}

while getopts "f:q:c:" opt
do
   case "$opt" in
      f ) FRAME_FEATURE_NAME="$OPTARG" ;;
      q ) QUARTER_INDEX="$OPTARG" ;;
      c ) CUDA_VISIBLE_DEVICES="$OPTARG" ;;
      ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
   esac
done

# Print helpFunction in case parameters are empty
if [ -z "$FRAME_FEATURE_NAME" ] || [ -z "$QUARTER_INDEX" ] || [ -z "$CUDA_VISIBLE_DEVICES" ]
then
   echo "Some or all of the parameters are empty";
   helpFunction
fi

export TMPDIR="$CODE/tmp"
cd $CODE/scripts/04_extract_frame_features
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python3 main.py --frame_feature_name $FRAME_FEATURE_NAME --quarter_index $QUARTER_INDEX
