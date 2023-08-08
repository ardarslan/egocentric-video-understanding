#!/bin/bash

helpFunction()
{
   echo ""
   echo "Usage: $0 -f FRAME_FEATURE_NAME -q QUARTER_INDEX  "
   echo -e "\t-f One of unidet, ofa, visor_hos, ego_hos, blip_vqa, blip_captioning or gsam."
   echo -e "\t-q One of 0, 1, 2 or 3."
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

export TMPDIR="/home/aarslan/mq/tmp"
cd ~/mq/scripts/04_extract_frame_features
mamba deactivate
mamba activate mq
python3 main.py --frame_feature_name $FRAME_FEATURE_NAME --quarter_index $QUARTER_INDEX
