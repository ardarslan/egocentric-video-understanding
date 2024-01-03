#!/bin/bash

helpFunction()
{
   echo ""
   echo "Usage: $0 -q QUARTER_INDEX"
   echo -e "\t-q One of 0,1,2,3,4,5,6,7,8,9."
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
   echo "Some or all of the parameters are empty";
   helpFunction
fi

python3 postprocess_data.py --quarter_index $QUARTER_INDEX
