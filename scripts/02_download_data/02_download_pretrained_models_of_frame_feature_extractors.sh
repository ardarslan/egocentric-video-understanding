#!/bin/bash

mkdir $SCRATCH/mq_libs

cd $SCRATCH/mq_libs

# gsam
mkdir gsam

wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth -O gsam/groundingdino_swint_ogc.pth

wget https://huggingface.co/spaces/xinyu1205/Tag2Text/resolve/main/ram_swin_large_14m.pth -O gsam/ram_swin_large_14m.pth

# ego_hos
mkdir -p ego_hos/seg_twohands_ccda

gdown 1TF5VCkWKZG6IRgVpnpiwA2P4GHx-0Drc -O ego_hos/seg_twohands_ccda/best_mIoU_iter_56000.pth

mkdir -p ego_hos/twohands_to_cb_ccda

gdown 1dKnwQYF-3TeyoyfmbFAw8tt1TfhUq4Qj -O ego_hos/twohands_to_cb_ccda/best_mIoU_iter_76000.pth

mkdir -p ego_hos/twohands_cb_to_obj2_ccda

gdown 1JdewAV1XJyR9reVxEVrwwCH-MwQD6Aqy -O ego_hos/twohands_cb_to_obj2_ccda/best_mIoU_iter_32000.pth

# blip2
git clone https://huggingface.co/Salesforce/blip2-opt-2.7b

mv blip2-opt-2.7b blip2

wget https://huggingface.co/Salesforce/blip2-opt-2.7b/resolve/main/pytorch_model-00001-of-00002.bin -O blip2/pytorch_model-00001-of-00002.bin

wget https://huggingface.co/Salesforce/blip2-opt-2.7b/resolve/main/pytorch_model-00002-of-00002.bin -O blip2/pytorch_model-00002-of-00002.bin
