mkdir $SCRATCH/mq_libs

cd $SCRATCH/mq_libs

mkdir unidet

gdown 1C4sgkirmgMumKXXiLOPmCKNTZAc3oVbq -O unidet/Unified_learned_OCIM_R50_6x+2x.pth

mkdir visor_hos

wget https://www.dropbox.com/s/bfu94fpft2wi5sn/model_final_hos.pth?dl=0 -O visor_hos/model_final_hos.pth

git clone https://huggingface.co/OFA-Sys/OFA-huge ofa

wget https://huggingface.co/OFA-Sys/ofa-huge/resolve/main/pytorch_model.bin -O ofa/pytorch_model.bin

wget https://ofa-beijing.oss-cn-beijing.aliyuncs.com/checkpoints/ofa_huge.pt -O ofa/ofa_huge.pt

mkdir blip

wget https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_capfilt_large.pth -O blip/model_base_capfilt_large.pth

wget https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_vqa_capfilt_large.pth -O blip/model_base_vqa_capfilt_large.pth

mkdir gsam

wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth -O gsam/groundingdino_swint_ogc.pth

wget https://huggingface.co/spaces/xinyu1205/Tag2Text/resolve/main/ram_swin_large_14m.pth -O gsam/ram_swin_large_14m.pth

mkdir -p ego_hos/seg_twohands_ccda

gdown 1TF5VCkWKZG6IRgVpnpiwA2P4GHx-0Drc -O ego_hos/seg_twohands_ccda/best_mIoU_iter_56000.pth

mkdir -p ego_hos/twohands_to_cb_ccda

gdown 1dKnwQYF-3TeyoyfmbFAw8tt1TfhUq4Qj -O ego_hos/twohands_to_cb_ccda/best_mIoU_iter_76000.pth

mkdir -p ego_hos/twohands_cb_to_obj2_ccda

gdown 1JdewAV1XJyR9reVxEVrwwCH-MwQD6Aqy -O ego_hos/twohands_cb_to_obj2_ccda/best_mIoU_iter_32000.pth