#!/bin/bash

CUDA_VISIBLE_DEVICES=0,4 python3 01_extract_frame_features.py --frame_feature_name blip2_vqa --quarter_index 0
CUDA_VISIBLE_DEVICES=1,5 python3 01_extract_frame_features.py --frame_feature_name blip2_vqa --quarter_index 1
CUDA_VISIBLE_DEVICES=2,6 python3 01_extract_frame_features.py --frame_feature_name blip2_vqa --quarter_index 2
CUDA_VISIBLE_DEVICES=3,7 python3 01_extract_frame_features.py --frame_feature_name blip2_vqa --quarter_index 3
