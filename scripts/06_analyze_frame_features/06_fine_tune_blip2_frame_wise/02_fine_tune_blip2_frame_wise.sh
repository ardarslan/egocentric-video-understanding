#!/bin/bash

CONDA_OVERRIDE_CUDA=11.7 CUDA_VISIBLE_DEVICES=0,1,2,3 python3 02_fine_tune_blip2_frame_wise.py

