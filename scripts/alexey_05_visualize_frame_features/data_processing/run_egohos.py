# %%

import argparse
import cv2
import multiprocessing as mp
from multiprocessing.pool import Pool
import numpy as np
import os
from os.path import basename, dirname, isdir, isfile, join
import pickle
from PIL import Image
import shutil
import ssl
import sys
import time
import torch
from tqdm import tqdm
import zipfile

sys.path.append(dirname(dirname(__file__)))

from data_handling.specific.ek100 import *
from data_handling.video_reader import VideoReader
from utils.args import arg_dict_to_list
from utils.globals import *
from utils.imaging import superimpose_colored_mask
from utils.exceptions import ToggleableException
from utils.io import read_pkl

os.chdir(EGOHOS_PATH)

import glob
from mmseg.apis import inference_segmentor, init_segmentor
import mmcv
from skimage.io import imsave
import pdb


def main(arg_dict=None):
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument("--visualization_interval", type=int, default=100)
    parser.add_argument("--generator_videos", action="append", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
        
    args, _ = parser.parse_known_args(arg_dict_to_list(arg_dict))

    if args.output_dir is None:
        args.output_dir = join(HOS_DATA_DIR, "egohos")
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    TWOHANDS_CONFIG_PATH = "./work_dirs/seg_twohands_ccda/seg_twohands_ccda.py"
    TWOHANDS_CHECKPOINT_PATH = "./work_dirs/seg_twohands_ccda/best_mIoU_iter_56000.pth"

    TWOHANDS_TO_CB_CONFIG_PATH = "./work_dirs/twohands_to_cb_ccda/twohands_to_cb_ccda.py"
    TWOHANDS_TO_CB_CHECKPOINT_PATH = "./work_dirs/twohands_to_cb_ccda/best_mIoU_iter_76000.pth"

    TWOHANDS_CB_TO_OBJ_CONFIG_PATH = "./work_dirs/twohands_cb_to_obj1_ccda/twohands_cb_to_obj1_ccda.py"
    TWOHANDS_CB_TO_OBJ_CHECKPOINT_PATH = "./work_dirs/twohands_cb_to_obj1_ccda/best_mIoU_iter_34000.pth"

    # predict hands, then cb, then object

    if args.generator_videos is None:
        args.generator_videos = get_video_list()
    else:
        args.generator_videos = [s.strip() for v in args.generator_videos for s in v.split(",")]

    print()
    print("Processing step 1/3: hands")
    print()

    twohands_png_paths_videos = {}
    twohands_model = init_segmentor(TWOHANDS_CONFIG_PATH, TWOHANDS_CHECKPOINT_PATH, device=args.device)

    hands_png_path = join(args.output_dir, "pred_twohands")
    hands_pkl_path = join(args.output_dir, "hands")
    
    global_frame_idx = 0
    for video_idx, video_id in enumerate(args.generator_videos):
        reader = VideoReader(get_video_path(video_id), get_extracted_frame_dir_path(video_id),
                             assumed_fps=EK_ASSUMED_FPS)
        twohands_png_paths_videos[video_id] = []
        os.makedirs(join(hands_png_path, video_id), exist_ok=True)
        os.makedirs(join(hands_pkl_path, video_id, "pkls"), exist_ok=True)
        os.makedirs(join(hands_pkl_path, video_id, "jpgs"), exist_ok=True)
        
        try:
            range_obj = range(reader.get_virtual_frame_count())
            
            for frame_idx in tqdm(range_obj):
                frame_id = fmt_frame(video_id, frame_idx)
                seg_result_png_path = join(hands_png_path, f"{frame_id}.png")
                pkl_out_path = join(hands_pkl_path, video_id, "pkls", f"{frame_id}.pkl")
                if isfile(pkl_out_path + ".zip"):
                    # need to temporarily save result as PNG (framework set up this way)
                    img = Image.fromarray(read_pkl(pkl_out_path + ".zip").astype(np.uint8))
                    img.save(seg_result_png_path)
                    twohands_png_paths_videos[video_id].append((frame_idx, frame_id, seg_result_png_path))
                    continue
                else:
                    try:
                        img_np = reader.get_frame(frame_idx)
                    except ToggleableException as ex:
                        print(f"Exception when processing {frame_id}:", ex)
                        continue
                
                twohands_png_paths_videos[video_id].append((frame_idx, frame_id, seg_result_png_path))

                # no image_width and image_height needed in output PKL since we can use the shape here
                seg_result = inference_segmentor(twohands_model, img_np)[0]
                seg_result_np = np.array(seg_result)
                img = Image.fromarray(img_np)

                # need to temporarily save result as PNG (framework set up this way)
                Image.fromarray(seg_result.astype(np.uint8)).save(seg_result_png_path)
                
                with zipfile.ZipFile(f"{pkl_out_path}.zip", "w", zipfile.ZIP_DEFLATED, False) as zip_file:
                    zip_file.writestr(basename(pkl_out_path), pickle.dumps(seg_result_np))

                global_frame_idx += 1

                if args.visualization_interval > 0 and global_frame_idx % args.visualization_interval == 0:
                    mask = np.array(seg_result) > 0
                    sup_img = superimpose_colored_mask(img, mask, (255, 0, 0))
                    sup_img.save(join(hands_pkl_path, video_id, "jpgs", f"{frame_id}.jpg"))
        except ToggleableException as ex:
            print(f"Error processing {video_id}:", ex)
            continue

    del twohands_model

    print()
    print("Processing step 2/3: contact boundary")
    print()

    twohands_to_cb_png_paths_videos = {}
    twohands_to_cb_model = init_segmentor(TWOHANDS_TO_CB_CONFIG_PATH, TWOHANDS_TO_CB_CHECKPOINT_PATH, device=args.device)
    
    twohands_to_cb_png_path = join(args.output_dir, "pred_cb")
    twohands_to_cb_pkl_path = join(args.output_dir, "contact_boundary")
    
    global_frame_idx = 0
    for video_id, twohands_png_paths in twohands_png_paths_videos.items():
        reader = VideoReader(get_video_path(video_id), get_extracted_frame_dir_path(video_id),
                             assumed_fps=EK_ASSUMED_FPS)
        
        twohands_to_cb_png_paths_videos[video_id] = []
        os.makedirs(join(twohands_to_cb_png_path, video_id), exist_ok=True)
        os.makedirs(join(twohands_to_cb_pkl_path, video_id, "pkls"), exist_ok=True)
        os.makedirs(join(twohands_to_cb_pkl_path, video_id, "jpgs"), exist_ok=True)

        for (frame_idx, frame_id, twohands_png_img_path) in tqdm(twohands_png_paths):
            try:
                seg_result_png_path = join(twohands_to_cb_png_path, f"{frame_id}.png")
                pkl_out_path = join(twohands_to_cb_pkl_path, video_id, "pkls", f"{frame_id}.pkl")
                twohands_to_cb_png_paths_videos[video_id].append((frame_idx, frame_id, seg_result_png_path))
                if isfile(pkl_out_path + ".zip"):
                    # need to temporarily save result as PNG (framework set up this way)
                    img = Image.fromarray(read_pkl(pkl_out_path + ".zip").astype(np.uint8))
                    img.save(seg_result_png_path)
                    continue

                # no image_width and image_height needed in output PKL since we can use the shape here
                seg_result = inference_segmentor(twohands_to_cb_model, twohands_png_img_path)[0]
                seg_result_np = np.array(seg_result)

                # need to temporarily save result as PNG (framework set up this way)
                Image.fromarray(seg_result.astype(np.uint8)).save(seg_result_png_path)

                with zipfile.ZipFile(f"{pkl_out_path}.zip", "w", zipfile.ZIP_DEFLATED, False) as zip_file:
                    zip_file.writestr(basename(pkl_out_path), pickle.dumps(seg_result_np))

                    global_frame_idx += 1

                    if args.visualization_interval > 0 and global_frame_idx % args.visualization_interval == 0:
                        try:
                            img_np = reader.get_frame(frame_idx)
                        except ToggleableException as ex:
                            print(f"Exception when processing {frame_id}:", ex)
                            continue
                        
                        img = Image.fromarray(img_np)
                        mask = np.array(seg_result) > 0
                        sup_img = superimpose_colored_mask(img, mask, (0, 0, 255))
                        sup_img_output_path = join(twohands_to_cb_pkl_path, video_id, "jpgs", f"{frame_id}.jpg")
                        os.makedirs(dirname(sup_img_output_path), exist_ok=True)
                        sup_img.save(sup_img_output_path)
            except ToggleableException as ex:
                print(f"Error processing {frame_id}:", ex)
                continue

    del twohands_to_cb_model

    print()
    print("Processing step 3/3: contacted objects")
    print()

    twohands_cb_to_obj_model = init_segmentor(TWOHANDS_CB_TO_OBJ_CONFIG_PATH, TWOHANDS_CB_TO_OBJ_CHECKPOINT_PATH, device=args.device)
    
    obj_pkl_path = join(args.output_dir, "object")
    
    global_frame_idx = 0
    for video_id, twohands_to_cb_png_paths in twohands_to_cb_png_paths_videos.items():
        reader = VideoReader(get_video_path(video_id),
                             get_extracted_frame_dir_path(video_id),
                             assumed_fps=EK_ASSUMED_FPS)

        os.makedirs(join(obj_pkl_path, video_id, "pkls"), exist_ok=True)
        os.makedirs(join(obj_pkl_path, video_id, "jpgs"), exist_ok=True)

        for (frame_idx, frame_id, twohands_to_cb_png_img_path) in tqdm(twohands_to_cb_png_paths):
            try:
                frame_id = fmt_frame(video_id, frame_idx)
                pkl_out_path = join(obj_pkl_path, video_id, "pkls", f"{frame_id}.pkl")
                if isfile(pkl_out_path + ".zip"):
                    continue
                
                # no image_width and image_height needed in output PKL since we can use the shape here
                seg_result = inference_segmentor(twohands_cb_to_obj_model, twohands_to_cb_png_img_path)[0]
                seg_result_np = np.array(seg_result)
                
                with zipfile.ZipFile(f"{pkl_out_path}.zip", "w", zipfile.ZIP_DEFLATED, False) as zip_file:
                    zip_file.writestr(basename(pkl_out_path), pickle.dumps(seg_result_np))

                global_frame_idx += 1

                if args.visualization_interval > 0 and global_frame_idx % args.visualization_interval == 0:
                    try:
                        img_np = reader.get_frame(frame_idx)
                    except ToggleableException as ex:
                        print(f"Exception when processing {frame_id}:", ex)
                        continue
                    img = Image.fromarray(img_np)
                    mask = np.array(seg_result) > 0
                    sup_img = superimpose_colored_mask(img, mask, (0, 255, 0))
                    sup_img.save(join(obj_pkl_path, video_id, "jpgs", f"{frame_id}.jpg"))
            except ToggleableException as ex:
                print(f"Error processing {frame_id}:", ex)
                continue

    shutil.rmtree(hands_png_path)
    shutil.rmtree(twohands_to_cb_png_path)


if __name__ == "__main__":
    main()
