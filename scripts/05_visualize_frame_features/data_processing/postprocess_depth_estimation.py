import argparse
from collections import OrderedDict
import cv2
import math
import matplotlib.pyplot as plt
import multiprocessing as mp
from multiprocessing.pool import Pool
import numpy as np
import os
from os.path import basename, dirname, isfile, isdir, join
import pickle
from PIL import Image
import requests
from scipy.sparse import csr_matrix
import sys
import time
import torch
from tqdm import tqdm
import zipfile

sys.path.append(dirname(dirname(__file__)))

from data_handling.video_reader import VideoReader
from data_handling.reflection import *
from data_handling.specific.ek100 import *

from utils.args import arg_dict_to_list
from utils.globals import *
from utils.imaging import load_raw_float32_image, color_map_magma


def visualize_depth(depth, depth_min=None, depth_max=None):
    # from https://github.com/facebookresearch/robust_cvd/blob/main/utils/visualization.py
    if depth_min is None:
        depth_min = np.nanmin(depth)

    if depth_max is None:
        depth_max = np.nanmax(depth)

    depth_scaled = (depth - depth_min) / (depth_max - depth_min)
    depth_scaled = depth_scaled ** 0.5
    depth_scaled_uint8 = np.uint8(depth_scaled * 255)

    return ((cv2.applyColorMap(
        depth_scaled_uint8, color_map_magma) / 255) ** 2.2) * 255


def process(process_idx, data):
    args, video_ids, dir_paths_top = data

    for video_id, dir_paths in zip(video_ids, dir_paths_top):
        print()
        print(f"Process #{process_idx}: processing {video_id}")
        print()
        input_output_paths = []
        video_min_depth = {"midas2": np.inf, "cvd": np.inf}
        video_max_depth = {k: -np.inf for k in video_min_depth.keys()}
        video_min_depth_full = {k: np.inf for k in video_min_depth.keys()}
        video_max_depth_full = {k: -np.inf for k in video_min_depth.keys()}
        # limit range to eliminate outliers:
        min_percentile = 0.5
        max_percentile = 99.5
        reader = VideoReader(get_video_path(video_id), get_extracted_frame_dir_path(video_id), assumed_fps=EK_ASSUMED_FPS)

        scaling_dict = {k: {} for k in video_min_depth.keys()}

        for dir_path in reversed(sorted(dir_paths)):
            if not isdir(dir_path):
                continue
            
            original_start_frame_idx = next((int(spl[2:]) for spl in basename(dir_path).split("_") if spl.startswith("OS")), None)
            if original_start_frame_idx is None:
                continue

            # !!!!!!!!!!
            if original_start_frame_idx > 8000:
                continue

            section_min_depth = {k: np.inf for k in video_min_depth.keys()}
            section_max_depth = {k: -np.inf for k in video_min_depth.keys()}
            section_min_depth_full = {k: np.inf for k in video_min_depth.keys()}
            section_max_depth_full = {k: -np.inf for k in video_min_depth.keys()}
            
            print(f"{dir_path=} {original_start_frame_idx=}")

            dirs_to_scan = []        
            aggregated_output_dirname = next(filter(lambda n: n.startswith("R"), os.listdir(dir_path)), None)
            if (aggregated_output_dirname is not None
                and isdir(aggregated_output_dir_path := join(dir_path, aggregated_output_dirname, "StD100.0_StR1.0_SmD0_SmR0.0", "depth_e0000", "e0000_filtered", "depth"))):
                dirs_to_scan.append(("cvd", aggregated_output_dir_path))

            depth_output_path = join(dir_path, "depth_midas2", "depth")
            if isdir(depth_output_path):
                dirs_to_scan.append(("midas2", depth_output_path))

            for version, dir_to_scan in dirs_to_scan:
                for fn in sorted(os.listdir(dir_to_scan)):
                    if not fn.endswith(".raw"):
                        continue

                    input_path = join(dir_to_scan, fn)
                    # format: frame_NNNNNN.png
                    original_frame_delta_idx = next((int(spl) for spl in fn.replace(".", "_").split("_") if spl.isnumeric()), None)
                    if original_frame_delta_idx is not None:
                        original_frame_idx = original_start_frame_idx + original_frame_delta_idx
                        virtual_frame_idx = reader.get_virtual_frame_idx(original_frame_idx)
                        frame_id = fmt_frame(video_id, virtual_frame_idx)
                        video_output_path = join(args.output_dir, version + "_video_scale", video_id, f"{frame_id}.png")
                        section_output_path = join(args.output_dir, version + "_section_scale", video_id, f"{frame_id}.png")

                        try:
                            input_data = load_raw_float32_image(input_path)
                        except Exception as ex:
                            print(f"Cannot process {input_path}:", ex)
                            continue
                                        
                        finite = np.isfinite(input_data)
                        if np.sum(finite) == 0:
                            print(f"Cannot process {input_path}: no valid depth")
                            continue

                        valid_input_data = input_data[finite]
                        lower = np.percentile(valid_input_data, min_percentile)
                        upper = np.percentile(valid_input_data, max_percentile)
                        lower_full = np.percentile(valid_input_data, 0)
                        upper_full = np.percentile(valid_input_data, 100)
                        video_min_depth[version] = min(video_min_depth[version], lower)
                        video_max_depth[version] = max(video_max_depth[version], upper)
                        video_min_depth_full[version] = min(video_min_depth_full[version], lower_full)
                        video_max_depth_full[version] = max(video_max_depth_full[version], upper_full)
                        section_min_depth[version] = min(section_min_depth[version], lower)
                        section_max_depth[version] = max(section_max_depth[version], upper)
                        section_min_depth_full[version] = min(section_min_depth_full[version], lower_full)
                        section_max_depth_full[version] = max(section_max_depth_full[version], upper_full)

                        if not isfile(video_output_path) or not isfile(section_output_path):
                            input_output_paths.append( (version, input_path, video_output_path, section_output_path) )

                scaling_dict[version][original_start_frame_idx] = {"min": section_min_depth[version],
                                                                   "max": section_max_depth[version],
                                                                   "min_full": section_min_depth_full[version],
                                                                   "max_full": section_max_depth_full[version]}

        for version in ["midas2", "cvd"]:
            # only store result in "_video_scale"
            summary_output_path = join(args.output_dir, version + "_video_scale", video_id, "_summary.pkl")
            os.makedirs(dirname(summary_output_path), exist_ok=True)
            with open(summary_output_path, "wb") as f:
                pickle.dump({"section_scaling": OrderedDict(sorted(scaling_dict[version].items())),
                             "video_scaling": {"min": video_min_depth[version], "max": video_max_depth[version],
                                               "min_full": video_min_depth_full[version], "max_full": video_max_depth_full[version]},
                             "min_percentile": min_percentile, "max_percentile": max_percentile}, f)

        for (version, input_path, video_output_path, section_output_path) in input_output_paths:
            input_data = load_raw_float32_image(input_path)

            color_img_video_np = visualize_depth(input_data, video_min_depth[version], video_max_depth[version])
            color_img_video = Image.fromarray(color_img_video_np[:, :, ::-1].astype(np.uint8))
            color_img_video_res = color_img_video.resize((reader.video_width, reader.video_height), Image.BICUBIC)
            
            color_img_section_np = visualize_depth(input_data, section_min_depth[version], section_max_depth[version])
            color_img_section = Image.fromarray(color_img_section_np[:, :, ::-1].astype(np.uint8))
            color_img_section_res = color_img_section.resize((reader.video_width, reader.video_height), Image.BICUBIC)

            os.makedirs(dirname(video_output_path), exist_ok=True)
            os.makedirs(dirname(section_output_path), exist_ok=True)
            color_img_video_res.save(video_output_path)
            color_img_section_res.save(section_output_path)
            print(f"Converted {input_path}  -->  {video_output_path} & {section_output_path}")


def main(arg_dict=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--generator_videos", action="append", type=str, default=None)
    parser.add_argument("--input_dir", type=str, default=DEPTH_ESTIMATION_DATA_DIR)
    parser.add_argument("--output_dir", type=str, default=DEPTH_ESTIMATION_POSTPROCESSING_DATA_DIR)
    parser.add_argument("--num_jobs", type=int, default=1)
    
    parser.add_argument("-f", "--f", help="Dummy argument to make ipython work", default="")
    args, _ = parser.parse_known_args(arg_dict_to_list(arg_dict))

    if args.generator_videos is None:
        args.generator_videos = get_video_list()
    else:
        args.generator_videos = [s.strip() for v in args.generator_videos for s in v.split(",")]

    # structure: (video id, dir path)
    dirs_to_process = {}
    for video_id in args.generator_videos:
        dirs_to_process[video_id] = []
        video_raw_data_path = join(args.input_dir, video_id)
        if isdir(video_raw_data_path):
            for subdir in os.listdir(video_raw_data_path):
                if isdir(join(video_raw_data_path, subdir)):
                    dirs_to_process[video_id].append(join(video_raw_data_path, subdir))
        
        if len(dirs_to_process[video_id]) == 0:
            del dirs_to_process[video_id]
    
    # split up dirs
    
    args.num_jobs = min(args.num_jobs, len(args.generator_videos))

    if args.num_jobs == 1:
        keys = list(dirs_to_process.keys())
        process(0, (args, keys, [dirs_to_process[k] for k in keys]))
    else:
        with Pool(processes=args.num_jobs) as pool:
            keys_split = list(map(lambda a: list(map(str, a)), np.array_split(list(dirs_to_process.keys()), args.num_jobs)))
            pool.starmap(process, enumerate(zip([args] * args.num_job, keys_split, [[dirs_to_process[k] for k in ks] for ks in keys_split])))


if __name__ == "__main__":
    main()
