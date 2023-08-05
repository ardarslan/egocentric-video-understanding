import argparse
from collections import OrderedDict
import cv2
import math
import matplotlib
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import multiprocessing as mp
from multiprocessing.pool import Pool
import numpy as np
import os
from os.path import basename, dirname, isfile, isdir, join
import pickle
from PIL import Image, ImageDraw
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
from data_processing.postprocess_depth_estimation import visualize_depth

from utils.args import arg_dict_to_list
from utils.globals import *
from utils.imaging import color_map_magma, load_raw_float32_image, scale_box


ELLIPSE_RADIUS = 10

def qvec2rotmat(qvec):
    # from COLMAP
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])


def project_point_image(pt,
                        pose_data: list,
                        camera: dict):
    # based on "project_line_image" from EPIC-Fields
    rot_w2c = qvec2rotmat(pose_data[:4])
    tvec = np.asarray(pose_data[4:7])
    # Represent as column vector
    pt = rot_w2c @ pt + tvec
    width, height = camera['width'], camera['height']
    fx, fy, cx, cy, k1, k2, p1, p2 = camera['params']

    pt_uv = pt[:2] / pt[2]
    pt_uv = pt_uv * np.array([fx, fy]) + np.array([cx, cy])
    return pt_uv, pt


def get_w2c_c2w(img_data: list) -> np.ndarray:
    # adapted from https://github.com/epic-kitchens/epic-fields-code/blob/main/tools/common_functions.py
    w2c = np.eye(4)
    w2c[:3, :3] = qvec2rotmat(img_data[:4])
    w2c[:3, -1] = img_data[4:7]
    c2w = np.linalg.inv(w2c)
    return w2c, c2w


def main(arg_dict=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--generator_videos", action="append", type=str, default=None)
    parser.add_argument("--depth_dir", type=str, default=DEPTH_ESTIMATION_DATA_DIR)
    parser.add_argument("--point_data_dir", type=str, default=POINT_DATA_DIR)
    parser.add_argument("--output_dir", type=str, default="/mnt/scratch/agavryushin/Thesis/data/EK_backprojected_points_vis/")
    parser.add_argument("--frame_sample_size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--max_color_diff_norm", type=float, default=10.0)
    parser.add_argument("--max_s_depth_diff", type=float, default=0.01)

    parser.add_argument("-f", "--f", help="Dummy argument to make ipython work", default="")
    args, _ = parser.parse_known_args(arg_dict_to_list(arg_dict))

    np.random.seed(args.seed)
    
    if args.generator_videos is None:
        args.generator_videos = get_video_list()
    else:
        args.generator_videos = [s.strip() for v in args.generator_videos for s in v.split(",")]
    
    for video_id in args.generator_videos:
        point_data_file_path = join(args.point_data_dir, f"{video_id}.json")
        if not isfile(point_data_file_path):
            continue

        with open(point_data_file_path) as f:
            point_data = json.load(f)

        camera = point_data["camera"]
        fx, fy, cx, cy, k1, k2, p1, p2 = camera['params']

        cam_width = point_data["camera"]["width"]
        cam_height = point_data["camera"]["height"]

        depth_top_path = join(args.depth_dir, video_id)

        reader = VideoReader(get_video_path(video_id), get_extracted_frame_dir_path(video_id), assumed_fps=-1)
        video_len = len(reader)

        # structure: (orig depth, estimated depth)
        pairs_depth_pt = []
        pairs_depth_est = []
        # coordinates relative to *full* image
        pairs_xy_img_pre_proj_framewise = {}  
        pairs_xy_img_post_proj_framewise = {}
        pairs_depth_pt_framewise = {}

        sampled_idxs = np.random.choice(video_len, args.frame_sample_size, replace=False)
        for original_frame_idx in sampled_idxs:
            # NOTE: the images are 1-indexed in EK
            original_frame_idx_str = f"frame_{'%010d' % (original_frame_idx + 1)}.jpg"
            if original_frame_idx_str not in point_data["images"]:
                continue

            for depth_top_subdir in os.listdir(depth_top_path):
                depth_top_subdir_path = join(depth_top_path, depth_top_subdir)
                if not isdir(depth_top_subdir_path):
                    continue

                original_frame_delta_idx = next((int(spl[2:]) for spl in depth_top_subdir.split("_") if spl.startswith("OS")), None)
                if original_frame_delta_idx > original_frame_idx:
                    continue

                aggregated_output_dirname = next(filter(lambda n: n.startswith("R"), os.listdir(depth_top_subdir_path)), None)
                frame_depth_path = join(depth_top_subdir_path, aggregated_output_dirname,
                                        "StD100.0_StR1.0_SmD0_SmR0.0", "depth_e0000", "e0000_filtered", "depth",
                                        f"frame_{'%06d' % (original_frame_idx-original_frame_delta_idx)}.raw")
                
                if original_frame_idx not in pairs_xy_img_pre_proj_framewise:
                    pairs_xy_img_pre_proj_framewise[original_frame_idx] = []
                    
                if original_frame_idx not in pairs_xy_img_post_proj_framewise:
                    pairs_xy_img_post_proj_framewise[original_frame_idx] = []
                
                if original_frame_idx not in pairs_depth_pt_framewise:
                    pairs_depth_pt_framewise[original_frame_idx] = []

                if isfile(frame_depth_path):
                    disparity_data = load_raw_float32_image(frame_depth_path)
                    disparity_data_blurred = cv2.GaussianBlur(disparity_data, (7, 7), 3)
                    depth_width = disparity_data.shape[1]
                    depth_height = disparity_data.shape[0]

                    disparity_data_q1 = np.quantile(disparity_data, 0.01)
                    disparity_data_q99 = np.quantile(disparity_data, 0.99)
                                    
                    disparity_scaled = (disparity_data - disparity_data_q1) / (disparity_data_q99 - disparity_data_q1)
                    disparity_scaled = disparity_scaled ** 0.5

                    depth_img = visualize_depth(disparity_data_blurred, disparity_data_q1, disparity_data_q99)

                    # backproject the points to this image

                    output_dir_path = join(args.output_dir, video_id)
                    os.makedirs(output_dir_path, exist_ok=True)
                    # backproject points

                    img_np = reader.get_frame(original_frame_idx)
                    img_np_blurred = cv2.GaussianBlur(img_np, (7, 7), 3)
                    img = Image.fromarray(img_np)
                    draw = ImageDraw.Draw(img)
                    for point in point_data["points"]:
                        image_pt, pt_real_cam_space = project_point_image(point[:3], point_data["images"][original_frame_idx_str], camera)
                        image_depth = pt_real_cam_space[2]
                        if not (min(image_pt[0], image_pt[1]) >= 0 and image_pt[0] < cam_width and image_pt[1] < cam_height):
                           # check if this is correct
                           continue
                        
                        image_pt_rescaled = [int(image_pt[0] / cam_width * img.width), int(image_pt[1] / cam_height * img.height)]
                        image_pt_rescaled_depth = [int(image_pt[0] / cam_width * depth_width), int(image_pt[1] / cam_height * depth_height)]

                        if not (0 <= image_pt_rescaled_depth[0] < depth_width and 0 <= image_pt_rescaled_depth[1] < depth_height):
                            continue
                        
                        if not (0 <= image_pt_rescaled[0] < img.width and 0 <= image_pt_rescaled[1] < img.height):
                            continue

                        # get color diff
                        if np.linalg.norm(img_np_blurred[image_pt_rescaled[1], image_pt_rescaled[0]][::-1] - np.array(point[3:])) > args.max_color_diff_norm:
                            continue
                        
                        # !!!!!!!!!!
                        if np.linalg.norm(np.array(point[3:])) < 100.0:
                            continue

                        #image_pt_rescaled = image_pt

                        disparity = disparity_data_blurred[image_pt_rescaled_depth[1], image_pt_rescaled_depth[0]]
                        if abs(disparity) < 1e-8:
                            continue

                        if disparity < disparity_data_q1 or disparity > disparity_data_q99:
                            continue

                        est_depth = 1.0 / disparity
                        pairs_depth_pt.append(image_depth)  # (point[2])
                        pairs_depth_est.append(est_depth)
                        
                        pairs_xy_img_pre_proj_framewise[original_frame_idx].append(image_pt_rescaled)
                        pairs_xy_img_post_proj_framewise[original_frame_idx].append(pt_real_cam_space)
                        pairs_depth_pt_framewise[original_frame_idx].append(image_depth)
                        
                        draw.ellipse((image_pt_rescaled[0] - ELLIPSE_RADIUS, image_pt_rescaled[1] - ELLIPSE_RADIUS,
                                      image_pt_rescaled[0] + ELLIPSE_RADIUS, image_pt_rescaled[1] + ELLIPSE_RADIUS), 
                                     fill=tuple(point[3:]), outline=(255, 0, 0), width=1)
                        
                    output_file_path = join(output_dir_path, video_id, original_frame_idx_str)
                    os.makedirs(dirname(output_file_path), exist_ok=True)
                    img.save(output_file_path)


                    print(f"Saved {output_file_path}")
        

        # Credit to https://github.com/isl-org/MiDaS/issues/171#issue-1242702825
        A = np.vstack([pairs_depth_est, np.ones(len(pairs_depth_est))]).T
        s, t = np.linalg.lstsq(A, np.array(pairs_depth_pt), rcond=None)[0]

        # compute pairwise distances

        """
        pairs_xy_img_pre_proj_framewise[original_frame_idx].append(image_pt_rescaled)
        pairs_xy_img_post_proj_framewise[original_frame_idx].append(image_pt)
        pairs_depth_pt_framewise[original_frame_idx].append(image_depth)
        """

        s_list_x = []
        s_list_y = []

        for key in pairs_depth_pt_framewise.keys():
            num_entries = len(pairs_depth_pt_framewise[key])
            for idx_1 in range(num_entries):
                depth_pt_1 = pairs_depth_pt_framewise[key][idx_1]
                xy_img_pre_proj_1 = pairs_xy_img_pre_proj_framewise[key][idx_1]
                xy_img_post_proj_1 = pairs_xy_img_post_proj_framewise[key][idx_1]

                for idx_2 in range(idx_1 + 1, num_entries):
                    depth_pt_2 = pairs_depth_pt_framewise[key][idx_2]
                    xy_img_pre_proj_2 = pairs_xy_img_pre_proj_framewise[key][idx_2]
                    xy_img_post_proj_2 = pairs_xy_img_post_proj_framewise[key][idx_2]
                    
                    if abs(depth_pt_1 - depth_pt_2) <= args.max_s_depth_diff:

                        # consider x distance and y distance separately
                        #real_xy_dist = np.linalg.norm(np.array(xy_img_post_proj_1) - np.array(xy_img_post_proj_2))
                        #img_xy_dist = np.linalg.norm(np.array(xy_img_pre_proj_1) - np.array(xy_img_pre_proj_2))
                        real_x_dist = abs(xy_img_post_proj_1[0] - xy_img_post_proj_2[0])
                        img_x_dist = abs(xy_img_pre_proj_1[0] - xy_img_pre_proj_2[0])

                        real_y_dist = abs(xy_img_post_proj_1[1] - xy_img_post_proj_2[1])
                        img_y_dist = abs(xy_img_pre_proj_1[1] - xy_img_pre_proj_2[1])

                        if len(set([0, np.nan]).intersection(set([real_x_dist, img_x_dist]))) == 0:
                            s_x = real_x_dist / (img_x_dist * depth_pt_1)
                            s_list_x.append(s_x)
                            
                        if len(set([0, np.nan]).intersection(set([real_y_dist, img_y_dist]))) == 0:
                            s_y = real_y_dist / (img_y_dist * depth_pt_1)
                            s_list_y.append(s_y)

        s_x_mean = np.mean(s_list_x) if len(s_list_x) > 0 else np.nan
        s_x_std = np.std(s_list_x) if len(s_list_x) > 0 else np.nan
        
        s_y_mean = np.mean(s_list_y) if len(s_list_y) > 0 else np.nan
        s_y_std = np.std(s_list_y) if len(s_list_y) > 0 else np.nan

        print(f"{s=} {t=} {len(s_list_x)=} s_x_mean={'%.4f' % s_x_mean} s_x_std={'%.4f' % s_x_std} s_y_mean={'%.4f' % s_y_mean} s_y_std={'%.4f' % s_y_std}")

        fig, ax = plt.subplots()
        ax.scatter(pairs_depth_est, pairs_depth_pt)
        ax.set_xlabel('Estimated depth (CVD)', fontsize=16)
        ax.set_ylabel('Point depth (EPIC-Fields)', fontsize=16)
        x0 = min(pairs_depth_est)
        x1 = max(pairs_depth_est)
        ax.plot([x0, x1], [s * x0 + t, s * x1 + t], color='red')
        fig.savefig("pairs_depth_pt_est.png")


if __name__ == "__main__":
    main()
