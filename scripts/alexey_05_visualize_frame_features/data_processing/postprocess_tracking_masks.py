import argparse
from collections import OrderedDict
import cv2
import math
import matplotlib.pyplot as plt
import multiprocessing as mp
from multiprocessing.pool import Pool
import numpy as np
import os
from os.path import join, dirname, isfile
import pickle
from PIL import Image, ImageDraw, ImageFont
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
from utils.imaging import (bbox_from_mask,
                           check_boxes_intersect,
                           get_bbox_intersection,
                           get_bbox_area,
                           scale_box,
                           calculate_iou,
                           calculate_ioa)
from utils.io import read_pkl
from utils.metrics import chamfer_distance


EPS = 1e-5


def process(process_idx, data):
    font = ImageFont.truetype("/mnt/scratch/agavryushin/Thesis/webserver/fonts/DejaVuSans.ttf", 20)

    args, timestamp, video_ids = data
    ewm_span_abs = round(EK_ASSUMED_FPS * args.ewm_span_rel_to_fps)
    num_processed_frames = 0
    for video_idx, video_id in tqdm(enumerate(video_ids)):
        reader = VideoReader(get_video_path(video_id), get_extracted_frame_dir_path(video_id),
                             assumed_fps=EK_ASSUMED_FPS)
        virtual_frame_count = reader.get_virtual_frame_count()

        seen_tracks = set()

        for image_version in args.image_version:
            for hos_version in args.hos_version:
                for min_length in args.min_length:
                    for track_type in args.track_type:
                        tracking_mask_path = CHANNEL_VIDEO_PATH_FUNCTS["tracking_mask"](video_id, image_version, hos_version, min_length, track_type)
                        if not isdir(tracking_mask_path):
                            continue
                        
                        print(f"Process {process_idx}: {image_version=}, {hos_version=}, {min_length=}, {track_type=}")
                        
                        os.makedirs(join(ROOT_PATH, "data", "contours_vis", video_id), exist_ok=True)

                        tracks = os.listdir(tracking_mask_path)
                        for track_idx, dn in enumerate(tracks):
                            if args.track_filter is not None and not any([dn.startswith(s) for s in args.track_filter]):
                                print(f'Ignoring track "{dn}" by filter', args.track_filter)
                                continue

                            dir_path = join(tracking_mask_path, dn)
                            if isdir(dir_path):
                                print(f'Track "{dn}"...')
                                track_final_tortuosities = []
                                track_avg_quantiles = {q: [] for q in np.arange(0, 1.01, 0.01)}
                                track_cd_avgs = []
                                track_cd_stds = []
                                track_mask_bbox_bottoms = []
                                track_hos_hands_frame_idxs = []
                                track_hos_hands_ious = []
                                track_hos_hands_ioas = []
                                track_hos_object_frame_idxs = []
                                track_hos_object_ious = []
                                track_hos_object_ioas = []
                                track_summary_dict = {"initial_frame_intersections": {},
                                                      "track_mask_appearances": [],
                                                      "track_mask_appearances_filter2_passed": [],
                                                      "track_initial_frame": None, "track_last_frame": None,
                                                      "track_object_intersection_count": 0,
                                                      "track_object_intersection_counts": {},
                                                      "track_object_intersection_bbox_ious": {},
                                                      "track_track_intersection_ious": {},
                                                      "track_track_intersection_ious_filtered": {},
                                                      "track_track_intersection_ioas": {},
                                                      "track_hos_hands_intersection_ious": {},
                                                      "track_hos_hands_intersection_ioas": {},
                                                      "track_hos_object_intersection_ious": {},
                                                      "track_hos_object_intersection_ioas": {},
                                                      "track_track_intersection_intersection_areas": {},
                                                      "track_track_intersection_union_areas": {},
                                                      "track_track_intersection_own_areas": {},  # masks may be resized
                                                      "args": args}
                                range_obj = range(virtual_frame_count)
                                track_mask_width = 0
                                track_mask_height = 0
                                out_dir = join(CHANNEL_FRAME_PATH_FUNCTS["tracking_mask_postprocessing"](video_id, -1, None, image_version, hos_version, min_length, track_type), dn)
                                if isdir(out_dir):
                                    continue
                                os.makedirs(out_dir, exist_ok=True)
                                
                                for frame_idx in (progress_bar := tqdm(range_obj)):
                                    frame_id = fmt_frame(video_id, frame_idx)
                                    path = join(dir_path, f"{frame_id}__{dn}.pkl.zip")
                                    num_contours = 0
                                    if isfile(path):
                                        try:
                                            output_dict = {"track_track_intersection_ious": {},
                                                        "track_track_intersection_ious_filtered": {},
                                                        "track_track_intersection_ioas": {},
                                                        "track_track_intersection_intersection_areas": {},
                                                        "track_track_intersection_union_areas": {},
                                                        "track_track_intersection_own_areas": {},  # masks may be resized
                                                        "track_hos_hands_intersection_iou": None,
                                                        "track_hos_hands_intersection_ioa": None,
                                                        "track_hos_object_intersection_iou": None,
                                                        "track_hos_object_intersection_ioa": None,
                                                        "track_hos_hands_intersection_iou_ema": None,
                                                        "track_hos_hands_intersection_ioa_ema": None,
                                                        "track_hos_object_intersection_iou_ema": None,
                                                        "track_hos_object_intersection_ioa_ema": None,
                                                        "intersection_check_performed": False,
                                                        "filter2_passed": False,
                                                        "args": args}
                                            data = read_pkl(path)
                                            masks = data["masks"]

                                            if track_summary_dict["track_initial_frame"] is None:
                                                track_summary_dict["track_initial_frame"] = frame_id

                                            track_summary_dict["track_last_frame"] = frame_id
                                            
                                            if 0 in [track_mask_width, track_mask_height]:
                                                track_mask_height = masks.shape[0]
                                                track_mask_width = masks.shape[1]

                                            image_diag_size = math.sqrt(masks.shape[0] ** 2.0 + masks.shape[1] ** 2.0)
                                            line_length_px = round(args.relative_segment_size * image_diag_size)
                                            *_, contours, hierarchy_top = cv2.findContours(masks, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                                            
                                            curvatures = []
                                            tortuosities = []

                                            this_frame_all_contour_points = []
                                            # list of ndarrays, one ndarray per contour, for this image:
                                            # list of image ndarrays, one ndarray per contour, for this image:
                                            this_frame_all_contours_images = []  
                                            num_contours = len(contours)

                                            if num_contours > 0:
                                                if dn not in seen_tracks:
                                                    # check for overlap
                                                    for seen_track_id in seen_tracks:
                                                        tracking_mask_dir_2 = CHANNEL_VIDEO_PATH_FUNCTS["tracking_mask"](video_id, image_version, hos_version, min_length, track_type)
                                                        # check for mask of this track
                                                        tracking_mask_path_2 = join(tracking_mask_dir_2, f"{frame_id}__{dn}.pkl.zip")
                                                        if isfile(tracking_mask_path_2):
                                                            data_2 = read_pkl(tracking_mask_path_2)
                                                            track_summary_dict["initial_frame_intersections"][seen_track_id] = np.logical_and(data["masks"], data_2["masks"]).sum()
                                                    
                                                    seen_tracks.add(dn)

                                                image = np.stack((masks * 255, masks * 255, masks * 255), axis=-1)
                                                
                                                #Image.fromarray(image).convert("RGB").save(join(ROOT_PATH, "data", "contours_vis", video_id, f"{frame_id}__contours__test__1.jpg"))

                                                for contour in contours:  # contours in current image
                                                    # for each contour separately, split into segments and check curvature change
                                                    pos = 0
                                                    lines = []
                                                    line_angles = []
                                                    current_line = []
                                                    current_line_length = 0
                                                    
                                                    this_contour_rel_points = []

                                                    while pos < len(contour):
                                                        x = contour[pos][0][0]
                                                        y = contour[pos][0][1]
                                                        this_frame_all_contour_points.append([x, y])
                                                        this_contour_rel_points.append([x / track_mask_width, y / track_mask_height])
                                                        if len(current_line) == 0:
                                                            current_line.append([x, y])
                                                            current_line_length = 0
                                                        else:
                                                            last_x = current_line[-1][0]
                                                            last_y = current_line[-1][1]
                                                            angle = math.atan2(y - last_y, x - last_x)  # y, x
                                                            remaining_length = math.sqrt((x - last_x) ** 2.0 + (y - last_y) ** 2.0)
                                                            while remaining_length > 0:
                                                                seg_length = min(line_length_px - current_line_length, remaining_length)
                                                                intermediate_x = last_x + seg_length * math.cos(angle)
                                                                intermediate_y = last_y + seg_length * math.sin(angle)
                                                                remaining_length -= seg_length
                                                                current_line_length += seg_length
                                                                current_line.append([intermediate_x, intermediate_y])
                                                                if current_line_length + EPS >= line_length_px or pos == len(contour) - 1:
                                                                    lines.append(current_line)
                                                                    line_angles.append(angle)
                                                                    #cv2.line(image, (int(round(current_line[0][0])), int(round(current_line[0][1]))), (int(round(current_line[-1][0])), int(round(current_line[-1][1]))), (0, 255, 0), thickness=1)
                                                                    current_line = [[intermediate_x, intermediate_y]]
                                                                    current_line_length = 0
                                                                
                                                                last_x = intermediate_x
                                                                last_y = intermediate_y
                                                        
                                                        pos += 1

                                                    contour_img = np.zeros((track_mask_height, track_mask_width))
                                                    # color=(255, 255, 255)
                                                    # cv2.drawContours(contour_img, contour, -1, color=1, thickness=cv2.FILLED)
                                                    cv2.fillPoly(contour_img, pts=[contour], color=1)
                                                    this_frame_all_contours_images.append(contour_img)

                                                    if len(line_angles) >= 2:
                                                        # based on https://stackoverflow.com/a/7597763
                                                        last_angle = None
                                                        angle_deltas = []
                                                        for angle in line_angles:
                                                            if last_angle is not None:
                                                                angle_deltas.append(abs(angle-last_angle))
                                                            last_angle = angle

                                                        tortuosity = sum(angle_deltas) / (len(angle_deltas) * np.pi)
                                                        tortuosities.append((tortuosity, len(angle_deltas)))
                                                
                                                final_tortuosity = sum([t[0] * t[1] for t in tortuosities]) / max(1, sum([t[1] for t in tortuosities]))
                                                
                                                track_final_tortuosities.append(final_tortuosity)

                                                # format for "intersections": (this frame contour idx, other track id, other frame contour idx): IoU / (w*h)
                                                output_dict.update({"tortuosity": final_tortuosity,
                                                                    "tortuosity_ema": pd.Series(track_final_tortuosities[-ewm_span_abs:]).ewm(span=ewm_span_abs).mean().iloc[-1],
                                                                    "image_width": masks.shape[1],
                                                                    "image_height": masks.shape[0],
                                                                    "num_contours": num_contours,
                                                                    "contour_ious": None,  # disabled for now
                                                                    })

                                                # get boundaries from SAM on full image, compute chamfer dist
                                                
                                                segmentation_mask_path = CHANNEL_FRAME_PATH_FUNCTS["segmentation_mask"](video_id, frame_idx, frame_id, image_version)
                                                
                                                if isfile(segmentation_mask_path):
                                                    segmentation_mask_data = read_pkl(segmentation_mask_path)
                                                    # get contours of masks
                                                    cds = []
                                                    this_frame_all_mask_contours = []
                                                    for top_level_box in segmentation_mask_data:
                                                        box = top_level_box["box"]
                                                        cls = top_level_box["cls"]
                                                        seg_masks = top_level_box["masks"]
                                                        for mask in seg_masks:
                                                            mask_img = mask["segmentation"].toarray().astype(np.uint8)
                                                            *_, mask_contours, mask_hierarchy_top = cv2.findContours(mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                                                                        
                                                            for contour in mask_contours:
                                                                # for each contour separately, split into segments and check curvature change
                                                                for entry in contour:
                                                                    x = entry[0][0]
                                                                    y = entry[0][1]
                                                                    this_frame_all_mask_contours.append([x, y])
                                                    
                                                    min_x_to_y, min_y_to_x, cd = chamfer_distance(np.array(this_frame_all_contour_points),
                                                                                                np.array(this_frame_all_mask_contours),
                                                                                                direction="x_to_y")
                                                    output_dict["cd_sum"] = cd
                                                    output_dict["cd_avg"] = cd / len(this_frame_all_contour_points)
                                                    output_dict["cd_std"] = np.std(min_x_to_y)
                                                    track_cd_avgs.append(output_dict["cd_avg"])
                                                    track_cd_stds.append(output_dict["cd_std"])

                                                    output_dict["cd_quantiles"] = np.quantile(min_x_to_y, np.arange(0.0, 1.01, 0.01))
                                                    output_dict["cd_avg_quantiles"] = np.quantile(min_x_to_y / len(this_frame_all_contour_points), np.arange(0.0, 1.01, 0.01))
                                                    for q_idx, q in enumerate(np.arange(0, 1.01, 0.01)):
                                                        track_avg_quantiles[q].append(output_dict["cd_avg_quantiles"][q_idx])
                                                    
                                                    output_dict["cd_avgs_ema"] = pd.Series(track_cd_avgs[-ewm_span_abs:]).ewm(span=ewm_span_abs).mean().iloc[-1]
                                                    output_dict["cd_stds_ema"] = pd.Series(track_cd_stds[-ewm_span_abs:]).ewm(span=ewm_span_abs).mean().iloc[-1]
                                                    output_dict["cd_avg_quantiles_ema"] = np.array([pd.Series(track_avg_quantiles[q][-ewm_span_abs:]).ewm(span=ewm_span_abs).mean().iloc[-1] for q in np.arange(0, 1.01, 0.01)])
                                                    
                                                    progress_bar.set_description(f'{frame_id=} {dn=} (#{track_idx+1}/{len(tracks)}) {output_dict["cd_avg"]=} {output_dict["cd_std"]=}')                                                        
                                                    # get CD

                                                os.makedirs(out_dir, exist_ok=True)

                                                # track_summary_dict = {"initial_frame_intersections": {}, "track_mask_appearances": [], "track_object_intersection_counts": {}, "track_object_intersection_bbox_ious": {}}
                                                track_summary_dict["track_mask_appearances"].append(frame_id)

                                                # get bbox
                                                bbox = bbox_from_mask(data["masks"])
                                                bbox_rel = scale_box(bbox, track_mask_width, track_mask_height)
                                                track_mask_bbox_bottoms.append(bbox[-1])
                                                output_dict["track_mask_bbox_bottom"] = bbox[-1]

                                                object_bbox_path_fn = CHANNEL_FRAME_PATH_FUNCTS["object_bbox"]
                                                object_bbox_path = object_bbox_path_fn(video_id,
                                                                                    frame_idx,
                                                                                    frame_id,
                                                                                    args.object_bbox_version)
                                                
                                                if isfile(object_bbox_path):
                                                    object_bbox_data = read_pkl(object_bbox_path)

                                                    for major_idx in range(len(object_bbox_data["boxes"])):
                                                        for box, cls, score in list(zip(object_bbox_data["boxes"][major_idx],
                                                                                        object_bbox_data["classes"][major_idx],
                                                                                        object_bbox_data["scores"][major_idx])):
                                                            cls = cls.lower()
                                                            if cls in UNIDET_IGNORE_CATEGORIES:
                                                                cls = "ignore_superclass"
                                                            
                                                            box_rel = scale_box(box,
                                                                                object_bbox_data.get("image_width", reader.video_width),
                                                                                object_bbox_data.get("image_height", reader.video_height))
                                                            # assume no scaling necessary
                                                            if check_boxes_intersect(box_rel, bbox_rel):
                                                                track_summary_dict["track_object_intersection_count"] += 1
                                                                bbox_intersection = get_bbox_intersection(box_rel, bbox_rel)
                                                                track_summary_dict["track_object_intersection_counts"][cls] = track_summary_dict["track_object_intersection_counts"].get(cls, 0) + 1
                                                                bbox_intersection_area = get_bbox_area(bbox_intersection)
                                                                bbox_union_area = get_bbox_area(box_rel) + get_bbox_area(bbox_rel) - bbox_intersection_area
                                                                bbox_iou = bbox_intersection_area / max(1e-6, bbox_union_area)
                                                                if cls not in track_summary_dict["track_object_intersection_bbox_ious"]:
                                                                    track_summary_dict["track_object_intersection_bbox_ious"][cls] = []
                                                                track_summary_dict["track_object_intersection_bbox_ious"][cls].append(bbox_iou)
                                                    
                                                hand_mask = None
                                                object_mask = None

                                                hos_hands_path_fn = CHANNEL_FRAME_PATH_FUNCTS["hos_hands"]
                                                hos_hands_path = hos_hands_path_fn(video_id, frame_idx, frame_id, args.hos_version_hands)

                                                if isfile(hos_hands_path):
                                                    hos_hands_data = read_pkl(hos_hands_path)
                                                    if args.hos_version_hands == "egohos":
                                                        # 1: left; 2: right
                                                        hand_mask = hos_hands_data > 0
                                                    else:
                                                        for cls, handside, mask, box in zip(hos_hands_data["instances"].pred_classes,
                                                                                            hos_hands_data["instances"].pred_handsides,
                                                                                            hos_hands_data["instances"].pred_masks,
                                                                                            hos_hands_data["instances"].pred_boxes):
                                                            if cls == 0:  # 0: hand
                                                                # 0: left; 1: right
                                                                hand_mask = mask > 0
                                                            elif cls == 1:  # 1: object
                                                                pass

                                                hos_object_path_fn = CHANNEL_FRAME_PATH_FUNCTS["hos_object"]
                                                hos_object_path = hos_object_path_fn(video_id, frame_idx, frame_id, args.hos_version_object)
                                                
                                                if isfile(hos_object_path):
                                                    hos_object_data = read_pkl(hos_object_path)
                                                    if args.hos_version_object == "egohos":
                                                        # 3: object
                                                        object_mask = hos_object_data > 0
                                                    else:
                                                        try:
                                                            for cls, handside, mask, box in zip(hos_object_data["instances"].pred_classes,
                                                                                                hos_object_data["instances"].pred_handsides,
                                                                                                hos_object_data["instances"].pred_masks,
                                                                                                hos_object_data["instances"].pred_boxes):
                                                                if cls == 0:  # 0: hand
                                                                    # 0: left; 1: right
                                                                    pass
                                                                elif cls == 1:  # 1: object
                                                                    object_mask = mask > 0
                                                        except ToggleableException as ex:
                                                            print(f"Exception in {frame_id}:", ex)

                                                # TODO: calculate HOS hand and object IoU with this track here
                                                
                                                """
                                                track_hos_hand_frame_idxs = []
                                                track_hos_hand_ious = []
                                                track_hos_hand_ioas = []
                                                track_hos_object_frame_idxs = []
                                                track_hos_object_ious = []
                                                track_hos_object_ioas = []
                                                """

                                                if hand_mask is not None:
                                                    hands_iou = calculate_iou(masks, hand_mask)
                                                    # smaller denominator -> larger IoA
                                                    hands_ioa = calculate_ioa(masks, hand_mask, denominator="min")
                                                    output_dict["track_hos_hands_intersection_iou"] = hands_iou
                                                    output_dict["track_hos_hands_intersection_ioa"] = hands_ioa
                                                    track_summary_dict["track_hos_hands_intersection_ious"][frame_id] = hands_iou
                                                    track_summary_dict["track_hos_hands_intersection_ioas"][frame_id] = hands_ioa

                                                    track_hos_hands_frame_idxs.append(frame_idx)
                                                    track_hos_hands_ioas.append(hands_ioa)
                                                    track_hos_hands_ious.append(hands_iou)
                                                
                                                # NOTE: may get stuck if no other updating frames
                                                if len(track_hos_hands_frame_idxs) > 0:
                                                    hands_iou_ema = pd.Series(OrderedDict(zip(track_hos_hands_frame_idxs[-ewm_span_abs:], track_hos_hands_ious[-ewm_span_abs:]))).ewm(span=ewm_span_abs).mean().iloc[-1]
                                                    hands_ioa_ema = pd.Series(OrderedDict(zip(track_hos_hands_frame_idxs[-ewm_span_abs:], track_hos_hands_ioas[-ewm_span_abs:]))).ewm(span=ewm_span_abs).mean().iloc[-1]
                                                    output_dict["track_hos_hands_intersection_iou_ema"] = hands_iou_ema
                                                    output_dict["track_hos_hands_intersection_ioa_ema"] = hands_ioa_ema

                                                if object_mask is not None:
                                                    object_iou = calculate_iou(masks, object_mask)
                                                    # larger denominator -> smaller IoA
                                                    object_ioa = calculate_ioa(masks, object_mask, denominator="min")
                                                    output_dict["track_hos_object_intersection_ious"] = object_iou
                                                    output_dict["track_hos_object_intersection_ioas"] = object_ioa
                                                    track_summary_dict["track_hos_object_intersection_ious"][frame_id] = object_iou
                                                    track_summary_dict["track_hos_object_intersection_ioas"][frame_id] = object_ioa

                                                    track_hos_object_frame_idxs.append(frame_idx)
                                                    track_hos_object_ioas.append(object_ioa)
                                                    track_hos_object_ious.append(object_iou)
                                                    
                                                # NOTE: may get stuck if no other updating frames
                                                if len(track_hos_object_frame_idxs) > 0:
                                                    object_iou_ema = pd.Series(OrderedDict(zip(track_hos_object_frame_idxs[-ewm_span_abs:], track_hos_object_ious[-ewm_span_abs:]))).ewm(span=ewm_span_abs).mean().iloc[-1]
                                                    object_ioa_ema = pd.Series(OrderedDict(zip(track_hos_object_frame_idxs[-ewm_span_abs:], track_hos_object_ioas[-ewm_span_abs:]))).ewm(span=ewm_span_abs).mean().iloc[-1]
                                                    output_dict["track_hos_object_intersection_iou_ema"] = object_iou_ema
                                                    output_dict["track_hos_object_intersection_ioa_ema"] = object_ioa_ema

                                                #img = Image.fromarray(image)
                                                #draw = ImageDraw.Draw(img)
                                                #draw.text((30, 30), "%.2f" % final_tortuosity, font=font)
                                                
                                                
                                                #img.convert("RGB").save(join(ROOT_PATH, "data", "contours_vis", video_id, f"{frame_id}__contours__test__2.jpg"))

                                                #print(f"Processed {frame_id}")
                                            
                                                #print(f"{contours.shape=}")

                                                # cv2.drawContours(image, contours, -1, (255, 0, 0), 3)

                                            if masks.max() > 0:
                                                track_mask_bbox = bbox_from_mask(masks)
                                                track_mask_bbox_rel = scale_box(track_mask_bbox, track_mask_width, track_mask_height)

                                                # check whether this frame passes the filter
                                                # TODO: refactor into new function (also in views.py)

                                                tortuosity_ema = output_dict["tortuosity_ema"]
                                                cd_90 = float(output_dict["cd_avg_quantiles_ema"][90]) if "cd_avg_quantiles_ema" in output_dict else np.nan
                                                hand_ioa = float(output_dict["track_hos_hands_intersection_ioa_ema"]) if output_dict.get("track_hos_hands_intersection_ioa_ema") is not None else np.nan
                                                filter_passed = (tortuosity_ema <= args.tracking_mask_merging_max_tortuosity
                                                                 and cd_90 <= args.tracking_mask_merging_max_cd_q90
                                                                 and (hand_ioa if not np.isnan(hand_ioa) else 0.0) <= args.tracking_mask_merging_max_hand_ioa)
                                                if filter_passed:
                                                    output_dict["intersection_check_performed"] = True
                                                    output_dict["filter2_passed"] = True
                                                    track_summary_dict["track_mask_appearances_filter2_passed"].append(frame_id)

                                                    # check for intersections with other tracks' contours on this frame
                                                    for other_track_idx, other_track_id in enumerate(tracks):
                                                        if other_track_id == dn:
                                                            continue

                                                        if args.track_filter is not None and not any([other_track_id.startswith(s) for s in args.track_filter]):
                                                            continue

                                                        # "tracking_mask": lambda video_id, image_version, hos_version, min_length, track_type:
                                                        # join(TRACKING_MASK_DATA_DIR, image_version.replace("inpainted_", ""), hos_version, f"min-length={min_length}", video_id, track_type),
                                                        other_tracking_mask_dir = CHANNEL_VIDEO_PATH_FUNCTS["tracking_mask"](video_id, image_version, hos_version, min_length, track_type)
                                                        other_track_mask_path = join(other_tracking_mask_dir, other_track_id, f"{frame_id}__{other_track_id}.pkl.zip")
                                                        # load other track's frame info to check filter
                                                        other_postprocessing_data_path = join(CHANNEL_FRAME_PATH_FUNCTS["tracking_mask_postprocessing"](video_id, frame_idx, frame_id, image_version, hos_version, min_length, track_type),
                                                                                            other_track_id, f"{frame_id}__{other_track_id}.pkl")
                                                        if isfile(other_track_mask_path) and isfile(other_postprocessing_data_path):
                                                            other_postprocessing_data = read_pkl(other_postprocessing_data_path)
                                                            other_tortuosity_ema = other_postprocessing_data["tortuosity_ema"] if "tortuosity_ema" in other_postprocessing_data else np.nan
                                                            other_cd_90 = float(other_postprocessing_data["cd_avg_quantiles_ema"][90]) if "cd_avg_quantiles_ema" in other_postprocessing_data else np.nan
                                                            other_hand_ioa = float(other_postprocessing_data["track_hos_hands_intersection_ioa_ema"]) if other_postprocessing_data.get("track_hos_hands_intersection_ioa_ema") is not None else np.nan
                                                            other_filter_passed = (other_tortuosity_ema <= args.tracking_mask_merging_max_tortuosity
                                                                                  and other_cd_90 <= args.tracking_mask_merging_max_cd_q90
                                                                                  and (other_hand_ioa if not np.isnan(other_hand_ioa) else 0.0) <= args.tracking_mask_merging_max_hand_ioa)
                                                            if other_filter_passed:
                                                                other_track_mask_data = read_pkl(other_track_mask_path)
                                                                other_mask_img = other_track_mask_data["masks"]
                                                                if other_mask_img.max() > 0:
                                                                    other_track_mask_bbox = bbox_from_mask(other_mask_img)
                                                                    other_track_mask_bbox_rel = scale_box(other_track_mask_bbox,
                                                                                                        other_track_mask_data.get("track_mask_width", track_mask_width),
                                                                                                        other_track_mask_data.get("track_mask_height", track_mask_height))
                                                                    if check_boxes_intersect(track_mask_bbox_rel, other_track_mask_bbox_rel):
                                                                        *_, other_mask_contours, other_mask_hierarchy_top = cv2.findContours(other_mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                                                                        
                                                                        if other_track_id not in track_summary_dict["track_track_intersection_ious"]:
                                                                            track_summary_dict["track_track_intersection_ious"][other_track_id] = {}
                                                                            track_summary_dict["track_track_intersection_ioas"][other_track_id] = {}
                                                                            track_summary_dict["track_track_intersection_intersection_areas"][other_track_id] = {}
                                                                            track_summary_dict["track_track_intersection_union_areas"][other_track_id] = {}
                                                                            track_summary_dict["track_track_intersection_own_areas"][other_track_id] = {}


                                                                        if other_track_id not in track_summary_dict["track_track_intersection_ious_filtered"]:
                                                                            track_summary_dict["track_track_intersection_ious_filtered"][other_track_id] = {}
                                                                        
                                                                        if masks.shape[0] != other_mask_img.shape[0] or masks.shape[1] != other_mask_img.shape[1]:
                                                                            this_frame_mask_image_res = Image.fromarray(masks.astype(np.uint8))
                                                                            this_frame_mask_image_res =\
                                                                                this_frame_mask_image_res.resize((other_mask_img.shape[1], other_mask_img.shape[0]),
                                                                                                                Image.NEAREST)
                                                                            this_frame_mask_image_res = np.array(this_frame_mask_image_res)
                                                                        else:
                                                                            this_frame_mask_image_res = masks
                                                                        
                                                                        intersection_img = np.logical_and(this_frame_mask_image_res, other_mask_img)
                                                                        union_img = np.logical_or(this_frame_mask_image_res, other_mask_img)
                                                                        mask_iou = intersection_img.sum() / max(1, union_img.sum())
                                                                        mask_ioa = intersection_img.sum() / max(1, this_frame_mask_image_res.sum())
                                                                        track_summary_dict["track_track_intersection_ious"][other_track_id][frame_id] = mask_iou
                                                                        track_summary_dict["track_track_intersection_ioas"][other_track_id][frame_id] = mask_ioa
                                                                        track_summary_dict["track_track_intersection_intersection_areas"][other_track_id][frame_id] = intersection_img.sum()
                                                                        track_summary_dict["track_track_intersection_union_areas"][other_track_id][frame_id] = union_img.sum()
                                                                        track_summary_dict["track_track_intersection_own_areas"][other_track_id][frame_id] = this_frame_mask_image_res.sum()

                                                                        output_dict["track_track_intersection_ious"][other_track_id] = mask_iou
                                                                        output_dict["track_track_intersection_ioas"][other_track_id] = mask_ioa
                                                                        output_dict["track_track_intersection_intersection_areas"][other_track_id] = intersection_img.sum()
                                                                        output_dict["track_track_intersection_union_areas"][other_track_id] = union_img.sum()
                                                                        output_dict["track_track_intersection_own_areas"][other_track_id] = this_frame_mask_image_res.sum()
                                                                        
                                                                        if mask_iou >= args.tracking_mask_merging_overlap_frame_iou_fraction:
                                                                            track_summary_dict["track_track_intersection_ious_filtered"][other_track_id][frame_id] = mask_iou
                                                                            output_dict["track_track_intersection_ious_filtered"][other_track_id] = mask_iou
                                                                        
                                                                        # disabled for now; suspect performance drop
                                                                        # for other_contour_idx, other_contour in enumerate(other_mask_contours):
                                                                        #     # draw contour image for other frame
                                                                        #     other_contour_img = np.zeros((other_mask_img.shape[0], other_mask_img.shape[1]))
                                                                        #     # cv2.drawContours(other_contour_img, other_contour, -1, color=1, thickness=cv2.FILLED)
                                                                        #     cv2.fillPoly(contour_img, pts=[contour], color=1)
                                                                            
                                                                        #     # calculate IoU with contours of this frame
                                                                        #     for this_frame_contour_idx, this_frame_contour_image in enumerate(this_frame_all_contours_images):
                                                                        #         if (this_frame_contour_image.shape[0] != other_contour_img.shape[0]
                                                                        #             or this_frame_contour_image.shape[1] != other_contour_img.shape[1]):
                                                                        #             this_frame_contour_image_res = Image.fromarray(this_frame_contour_image.astype(np.uint8))
                                                                        #             this_frame_contour_image_res =\
                                                                        #                 this_frame_contour_image_res.resize((other_contour_img.shape[1], other_contour_img.shape[0]),
                                                                        #                                                     Image.NEAREST)
                                                                        #                 this_frame_contour_image_res = np.array(this_frame_contour_image_res)
                                                                        #         else:
                                                                        #             this_frame_contour_image_res = this_frame_contour_image
                                                                                
                                                                        #         intersection_img = np.logical_and(this_frame_contour_image_res, other_contour_img)
                                                                        #         union_img = np.logical_or(this_frame_contour_image_res, other_contour_img)
                                                                        #         contour_iou = intersection_img.sum() / max(1, union_img.sum())
                                                                        #         # format for "intersections": (this frame contour idx, other track id, other frame contour idx): IoU / (w*h)
                                                                        #         output_dict["contour_ious"][(this_frame_contour_idx, other_track_id, other_contour_idx)] = contour_iou
                                                    
                                            with open(join(out_dir, f"{frame_id}__{dn}.pkl"), "wb") as f:
                                                pickle.dump(output_dict, f)
                                        except ToggleableException as ex:
                                            print(f"Exception in {frame_id}:", ex)

                                
                                track_summary_dict["track_mask_width"] = track_mask_width
                                track_summary_dict["track_mask_height"] = track_mask_height
                                track_summary_dict["track_cd_avgs_avg"] = np.mean(track_cd_avgs)
                                track_summary_dict["track_cd_avgs_std"] = np.std(track_cd_avgs)
                                track_summary_dict["track_cd_stds_avg"] = np.mean(track_cd_avgs)
                                track_summary_dict["track_cd_stds_std"] = np.std(track_cd_avgs)

                                track_summary_dict["track_mask_bbox_bottoms_avg"] = np.mean(track_mask_bbox_bottoms)
                                track_summary_dict["track_mask_bbox_bottoms_std"] = np.std(track_mask_bbox_bottoms)
                                track_summary_dict["track_mask_bbox_bottoms_quantiles"] = np.quantile(track_mask_bbox_bottoms, np.arange(0.0, 1.01, 0.01))
                                
                                track_mask_bbox_bottoms_rel = np.array(track_mask_bbox_bottoms) / track_mask_height
                                track_summary_dict["track_mask_bbox_bottoms_avg_rel"] = np.mean(track_mask_bbox_bottoms_rel)
                                track_summary_dict["track_mask_bbox_bottoms_std_rel"] = np.std(track_mask_bbox_bottoms_rel)
                                track_summary_dict["track_mask_bbox_bottoms_quantiles_rel"] = np.quantile(track_mask_bbox_bottoms_rel, np.arange(0.0, 1.01, 0.01))
                                track_summary_dict["merge_track_into"] = []

                                # find intersecting tracks with which to merge

                                for intersection_track_id, intersection_data in track_summary_dict["track_track_intersection_ious_filtered"].items():
                                    # load track's summary
                                    # this will execute independently for both tracks
                                    # NOTE: we need to check the other "side" too, since this track had not yet been processed then
                                    other_tracking_mask_dir = CHANNEL_VIDEO_PATH_FUNCTS["tracking_mask_postprocessing"](video_id, image_version, hos_version, min_length, track_type)
                                    other_track_summary_path = join(other_tracking_mask_dir, intersection_track_id, f"_summary.pkl")
                                    if isfile(other_track_summary_path):
                                        other_track_summary_data = read_pkl(other_track_summary_path)

                                        num_intersections = len(intersection_data)
                                        this_track_intersection_fraction = num_intersections / max(1, len(track_summary_dict["track_mask_appearances_filter2_passed"]))
                                        # we already prefiltered the individual frames based on their IoU
                                        if this_track_intersection_fraction >= args.tracking_mask_merging_overlap_frame_count_fraction:
                                            track_summary_dict["merge_track_into"].append(intersection_track_id)
                                            print(f">>> MERGING: Will merge {dn} into {intersection_track_id}")

                                        other_track_intersection_fraction = num_intersections / max(1, len(other_track_summary_data["track_mask_appearances_filter2_passed"]))
                                        if other_track_intersection_fraction >= args.tracking_mask_merging_overlap_frame_count_fraction:
                                            other_track_summary_data["merge_track_into"].append(dn)
                                            print(f">>> MERGING: Will merge {intersection_track_id} into {dn} (retroactively)")
                                            with open(other_track_summary_path, "wb") as f:
                                                pickle.dump(other_track_summary_data, f)

                                summary_out_dir = join(CHANNEL_FRAME_PATH_FUNCTS["tracking_mask_postprocessing"](video_id, -1, None, image_version, hos_version, min_length, track_type), dn)
                                with open(join(summary_out_dir, "_summary.pkl"), "wb") as f:
                                    pickle.dump(track_summary_dict, f)


def main(arg_dict=None):
    start_time = int(time.time())
    parser = argparse.ArgumentParser()
    parser.add_argument("--generator_videos", action="append", type=str, default=None)
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument("--num_jobs", type=int, default=1)
    parser.add_argument("--ewm_span_rel_to_fps", type=float, default=0.5)
    parser.add_argument("--object_bbox_version", type=str, default=DEFAULT_OBJECT_BBOX_VERSION)
    parser.add_argument("--track_type", action="append", type=str, default=None)
    parser.add_argument("--min_length", action="append", type=str, default=None)
    parser.add_argument("--hos_version", action="append", type=str, default=None)
    parser.add_argument("--image_version", action="append", type=str, default=None)
    parser.add_argument("--track_filter", action="append", type=str, default=None)
    parser.add_argument("--hos_version_hands", type=str, default="egohos")
    parser.add_argument("--hos_version_object", type=str, default="threshold=0.9")

    parser.add_argument("--tracking_mask_merging_overlap_frame_count_fraction", type=float,
                        default=DEFAULT_TRACKING_MASK_MERGING_OVERLAP_FRAME_COUNT_FRACTION)
    parser.add_argument("--tracking_mask_merging_overlap_frame_iou_fraction", type=float,
                        default=DEFAULT_TRACKING_MASK_MERGING_OVERLAP_FRAME_IOU_FRACTION)
    
    parser.add_argument("--tracking_mask_merging_max_tortuosity", type=float,
                        default=DEFAULT_TRACKING_MASK_MAX_TORTUOSITY)
    parser.add_argument("--tracking_mask_merging_max_cd_q90", type=float,
                        default=DEFAULT_TRACKING_MASK_MAX_CD_Q90)
    parser.add_argument("--tracking_mask_merging_max_hand_ioa", type=float,
                        default=DEFAULT_TRACKING_MASK_MAX_HAND_IOA)

    # round(relative_segment_size * image diagonal size) will be size of one line segment
    parser.add_argument("--relative_segment_size", type=float, default=0.01)
    
    parser.add_argument("-f", "--f", help="Dummy argument to make ipython work", default="")
    args, _ = parser.parse_known_args(arg_dict_to_list(arg_dict))

    if args.generator_videos is None:
        args.generator_videos = get_video_list()
    else:
        args.generator_videos = [s.strip() for v in args.generator_videos for s in v.split(",")]
    
    if args.track_type is None:
        args.track_type = ["left_hand", "right_hand", "object"]
    else:
        args.track_type = [s.strip() for v in args.track_type for s in v.split(",")]

    if args.min_length is None:
        args.min_length = get_available_tracking_mask_min_lengths()
    else:
        args.min_length = [s.strip() for v in args.min_length for s in v.split(",")]

    if args.hos_version is None:
        args.hos_version = get_available_hos_versions()
    else:
        args.hos_version = [s.strip() for v in args.hos_version for s in v.split(",")]
    
    if args.image_version is None:
        args.image_version = get_available_image_versions()
    else:
        args.image_version = [s.strip() for v in args.image_version for s in v.split(",")]
    
    if args.track_filter is not None:
        args.track_filter = [s.strip() for v in args.track_filter for s in v.split(",")]

    timestamp = time.time()

    args.num_jobs = min(args.num_jobs, len(args.generator_videos))

    if args.num_jobs == 1:
        process(0, (args, timestamp, args.generator_videos))
    else:
        with Pool(processes=args.num_jobs) as pool:
            paths_split = list(map(lambda a: list(map(str, a)), np.array_split(args.generator_videos, args.num_jobs)))
            pool.starmap(process, enumerate(zip([args] * args.num_jobs, [timestamp] * args.num_jobs, paths_split)))


if __name__ == "__main__":
    main()
