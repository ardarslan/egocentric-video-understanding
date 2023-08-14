# %%

# partly from https://github.com/facebookresearch/segment-anything/blob/main/notebooks/automatic_mask_generator_example.ipynb

import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from os.path import dirname, isdir, isfile, join
import pickle
from PIL import Image
from scipy.sparse import csr_matrix
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
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
from utils.imaging import bbox_from_mask, scale_box
from utils.io import read_pkl


def main(arg_dict=None):
    start_time = int(time.time())
    parser = argparse.ArgumentParser()
    parser.add_argument("--generator_videos", action="append", type=str, default=None)
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument("--hos_version", type=str, default=DEFAULT_HOS_VERSION)
    parser.add_argument("--full_image", action="store_true")
    parser.add_argument("--tracking_mask", action="store_true")
    parser.add_argument("--min_length", type=int, default=DEFAULT_TRACKING_MASK_MIN_LENGTH)
    parser.add_argument("--image_version", type=str, default="full_image")
    parser.add_argument("--focus_path", type=str, default=None)
    parser.add_argument("--focus_threshold", type=float, default=DEFAULT_FOCUS_THRESHOLD)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("-f", "--f", help="Dummy argument to make ipython work", default="")
    args, _ = parser.parse_known_args(arg_dict_to_list(arg_dict))

    if args.output_dir is None:
        args.output_dir = SEGMENTATION_MASK_DATA_DIR

    if args.tracking_mask:
        output_subdir = f"tracking_mask__{args.image_version}__{args.hos_version}__min_length={args.min_length}"
    elif args.full_image:
        output_subdir = {"image": "full_image"}.get(args.image_version, args.image_version)
    else:
        output_subdir = args.hos_version

    sam_checkpoint = join(CHECKPOINTS_PATH, "sam_vit_h_4b8939.pth")
    model_type = "vit_h"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=args.device)

    mask_generator = SamAutomaticMaskGenerator(sam)

    gen = get_action_recognition_frame_gen(subsets=["val"],
                                           videos=[s.strip() for v in args.generator_videos for s in v.split(",")]
                                                   if args.generator_videos is not None else None)

    if args.focus_path not in [None, ""]:
        with open(args.focus_path, "rb") as f:
            focus_dict = pickle.load(f)
    else:
        focus_dict = None

    focus_threshold = args.focus_threshold

    out_dict = {}

    global_frame_idx = 0

    if args.generator_videos is None:
        args.generator_videos = get_video_list()
    else:
        args.generator_videos = [s.strip() for v in args.generator_videos for s in v.split(",")]

    for video_id in args.generator_videos:
        
        reader = VideoReader(get_video_path(video_id), get_extracted_frame_dir_path(video_id),
                             assumed_fps=EK_ASSUMED_FPS)
        virtual_frame_count = reader.get_virtual_frame_count()
        range_obj = range(virtual_frame_count)

        for frame_idx in tqdm((range_obj)):
            frame_id = fmt_frame(video_id, frame_idx)
            if args.image_version is None or args.image_version in ["", "image", "full_image"]:
                im = reader.get_frame(frame_idx)
                pil_img = Image.fromarray(im)
            else:
                im_path = CHANNEL_FRAME_PATH_FUNCTS["inpainted"](video_id, frame_idx, frame_id, args.image_version)
                if not isfile(im_path):
                    print(f"Frame not found: {im_path}")
                    continue
                
                with Image.open(im_path) as im_pre:
                    pil_img = im_pre.copy()
                    im = np.array(pil_img)
            
            frame_outputs = []
            pkl_out_path = join(args.output_dir, output_subdir, video_id, f"{frame_id}.pkl")
            if isfile(pkl_out_path + ".zip"):
                continue

            os.makedirs(dirname(pkl_out_path), exist_ok=True)

            if args.tracking_mask:
                tracking_mask_dir_path = CHANNEL_FRAME_PATH_FUNCTS["tracking_mask"](video_id, frame_idx, frame_id, args.image_version, args.hos_version, args.min_length, "object")
                tracking_mask_postprocessing_dir_path = CHANNEL_FRAME_PATH_FUNCTS["tracking_mask_postprocessing"](video_id, frame_idx, frame_id, args.image_version, args.hos_version, args.min_length, "object")

                if not isdir(tracking_mask_dir_path):
                    print(f"{frame_id}: Dir not found: {tracking_mask_dir_path}")
                    continue

                # get tracks
                for dn in os.listdir(tracking_mask_dir_path):
                    track_dir_path = join(tracking_mask_dir_path, dn)
                    if not isdir(track_dir_path):
                        continue
                    
                    track_file_path = join(track_dir_path, f"{frame_id}__{dn}.pkl.zip")
                    if not isfile(track_file_path):
                        continue

                    track_mask_data = read_pkl(track_file_path)

                    if not track_mask_data["masks"].max():
                        continue

                    # calculate bbox
                                
                    *_, mask_contours, mask_hierarchy_top = cv2.findContours(track_mask_data["masks"],
                                                                             cv2.RETR_EXTERNAL,
                                                                             cv2.CHAIN_APPROX_NONE)
                    for contour_idx, contour in enumerate(mask_contours):
                        # for each contour separately, split into segments and check curvature change
                        mask_img = np.zeros(track_mask_data["masks"].shape)
                        cv2.fillPoly(mask_img, pts=[contour], color=1)
                        
                        box = bbox_from_mask(mask_img)
                        # rescale
                        orig_width = track_mask_data["image_width"]
                        orig_height = track_mask_data["image_height"]
                        box_rescaled = scale_box(box, orig_width, orig_height,
                                                 target_width=pil_img.width, target_height=pil_img.height)
                        if min(int(box_rescaled[2] - box_rescaled[0]), int(box_rescaled[3] - box_rescaled[1])) > 0:
                            pil_img_cropped = pil_img.crop(box_rescaled)
                            masks = mask_generator.generate(np.array(pil_img_cropped))
                            masks_out = [{**mask, "segmentation": csr_matrix(mask["segmentation"])} for mask in masks]
                            # cls 1: active object
                            frame_outputs.append({"box": box_rescaled, "cls": 1, "masks": masks_out,
                                                    "image_width": im.shape[1], "image_height": im.shape[0],
                                                    "track_id": dn, "contour_idx": contour_idx,
                                                    "contour_box_orig": box, "contour_box_rescaled": box_rescaled})

            elif args.full_image:
                masks = mask_generator.generate(np.array(im))
                masks_out = [{**mask, "segmentation": csr_matrix(mask["segmentation"])} for mask in masks]
                frame_outputs.append({"box": np.array([0, 0, im.shape[1], im.shape[0]]), "cls": -1, "masks": masks_out,
                                      "image_width": im.shape[1], "image_height": im.shape[0]})
            else:
                # lambda video_id, frame_idx, frame_id, version
                hand_pkl_zip_path = CHANNEL_FRAME_PATH_FUNCTS["hos_object"](video_id, frame_idx, frame_id, args.hos_version)

                if not isfile(hand_pkl_zip_path):
                    print(f"{frame_id}: File not found: {hand_pkl_zip_path}")
                    continue
                
                print(f"Found: {hand_pkl_zip_path}")
                zip_obj = zipfile.ZipFile(hand_pkl_zip_path)
                pkl_bytes = zip_obj.read(zip_obj.namelist()[0])
                pkl = pickle.loads(pkl_bytes)

                boxes = pkl["instances"].pred_boxes.tensor.cpu().numpy()
                classes = pkl["instances"].pred_classes
                zipped = list(zip(boxes, classes))

                for box, cls in zipped:
                    if cls != 1:  # 0: hand; 1: active object
                        continue
                    
                    pil_img_cropped = pil_img.crop(box)
                    masks = mask_generator.generate(np.array(pil_img_cropped))
                    masks_out = [{**mask, "segmentation": csr_matrix(mask["segmentation"])} for mask in masks]
                    frame_outputs.append({"box": box, "cls": cls, "masks": masks_out,
                                          "image_width": im.shape[1], "image_height": im.shape[0]})

            if focus_dict is not None:
                if frame_id not in focus_dict:
                    print(f"{frame_id}: not in focus dict")
                    continue

                if focus_dict[frame_id] < focus_threshold:
                    continue
                else:
                    print(f"{frame_id}: keeping (focus: {focus_dict[frame_id]=} >= {focus_threshold=})")
            
            with zipfile.ZipFile(pkl_out_path + ".zip", "w",
                                 zipfile.ZIP_DEFLATED, False) as zip_file:
                zip_file.writestr(os.path.basename(pkl_out_path), pickle.dumps(frame_outputs))
            
            global_frame_idx += 1


if __name__ == "__main__":
    main()
