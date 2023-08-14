import os
from os.path import dirname, isfile, join
import pickle
from PIL import Image
import sys
from tqdm import tqdm

sys.path.append(dirname(dirname(__file__)))
from data_handling.specific.ek100 import *
from utils.args import arg_dict_to_list
from utils.globals import *

os.chdir(UNIDET_PATH)

import argparse
from detectron2.config import get_cfg
from detectron2.data.detection_utils import (
    convert_PIL_to_numpy,
    _apply_exif_orientation,
)
from functools import partial
import multiprocessing as mp
import numpy as np
from unidet.config import add_unidet_config
from unidet.predictor import UnifiedVisualizationDemo


def setup_cfg(args):
    # taken from demo.py
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_unidet_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = (
        args.confidence_threshold
    )
    cfg.freeze()
    return cfg


def main(arg_dict=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file",
        default="configs/Unified_learned_OCIM_RS200_6x+2x.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--output_dir", type=str, default=OBJECT_BBOX_DATA_DIR)

    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=0.25,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=["MODEL.WEIGHTS", "models/Unified_learned_OCIM_RS200_6x+2x.pth"],
        nargs=argparse.REMAINDER,
    )
    parser.add_argument("--generator_videos", action="append", type=str, default=None)
    parser.add_argument("--full_video", action="store_true")
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument("--image_version", type=str, default="image")
    args = parser.parse_args(arg_dict_to_list(arg_dict))
    cfg = setup_cfg(args)
    demo = UnifiedVisualizationDemo(cfg)

    if args.generator_videos is None:
        args.generator_videos = get_video_list()
    else:
        args.generator_videos = [
            s.strip() for v in args.generator_videos for s in v.split(",")
        ]

    class_labels = demo.metadata.get("thing_classes")
    for video_id in args.generator_videos:
        reader = VideoReader(
            get_video_path(video_id),
            get_extracted_frame_dir_path(video_id),
            assumed_fps=EK_ASSUMED_FPS,
        )
        virtual_frame_count = reader.get_virtual_frame_count()

        # generator = get_action_recognition_frame_gen(["val"], videos=[video_id],
        #                                             action_frames_only=not args.full_video)

        # for frame_data in tqdm(generator):
        output_dir = join(args.output_dir, "unidet_" + args.image_version, video_id)
        os.makedirs(output_dir, exist_ok=True)
        for frame_idx in tqdm(range(virtual_frame_count)):
            frame_id = fmt_frame(video_id, frame_idx)
            output_path = join(output_dir, f"{frame_id}.pkl")
            if isfile(output_path):
                print(f"isfile: {output_path}")
                continue

            if args.image_version is None or args.image_version in [
                "",
                "image",
                "full_image",
            ]:
                im = reader.get_frame(frame_idx)
                pil_img = Image.fromarray(im)
                cv2_image = im[:, :, ::-1]
            else:
                im_path = CHANNEL_FRAME_PATH_FUNCTS["inpainted"](
                    video_id, frame_idx, frame_id, args.image_version
                )
                if not isfile(im_path):
                    print(f"Frame not found: {im_path}")
                    continue

                with Image.open(im_path) as im_pre:
                    pil_img = im_pre.copy()
                    cv2_image = np.array(pil_img)[:, :, ::-1]

            ret_boxes = []
            ret_classes = []
            ret_scores = []

            # cv2_image = convert_PIL_to_numpy(_apply_exif_orientation(pil_image), format="BGR")
            # cv2_image = frame_data["image"][:, :, ::-1]
            predictions, vis = demo.run_on_image(cv2_image)

            # vis.save(f"{frame_id}.jpg")

            boxes = predictions["instances"].pred_boxes.tensor.cpu().numpy().tolist()
            ret_boxes.append(boxes)
            ret_classes.append(
                [
                    class_labels[class_num]
                    for class_num in predictions["instances"].pred_classes
                ]
            )
            scores = predictions["instances"].scores.cpu().numpy().tolist()
            ret_scores.append(scores)

            output_dict = {
                "boxes": ret_boxes,
                "classes": ret_classes,
                "scores": ret_scores,
                "image_width": cv2_image.shape[1],
                "image_height": cv2_image.shape[0],
            }
            os.makedirs(output_dir, exist_ok=True)
            with open(output_path, "wb") as f:
                pickle.dump(output_dict, f)


if __name__ == "__main__":
    main()
