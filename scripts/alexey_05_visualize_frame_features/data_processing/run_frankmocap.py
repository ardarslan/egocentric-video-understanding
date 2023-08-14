# %%

import argparse
import numpy as np
import os
from os.path import dirname, join
import pickle
from PIL import Image
import pytorch3d
import sys
import time
import ssl
import sys
import torch
from tqdm import tqdm
import zipfile

ssl._create_default_https_context = ssl._create_unverified_context

sys.path.append(dirname(dirname(__file__)))
from data_handling.specific.ek100 import *
from utils.args import arg_dict_to_list
from utils.globals import *
from utils.exceptions import ToggleableException

os.chdir(FRANKMOCAP_PATH)

from demo.demo_options import DemoOptions
import mocap_utils.general_utils as gnu
import mocap_utils.demo_utils as demo_utils

from handmocap.hand_mocap_api import HandMocap
from handmocap.hand_bbox_detector import HandBboxDetector

import renderer.image_utils as imu
from renderer.viewer2D import ImShow

# based on https://github.com/facebookresearch/frankmocap/blob/main/demo/demo_handmocap.py


def run_hand_mocap(args, bbox_detector, hand_mocap, visualizer, generator_videos=None):
    # Set up input data (images or webcam)

    # assert args.out_dir is not None, "Please specify output dir to store the results"

    if generator_videos is None:
        generator_videos = get_video_list()
    else:
        generator_videos = [s.strip() for v in generator_videos for s in v.split(",")]

    cur_frame = args.start_frame

    for video_id in generator_videos:
        generator = get_action_recognition_frame_gen(
            ["val"],
            videos=[video_id],
            max_width=1280,
            max_height=720,
            action_frames_only=not args.full_video,
        )  # , max_width=640, max_height=480

        if args.out_dir in [None, ""]:
            args.out_dir = CHANNEL_VIDEO_PATH_FUNCTS["hand_mesh"](
                video_id, args.version
            )

        while True:
            # load data
            frame_data = next(generator)
            if frame_data is None:
                break
            frame_id = frame_data["frame_id"]
            img_original_bgr = frame_data["image"][:, :, ::-1]
            image_path = frame_id + ".jpg"  # extension needed

            try:
                pkl_out_path = join(
                    args.out_dir, "mocap", f"{frame_id}_prediction_result.pkl"
                )  # join(args.out_dir, "mocap", f"{frame_id}_prediction_result.pkl")

                if os.path.isfile(pkl_out_path) or os.path.isfile(
                    pkl_out_path + ".zip"
                ):
                    continue

                cur_frame += 1
                if img_original_bgr is None or cur_frame > args.end_frame:
                    break

                # Input images has other body part or hand not cropped.
                # Use hand detection model & body detector for hand detection

                detect_output = bbox_detector.detect_hand_bbox(img_original_bgr.copy())
                (
                    body_pose_list,
                    body_bbox_list,
                    hand_bbox_list,
                    raw_hand_bboxes,
                ) = detect_output

                # save the obtained body & hand bbox to json file
                if args.save_bbox_output:
                    demo_utils.save_info_to_json(
                        args, image_path, body_bbox_list, hand_bbox_list
                    )

                if len(hand_bbox_list) < 1:
                    print(f"No hand detected: {image_path}")
                    continue

                # Hand Pose Regression
                pred_output_list = hand_mocap.regress(
                    img_original_bgr, hand_bbox_list, add_margin=True
                )
                assert len(hand_bbox_list) == len(body_bbox_list)
                assert len(body_bbox_list) == len(pred_output_list)

                # save the image (we can make an option here)
                if args.out_dir is not None:
                    # extract mesh for rendering (vertices in image space and faces) from pred_output_list
                    pred_mesh_list = demo_utils.extract_mesh_from_output(
                        pred_output_list
                    )

                    # visualize
                    res_img = visualizer.visualize(
                        np.zeros_like(img_original_bgr),
                        pred_mesh_list=pred_mesh_list,
                        hand_bbox_list=hand_bbox_list,
                    )
                    demo_utils.save_res_img(args.out_dir, image_path, res_img)

                # save predictions to pkl
                if args.save_pred_pkl:
                    demo_type = "hand"
                    demo_utils.save_pred_to_pkl(
                        args,
                        demo_type,
                        image_path,
                        body_bbox_list,
                        hand_bbox_list,
                        pred_output_list,
                    )

                    # compress

                    with zipfile.ZipFile(
                        pkl_out_path + ".zip", "w", zipfile.ZIP_DEFLATED, False
                    ) as zip_file:
                        with open(pkl_out_path, "rb") as pkl_file:
                            zip_file.writestr(
                                os.path.basename(pkl_out_path),
                                pickle.dumps(
                                    {
                                        **pickle.load(pkl_file),
                                        "image_width": img_original_bgr.shape[1],
                                        "image_height": img_original_bgr.shape[0],
                                    }
                                ),
                            )

                    os.unlink(pkl_out_path)

                print(f"Processed: {image_path.rsplit('.', 1)[0]}")
            except ToggleableException as ex:
                print(f"Exception when processing {frame_id}:", ex)


def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--output_dir", type=str, default=join(ROOT_PATH, "data", "EK_frames_deblurred"))
    # parser.add_argument("--device", type=str, default="cuda:2")
    # args, _ = parser.parse_known_args()

    with torch.no_grad():
        options = DemoOptions()
        options.parser.add_argument(
            "--generator_videos", action="append", type=str, default=None
        )
        options.parser.add_argument("--full_video", action="store_true", default=False)
        options.parser.add_argument(
            "--version", type=str, default=DEFAULT_HAND_MESH_VERSION
        )
        options.parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
        args = options.parse()
        args.use_smplx = True
        args.save_pred_pkl = True

        device = args.device or "cuda"
        assert torch.cuda.is_available(), "Current version only supports GPU"

        # Set Bbox detector
        bbox_detector = HandBboxDetector(args.view_type, device)

        # Set Mocap regressor
        hand_mocap = HandMocap(args.checkpoint_hand, args.smpl_dir, device=device)

        # Set Visualizer
        if args.renderer_type in ["pytorch3d", "opendr"]:
            from renderer.screen_free_visualizer import Visualizer
        else:
            from renderer.visualizer import Visualizer
        visualizer = Visualizer(args.renderer_type)

        # run
        run_hand_mocap(
            args, bbox_detector, hand_mocap, visualizer, args.generator_videos
        )


if __name__ == "__main__":
    main()
