# %%

import argparse
import os
from os.path import dirname, join
import pickle
from PIL import Image
import sys
from tqdm import tqdm


sys.path.append(dirname(dirname(__file__)))
from data_handling.specific.ek100 import *
from utils.args import arg_dict_to_list
from utils.globals import *

os.chdir(FRANKMOCAP_PATH)

from handmocap.hand_bbox_detector import *


def main(arg_dict=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default=HAND_BBOX_DATA_DIR)
    parser.add_argument("--full_video", action="store_true", default=False)
    parser.add_argument('--generator_videos', action='append', type=str, default=None)
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    args, _ = parser.parse_known_args(arg_dict_to_list(arg_dict))

    hd = HandBboxDetector("ego_centric", args.device)

    if args.generator_videos is None:
        args.generator_videos = get_video_list()
    else:
        args.generator_videos = [s.strip() for v in args.generator_videos for s in v.split(",")]


    for video_id in args.generator_videos:
        try:
            gen = get_action_recognition_frame_gen(subsets=["val"],
                                                action_frames_only=not args.full_video,
                                                videos=[video_id])
            for frame_data in tqdm(gen):
                frame_id = frame_data['frame_id']
                output_dir = join(args.output_dir, video_id)
                output_path = join(output_dir, f"{frame_id}.pkl")
                if os.path.isfile(output_path):
                    continue
                
                try:
                    bgr_image = frame_data["image"][:, :, ::-1]
                    detections = hd.detect_hand_bbox(bgr_image)
                    if isinstance(detections, dict):
                        detections["image_width"] = bgr_image.shape[1]
                        detections["image_height"] = bgr_image.shape[0]
                    elif isinstance(detections, tuple) and len(detections) >= 3 and isinstance(detections[2], list):
                        for idx in range(len(detections[2])): 
                            detections[2][idx]["image_width"] = bgr_image.shape[1]
                            detections[2][idx]["image_height"] = bgr_image.shape[0]
                    
                    os.makedirs(output_dir, exist_ok=True)
                    with open(output_path, "wb") as f:
                        pickle.dump(detections, f)
                except AssertionError as ex:
                    print(f"AssertionError for {frame_id}:", ex)
                    continue
                except Exception as ex:
                    print(f"Exception for {frame_id}:", ex)
                    continue
        except Exception as ex:
            print(f"Exception for {video_id}:", ex)
            continue


if __name__ == "__main__":
    main()
