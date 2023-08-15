# %%
# adapted from https://github.com/gaomingqi/Track-Anything/

import argparse
import cv2
import matplotlib.pyplot as plt
import multiprocessing as mp
from multiprocessing.pool import Pool
import numpy as np
import os
from os.path import join, dirname, isfile
import pickle
from PIL import Image
import requests
from scipy.sparse import csr_matrix
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import sys
import time
import torch
from tqdm import tqdm
import zipfile

sys.path.append(dirname(dirname(__file__)))

from data_handling.video_reader import VideoReader
from data_handling.specific.ek100 import *
from utils.args import arg_dict_to_list
from utils.globals import *
from utils.io import read_pkl

os.chdir(TRACK_ANYTHING_PATH)
sys.path.append(join(TRACK_ANYTHING_PATH, "tracker"))

from tools.interact_tools import SamControler
from tracker.base_tracker import BaseTracker


def download_checkpoint(url, folder, filename):
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, filename)

    if not os.path.exists(filepath):
        print("download checkpoints ......")
        response = requests.get(url, stream=True)
        with open(filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        print("download successfully!")

    return filepath


def process(process_idx, data):
    with torch.no_grad():
        args, timestamp, video_track_infos_kv = data

        if args.jobs_devices is None or len(args.jobs_devices) <= process_idx:
            device = DEFAULT_DEVICE
        else:
            device = args.jobs_devices[process_idx]

        SAM_checkpoint_dict = {
            "vit_h": "sam_vit_h_4b8939.pth",
            "vit_l": "sam_vit_l_0b3195.pth",
            "vit_b": "sam_vit_b_01ec64.pth",
        }
        SAM_checkpoint_url_dict = {
            "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
            "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
            "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        }

        sam_checkpoint = SAM_checkpoint_dict[args.sam_model_type]
        sam_checkpoint_url = SAM_checkpoint_url_dict[args.sam_model_type]
        xmem_checkpoint = "XMem-s012.pth"
        xmem_checkpoint_url = (
            "https://github.com/hkchengrex/XMem/releases/download/v1.0/XMem-s012.pth"
        )

        checkpoint_dir = join(CHECKPOINTS_PATH, "tam")

        sam_checkpoint = download_checkpoint(
            sam_checkpoint_url, checkpoint_dir, sam_checkpoint
        )
        xmem_checkpoint = download_checkpoint(
            xmem_checkpoint_url, checkpoint_dir, xmem_checkpoint
        )

        sam_controller = SamControler(sam_checkpoint, args.sam_model_type, device)
        xmem = BaseTracker(xmem_checkpoint, device=device)

        csv = get_dataset_csv("val")

        for video_track_id, track_info in video_track_infos_kv:
            video_id = video_track_id[0]
            track_id = video_track_id[1]
            reader = VideoReader(
                get_video_path(video_id),
                get_extracted_frame_dir_path(video_id),
                assumed_fps=EK_ASSUMED_FPS,
            )
            virtual_frame_count = reader.get_virtual_frame_count()

            last_seen = track_info["start"]
            track_length = track_info["end"] - track_info["start"] + 1
            if track_length < args.min_length:
                print("track_length < args.min_length")
                continue

            track_type = track_info["type"].replace("hos_", "")

            tracking_mask_path = CHANNEL_FRAME_PATH_FUNCTS["tracking_mask"](
                video_id,
                None,
                None,
                args.image_version,
                args.hos_version,
                args.min_length,
                track_type,
            )
            if os.path.isdir(join(tracking_mask_path, track_id)):
                print("Dir exists:", join(tracking_mask_path, track_id))
                continue

            xmem.clear_memory()

            # determine
            video_section_rows = csv[csv["video_id"] == video_id]

            if args.use_gt_segment_boundaries:
                stop_at_frame = virtual_frame_count
                last_stop_frame = -1

                for row in video_section_rows.iterrows():
                    if (
                        row[1].loc["start_frame"]
                        <= track_info["start"]
                        < row[1].loc["stop_frame"]
                    ):
                        stop_at_frame = row[1].loc["stop_frame"]
                    elif (
                        last_stop_frame
                        <= track_info["start"]
                        < row[1].loc["start_frame"]
                    ):
                        stop_at_frame = row[1].loc["start_frame"]
                        break
                    last_stop_frame = row[1].loc["stop_frame"]
            else:
                stop_at_frame = virtual_frame_count

            for frame_idx in (
                progress_bar := tqdm(range(track_info["start"], int(LARGE)))
            ):
                try:
                    progress_bar.ncols = 0
                    if frame_idx >= virtual_frame_count:
                        print("frame_idx >= virtual_frame_count")
                        break

                    if frame_idx >= stop_at_frame:
                        print("frame_idx >= stop_at_frame")
                        break

                    if frame_idx >= track_info["end"] + 1:
                        if frame_idx - last_seen > (
                            args.vanish_tolerance_frames_object
                            if "obj" in track_type
                            else args.vanish_tolerance_frames_hands
                        ):
                            break

                    frame_id = fmt_frame(video_id, frame_idx)

                    if args.image_version in [None, "image"]:
                        image = reader.get_frame(
                            frame_idx
                        ).copy()  # avoid "negative stride" error
                    elif args.image_version.startswith("inpaint"):
                        inpainted_version = args.image_version.split("_", 1)[1]
                        path = CHANNEL_FRAME_PATH_FUNCTS["inpainted"](
                            video_id, frame_idx, frame_id, inpainted_version
                        )
                        if not isfile(path):
                            print(f"Image source not found:", path)
                            continue
                        with Image.open(path) as image_pil:
                            image = np.array(image_pil)
                    else:
                        raise NotImplementedError()

                    image_shape_orig = [reader.video_height, reader.video_width]
                    if args.max_width is not None or args.max_height is not None:
                        image_res = Image.fromarray(image)
                        image_res.thumbnail(
                            (args.max_width or LARGE, args.max_height or LARGE)
                        )
                        image = np.array(image_res)

                    template_mask = None

                    if frame_idx == track_info["start"]:
                        # get template mask

                        box = track_info["frame_boxes"][track_info["start"]]
                        width_ratio = image.shape[1] / image_shape_orig[1]
                        height_ratio = image.shape[0] / image_shape_orig[0]
                        scaled_box = np.array(
                            [
                                box[0] * width_ratio,
                                box[1] * height_ratio,
                                box[2] * width_ratio,
                                box[3] * height_ratio,
                            ]
                        )

                        sam_controller.sam_controler.reset_image()
                        sam_controller.sam_controler.set_image(image)
                        (
                            masks,
                            scores,
                            logits,
                        ) = sam_controller.sam_controler.predictor.predict(
                            box=scaled_box, multimask_output=False
                        )

                        template_mask = masks

                    masks, logits, painted_images = xmem.track(
                        image,
                        template_mask[0].astype(int)
                        if template_mask is not None
                        else None,
                    )

                    found_mask = (masks > 0).max()
                    if found_mask:
                        last_seen = frame_idx

                    pkl_output_path = join(
                        tracking_mask_path, track_id, f"{frame_id}__{track_id}.pkl"
                    )
                    output_dir = dirname(pkl_output_path)
                    os.makedirs(output_dir, exist_ok=True)

                    frame_outputs = {
                        "timestamp": timestamp,
                        "args": args,
                        "masks": masks,
                        "logits": logits,
                        "type": track_info["type"],
                        "image_width": image.shape[1],
                        "image_height": image.shape[0],
                    }

                    # Image.fromarray(painted_images).save(f"__test__{frame_id}.jpg")

                    with zipfile.ZipFile(
                        pkl_output_path + ".zip", "w", zipfile.ZIP_DEFLATED, False
                    ) as zip_file:
                        zip_file.writestr(
                            os.path.basename(pkl_output_path),
                            pickle.dumps(frame_outputs),
                        )
                except ToggleableException as ex:
                    print(f"Exception processing {video_id}, frame {frame_idx}:", ex)
                    break


def main(arg_dict=None):
    start_time = int(time.time())
    parser = argparse.ArgumentParser()
    parser.add_argument("--generator_videos", action="append", type=str, default=None)
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument("--sam_model_type", type=str, default="vit_h")
    parser.add_argument("--track_type", action="append", type=str, default=None)
    parser.add_argument(
        "--min_length", type=int, default=DEFAULT_TRACKING_MASK_MIN_LENGTH
    )
    parser.add_argument("--max_width", type=int, default=None)
    parser.add_argument("--max_height", type=int, default=None)
    parser.add_argument("--num_jobs", type=int, default=1)
    parser.add_argument("--jobs_devices", action="append", type=str, default=None)
    parser.add_argument("--vanish_tolerance_frames_object", type=int, default=60)
    parser.add_argument("--vanish_tolerance_frames_hands", type=int, default=600)
    parser.add_argument("--hos_version", type=str, default="threshold=0.9")
    parser.add_argument("--image_version", type=str, default="image")
    parser.add_argument("--use_gt_segment_boundaries", action="store_true")

    parser.add_argument(
        "-f", "--f", help="Dummy argument to make ipython work", default=""
    )
    args, _ = parser.parse_known_args(arg_dict_to_list(arg_dict))

    if args.generator_videos is None:
        args.generator_videos = get_video_list()
    else:
        args.generator_videos = [
            s.strip() for v in args.generator_videos for s in v.split(",")
        ]

    if args.jobs_devices is not None:
        args.jobs_devices = [s.strip() for v in args.jobs_devices for s in v.split(",")]

    if args.track_type is not None:
        args.track_type = [s.strip() for v in args.track_type for s in v.split(",")]
    else:
        args.track_type = ["left_hand", "right_hand", "object"]

    print(f"{args.track_type=}")

    timestamp = time.time()

    video_track_infos = {}

    for video_id in args.generator_videos:
        # 1st pass: get track boundaries
        # gen = get_action_recognition_frame_gen(subsets=["val"], videos=[video_id])
        reader = VideoReader(
            get_video_path(video_id),
            get_extracted_frame_dir_path(video_id),
            assumed_fps=EK_ASSUMED_FPS,
        )
        virtual_frame_count = reader.get_virtual_frame_count()
        range_obj = range(virtual_frame_count)

        print(f"{video_id=} {virtual_frame_count=}")

        for frame_idx in range_obj:
            frame_id = fmt_frame(video_id, frame_idx)
            tracking_bbox_path = CHANNEL_FRAME_PATH_FUNCTS["tracking_bbox"](
                video_id, frame_idx, frame_id
            )
            if not isfile(tracking_bbox_path):
                print(f"not isfile(tracking_bbox_path):", tracking_bbox_path)
                continue

            tracking_bbox_data = read_pkl(tracking_bbox_path)

            channel_store = (
                tracking_bbox_data["tracks_segmented"]["gt"]
                if args.use_gt_segment_boundaries
                else tracking_bbox_data["tracks"]
            )
            for channel_name, channel_tracks in channel_store.items():
                if channel_name not in [
                    f"hos_object_{args.hos_version}",
                    f"hos_left_hand_{args.hos_version}",
                    f"hos_right_hand_{args.hos_version}",
                ]:
                    continue

                if (
                    channel_name == f"hos_object_{args.hos_version}"
                    and "object" not in args.track_type
                    and "obj" not in args.track_type
                ):
                    continue

                if (
                    f"left_hand_{args.hos_version}" in channel_name
                    and "left_hand" not in args.track_type
                    and "hands" not in args.track_type
                ):
                    continue

                if (
                    f"right_hand_{args.hos_version}" in channel_name
                    and "right_hand" not in args.track_type
                    and "hands" not in args.track_type
                ):
                    continue

                for track_id, track_data in channel_tracks.items():
                    key = (video_id, track_id)
                    if key not in video_track_infos:
                        video_track_infos[key] = {
                            "start": frame_idx,
                            "end": frame_idx,
                            "type": channel_name.rsplit("_", 1)[0],
                            "frame_boxes": {frame_idx: track_data["box"]},
                        }
                    else:
                        video_track_infos[key]["end"] = frame_idx
                        video_track_infos[key]["frame_boxes"][frame_idx] = track_data[
                            "box"
                        ]

    print(f"{len(video_track_infos)=}")
    # 2nd pass: track
    sorted_keys = sorted(list(video_track_infos.keys()))
    sorted_values = [video_track_infos[k] for k in sorted_keys]
    sorted_kv = list(zip(sorted_keys, sorted_values))
    if args.num_jobs == 1:
        process(0, (args, timestamp, sorted_kv))
    else:
        data_split = []
        split_share = len(sorted_keys) // args.num_jobs
        for job_idx in range(args.num_jobs):
            if job_idx == args.num_jobs - 1:
                data_split.append(sorted_kv[(job_idx * split_share) :])
            else:
                data_split.append(
                    sorted_kv[(job_idx * split_share) : ((job_idx + 1) * split_share)]
                )
        with Pool(processes=args.num_jobs) as pool:
            pool.starmap(
                process,
                enumerate(
                    zip([args] * args.num_jobs, [timestamp] * args.num_jobs, data_split)
                ),
            )


if __name__ == "__main__":
    main()
