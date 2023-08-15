import argparse
import multiprocessing as mp
from multiprocessing.pool import Pool
import numpy as np
from os.path import dirname, isfile, join
from PIL import Image
import sys
from tqdm import tqdm

sys.path.append(dirname(dirname(__file__)))

from data_handling.specific.ek100 import *
from data_handling.video_reader import VideoReader
from utils.args import arg_dict_to_list
from utils.globals import *


def process(process_idx, data):
    args, video_ids = data
    num_processed_frames = 0
    for video_idx, video_id in tqdm(enumerate(video_ids)):
        if args.output_dir is None:
            output_dir = get_extracted_frame_dir_path(video_id)
        else:
            output_dir = join(args.output_dir, video_id)
        # if os.path.isdir(output_dir):
        #    continue

        try:
            gen = get_action_recognition_frame_gen(
                subsets=["val"],
                videos=[video_id],
                action_frames_only=not args.full_video,
                max_width=args.max_width,
                max_height=args.max_height,
            )
            os.makedirs(output_dir, exist_ok=True)  # only create directory here!
            for frame_data in gen:
                # progress_bar.set_description(frame_data["frame_id"])
                output_path = join(
                    output_dir, f"frame_{(frame_data['original_frame_idx']):07}.jpg"
                )
                if isfile(output_path):
                    continue

                img = Image.fromarray(frame_data["image"])
                img.save(output_path)

                num_processed_frames += 1
                if num_processed_frames % 100 == 0:
                    print(
                        f"Process {process_idx}: at {frame_data['frame_id']} ({num_processed_frames} frames , {video_idx}/{len(video_ids)} videos)"
                    )
        except Exception as ex:
            print(f"Error processing {video_id}:", ex)
            continue


def main(arg_dict=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--generator_videos", action="append", type=str, default=None)
    parser.add_argument("--max_width", type=int, default=None)
    parser.add_argument("--max_height", type=int, default=None)
    parser.add_argument("--num_jobs", type=int, default=5)
    parser.add_argument("--full_video", action="store_true", default=False)
    args = parser.parse_args(arg_dict_to_list(arg_dict))

    current_reader = None

    if args.generator_videos is None:
        args.generator_videos = get_video_list()
    else:
        args.generator_videos = extend_video_list(
            [s.strip() for v in args.generator_videos for s in v.split(",")]
        )

    args.num_jobs = min(args.num_jobs, len(args.generator_videos))

    if args.num_jobs == 1:
        process(0, (args, args.generator_videos))
    else:
        with Pool(processes=args.num_jobs) as pool:
            paths_split = list(
                map(
                    lambda a: list(map(str, a)),
                    np.array_split(args.generator_videos, args.num_jobs),
                )
            )
            pool.starmap(process, enumerate(zip([args] * args.num_jobs, paths_split)))


if __name__ == "__main__":
    main()
