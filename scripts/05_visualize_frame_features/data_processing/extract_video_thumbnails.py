import argparse
from os.path import dirname, join
from PIL import Image
import sys

sys.path.append(dirname(dirname(__file__)))

from data_handling.specific.ek100 import *
from data_handling.video_reader import VideoReader
from tqdm import tqdm
from utils.args import arg_dict_to_list
from utils.globals import *


def main(arg_dict=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--generator_videos", action="append", type=str, default=None)
    parser.add_argument("--max_width", type=int, default=None)
    parser.add_argument("--max_height", type=int, default=None)
    args = parser.parse_args(arg_dict_to_list(arg_dict))

    current_reader = None

    if args.generator_videos is None:
        args.generator_videos = get_video_list()
    else:
        args.generator_videos = [s.strip() for v in args.generator_videos for s in v.split(",")]

    for video_id in tqdm(args.generator_videos):
        del current_reader
        current_reader = VideoReader(get_video_path(video_id), get_extracted_frame_dir_path(video_id), assumed_fps=-1)
        
        video_len = len(current_reader)
        mid_frame = current_reader.get_frame(video_len // 2)  # len(VideoReader) uses the actual number of frames

        img = Image.fromarray(mid_frame)
        img.thumbnail((args.max_width or 1e10, args.max_height or 1e10))

        img.save(join(args.output_dir, f"{video_id}.jpg"))


if __name__ == "__main__":
    main()
