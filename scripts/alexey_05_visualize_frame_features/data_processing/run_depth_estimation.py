import argparse
import cv2
import os
from os.path import basename, dirname, isdir, isfile, join, splitext
import sys

sys.path.append(dirname(dirname(__file__)))

try:
    from utils.globals import CVD2_PATH
    os.chdir(CVD2_PATH)
except:
    pass


def main(arg_dict=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--visualization_interval", type=int, default=100)
    parser.add_argument("--generator_videos", action="append", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    
    if arg_dict is None:
        args, _ = parser.parse_known_args()
    else:
        from utils.args import arg_dict_to_list
        args, _ = parser.parse_known_args(arg_dict_to_list(arg_dict))

    if args.input_dir is None:
        from utils.globals import DEPTH_ESTIMATION_DATA_DIR
        args.input_dir = DEPTH_ESTIMATION_DATA_DIR
    
    if args.generator_videos is None:
        from data_handling.specific.ek100 import get_video_list
        args.generator_videos = get_video_list()
    else:
        args.generator_videos = [s.strip() for v in args.generator_videos for s in v.split(",")]

    if args.output_dir is None or args.output_dir == "":
        from utils.globals import CVD2_PATH
        args.output_dir = join(CVD2_PATH, "outputs")

    print(f"Extracting actionwise videos...")

    try:
        from data_processing.extract_actionwise_videos import main as extract_actionwise_videos
        video_ids_and_paths = extract_actionwise_videos({"generator_videos": args.generator_videos})
    except Exception as ex:
        print("Exception when extracting actionwise videos:", ex)

    print(f"Running depth estimation...")

    for video_id, path in video_ids_and_paths:
        print()
        print(f'*** Processing "{video_id}"... ***')
        print()

        video_cap = cv2.VideoCapture(path)
        video_len = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))

        output_path = join(args.output_dir, video_id, splitext(basename(path))[0])
        os.system((f"python main.py --video_file {path} --path {output_path} --save_intermediate_depth_streams_freq 1 " +
                   f" --num_epochs 0 --post_filter --opt.adaptive_deformation_cost 10 --frame_range 0-{video_len-1} " +
                   f"--save_depth_visualization"))


if __name__ == "__main__":
    main()
