import argparse
import os
from os.path import join, dirname, isfile, isdir
import pickle
import sys
import time
import zipfile

sys.path.append(dirname(dirname(__file__)))


def main(arg_dict=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--generator_videos", action="append", type=str, default=None)
    parser.add_argument("--input_dir", type=str, default=None)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument(
        "-f", "--f", help="Dummy argument to make ipython work", default=""
    )

    if arg_dict is None:
        args, _ = parser.parse_known_args()
    else:
        from utils.args import arg_dict_to_list

        args, _ = parser.parse_known_args(arg_dict_to_list(arg_dict))

    if args.input_dir is None:
        from utils.globals import VIDEO_DEPTH_ESTIMATION_DATA_DIR

        args.input_dir = VIDEO_DEPTH_ESTIMATION_DATA_DIR

    if args.output_path is None or args.output_path == "":
        args.output_path = f"packed_video_depth_estimation_{time.time()}.zip"

    if args.generator_videos is None:
        from data_handling.specific.ek100 import get_video_list

        args.generator_videos = get_video_list()
    else:
        args.generator_videos = [
            s.strip() for v in args.generator_videos for s in v.split(",")
        ]

    # zip_file.writestr(os.path.basename(pkl_out_path), pickle.dumps(frame_outputs))
    dirs_to_add = []  # structure: (path in zip, local path)
    for video_id in args.generator_videos:
        video_depth_output_path = join(args.input_dir, video_id)
        if isdir(video_depth_output_path):
            for subdir in os.listdir(video_depth_output_path):
                subdir_output_path = join(video_depth_output_path, subdir)
                if isdir(subdir_output_path):
                    aggregated_output_dirname = next(
                        filter(
                            lambda n: n.startswith("R"), os.listdir(subdir_output_path)
                        ),
                        None,
                    )
                    if aggregated_output_dirname is not None and isdir(
                        midas_output_path := join(
                            subdir_output_path, aggregated_output_dirname
                        )
                    ):
                        dirs_to_add.append(
                            (
                                "/".join((video_id, subdir, aggregated_output_dirname)),
                                midas_output_path,
                            )
                        )

                    depth_output_path = join(
                        subdir_output_path, "depth_midas2", "depth"
                    )
                    if isdir(depth_output_path):
                        dirs_to_add.append(
                            (
                                "/".join((video_id, subdir, "depth_midas2", "depth")),
                                depth_output_path,
                            )
                        )

    os.makedirs(dirname(args.output_path), exist_ok=True)

    with zipfile.ZipFile(
        args.output_path, "w", zipfile.ZIP_DEFLATED, allowZip64=True
    ) as zip_file:
        for zip_dir_path, dir_path in dirs_to_add:
            for root, dirs, files in os.walk(dir_path):
                for fn in files:
                    path = join(root, fn)
                    zip_path = path.replace(dir_path, zip_dir_path)
                    zip_file.write(path, arcname=zip_path)

                    print(f"Added {zip_path}")

    print(f"Output written to {args.output_path}")


if __name__ == "__main__":
    main()
