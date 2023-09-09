import os
from functools import partial
from tqdm.contrib.concurrent import process_map

import sys

sys.path.append("../")
from utils import extract_frames


if __name__ == "__main__":
    clip_ids = [
        file_name[:-4]
        for file_name in os.listdir(
            os.path.join(os.environ["SCRATCH"], "ego4d_data/v2", "clips")
        )
    ]

    output_folder_path = os.path.join(os.environ["SCRATCH"], "ego4d_data/v2", "frames")

    extract_frames_partial = partial(
        extract_frames, output_folder_path=output_folder_path
    )

    process_map(extract_frames_partial, clip_ids, max_workers=8)
