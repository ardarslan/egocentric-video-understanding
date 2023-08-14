# %%

import numpy as np
from os.path import join, dirname
from PIL import Image
import sys
from tqdm import tqdm

sys.path.append(dirname(dirname(__file__)))

from data_handling.video_reader import VideoReader
from data_handling.specific.ek100 import *
from scipy.ndimage import binary_dilation
from utils.globals import *
from utils.imaging import *
from utils.io import read_pkl

video_id = "P09_07"
max_width = 640
max_height = 480

mask_output_dir = "/mnt/scratch/agavryushin/Thesis/data/EGO_HOS_test_vis/"
os.makedirs(mask_output_dir, exist_ok=True)


reader = VideoReader(
    get_video_path(video_id), get_extracted_frame_dir_path(video_id), assumed_fps=-1
)
for frame_idx in tqdm(range(len(reader))):
    if frame_idx < 500:
        continue
    if frame_idx >= 1250:
        break

    img_np = reader.get_frame(frame_idx)
    frame_id = fmt_frame(video_id, frame_idx)
    img = Image.fromarray(img_np)
    img.thumbnail((max_width, max_height))
    frame_fn = f"frame_{frame_idx:07}.png"

    hand_mask_fp = f"/mnt/scratch/agavryushin/EgoHOS/testimages/pred_twohands_P09_07/frame_{frame_idx:07}.png"
    obj_mask_fp = f"/mnt/scratch/agavryushin/EgoHOS/testimages/pred_obj1_P09_07/frame_{frame_idx:07}.png"
    with Image.open(hand_mask_fp) as hand_mask_img:
        with Image.open(obj_mask_fp) as obj_mask_img:
            hand_mask = np.array(hand_mask_img) > 0
            obj_mask = np.array(obj_mask_img) > 0
            sup_img = superimpose_colored_mask(img, hand_mask, (255, 0, 0))
            sup_img = superimpose_colored_mask(sup_img, obj_mask, (0, 255, 0))
            sup_img.save(join(mask_output_dir, frame_fn))
