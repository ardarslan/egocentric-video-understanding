# %%

from PIL import Image
from os.path import join, dirname
import sys
from tqdm import tqdm

sys.path.append(dirname(dirname(__file__)))

from data_handling.video_reader import VideoReader
from data_handling.specific.ek100 import *
from scipy.ndimage import binary_dilation
from utils.args import arg_dict_to_list
from utils.globals import *
from utils.exceptions import ToggleableException
from utils.io import read_pkl

video_id = "P09_07"
max_width = 640
max_height = 480
hos_version = "egohos"

video_output_dir = "/mnt/scratch/agavryushin/Thesis/data/EK_e2fgvi_test_video_red/"
mask_output_dir = "/mnt/scratch/agavryushin/Thesis/data/EK_e2fgvi_test_mask_red/"
os.makedirs(video_output_dir, exist_ok=True)
os.makedirs(mask_output_dir, exist_ok=True)


reader = VideoReader(get_video_path(video_id), get_extracted_frame_dir_path(video_id), assumed_fps=-1)
for frame_idx in tqdm(range(len(reader))):
    if frame_idx < 500:
        continue
    if frame_idx >= 1250:
        break

    try:
        img_np = reader.get_frame(frame_idx).copy()  # avoid "negative stride" error
    except ToggleableException as ex:
        print(f"Could not read frame {frame_idx} of {video_id}:", ex)
        continue
    frame_id = fmt_frame(video_id, frame_idx)
    img = Image.fromarray(img_np)
    img.thumbnail((max_width, max_height))
    frame_fn = f"frame_{frame_idx:07}.jpg"
    img.save(join(video_output_dir, frame_fn))

    universal_mask = np.zeros((img.height, img.width)).astype(bool)
    mask_path = CHANNEL_FRAME_PATH_FUNCTS["hos_hands"](video_id, frame_idx, frame_id, hos_version)
    pkl = read_pkl(mask_path)

    if hos_version == "egohos":
        mask = np.logical_or(pkl == 1, pkl == 2).astype(np.uint8)
        if mask.shape[0] != img.height or mask.shape[1] != img.width:
            mask = np.array(Image.fromarray(mask * 255).resize((img.width, img.height), Image.NEAREST))

        universal_mask |= mask.astype(bool)
    else:
        for cls, handside, mask, box in zip(pkl["instances"].pred_classes, pkl["instances"].pred_handsides, pkl["instances"].pred_masks, pkl["instances"].pred_boxes):
            if cls == 0:  # 0: hand
                # 0: left; 1: right
                handside = handside.argmax().item()
                mask = mask.cpu().numpy()
                if mask.shape[0] != img.height or mask.shape[1] != img.width:
                    mask = np.array(Image.fromarray(mask.astype(np.uint8) * 255).resize((img.width, img.height), Image.NEAREST))
                
                universal_mask |= (mask > 0)
    
    universal_mask = binary_dilation(universal_mask, iterations=10)
    Image.fromarray(universal_mask.astype(np.uint8) * 255).save(join(mask_output_dir, frame_fn))
