# %%
import json
import os
from os.path import dirname, isfile, join
import sys
from PIL import Image
import numpy as np


sys.path.append(dirname(dirname(__file__)))

from utils.args import arg_dict_to_list
from utils.globals import CHANNEL_FRAME_PATH_FUNCTS

BATCH_SIZE = 16
IMGS_PER_ROW = 2
IMGS_PER_SAMPLE = 11



for root, dirs, files in os.walk("/data/data3/agavryushin/tempclr/TempCLR_release/FreiHAND_model_P01_12/summary/"):
    for fn in files:
        json_path = join(root, fn)
        if not json_path.endswith(".json"):
            continue
        img_path = json_path.replace(".json", ".png")
        with open(json_path) as f:
            with Image.open(img_path) as img:
                w = img.width
                h = img.height

                # get the y boundaries

                y_boundaries = []
                start_y_coord = -1
                for y in range(0, h):
                    active = start_y_coord != -1
                    found_bright = False
                    for x in range(0, w, 10):
                        pixel_color = img.getpixel((x, y))
                        if pixel_color[0] != 0 or pixel_color[1] != 0 or pixel_color[2] != 0:
                            found_bright = True
                            break
                    if active and not found_bright:
                        y_boundaries.append([start_y_coord, y])
                        start_y_coord = -1
                    elif not active and found_bright:
                        start_y_coord = y

                if start_y_coord != -1:
                    y_boundaries.append([start_y_coord, h])

                data = json.load(f)
                for img_idx, img_data in enumerate(data):
                    column_group = img_idx % IMGS_PER_ROW
                    row = img_idx // IMGS_PER_ROW

                    x1 = column_group * (w / IMGS_PER_ROW)
                    y1 = y_boundaries[row][0]
                    x2 = (column_group + 1) * (w / IMGS_PER_ROW)
                    y2 = y_boundaries[row][1]
                    group_box = [x1, y1, x2, y2]
                    w_section = (x2 - x1) / IMGS_PER_SAMPLE
                    mesh_img_box = [x1 + w_section * 3, y1, x1 + w_section * 4, y2]
                    img_crop = img.crop(mesh_img_box)
                    # leftmost column sometimes still contains image
                    for y in range(img_crop.height):
                        img_crop.putpixel((0, y), (0, 0, 0))

                    # resize back to original scale

                    orig_w = img_data["orig_bbox"][2] - img_data["orig_bbox"][0]
                    orig_h = img_data["orig_bbox"][3] - img_data["orig_bbox"][1]
                    scale = img_data["orig_bbox_size"] / max(img_crop.width, img_crop.height)

                    img_crop = img_crop.resize((round(img_crop.width * scale), round(img_crop.height * scale)), Image.BILINEAR)

                    # now, make sure centers align

                    img_top_left_coord_x = img_data["orig_center"][0] - img_crop.width / 2
                    img_top_left_coord_y = img_data["orig_center"][1] - img_crop.height / 2
                    img_crop_box = (img_top_left_coord_x, img_top_left_coord_y,
                                    img_top_left_coord_x + img_crop.width,
                                    img_top_left_coord_y + img_crop.height)
                    
                    # paste to original image

                    with Image.open(img_data["image_name"]) as original_img:
                        img_crop_np = np.array(img_crop)
                        mask_np = np.any(img_crop_np != 0, axis=-1).astype(np.uint8)
                        mask_img = Image.fromarray(mask_np * 255)
                        original_img.paste(img_crop, (round(img_top_left_coord_x), round(img_top_left_coord_y)),
                                           mask_img)
                        no_bg_img = Image.new("RGB", (2 * original_img.width, original_img.height), (0, 0, 0))
                        no_bg_img.paste(img_crop, (original_img.width + round(img_top_left_coord_x), round(img_top_left_coord_y)),
                                        mask_img)

                    img_crop_out_path = f"{img_path}_{img_idx}.png"
                    original_img_out_path = f"{img_path}_{img_idx}_overlapped.jpg"
                    img_crop.save(img_crop_out_path)
                    original_img.save(original_img_out_path)
                    video_id = img_data["video_id"]
                    frame_id = img_data["frame_id"]

                    if video_id.startswith(frame_id):
                        temp = frame_id
                        frame_id = video_id
                        video_id = temp

                    no_bg_img_path = CHANNEL_FRAME_PATH_FUNCTS["hand_mesh_vis"](video_id, -1, frame_id, "tempclr")
                    if isfile(no_bg_img_path):
                        with Image.open(no_bg_img_path) as existing_img:
                            existing_img.paste(no_bg_img, (0, 0))
                            existing_img.save(no_bg_img_path)
                    else:
                        os.makedirs(dirname(no_bg_img_path), exist_ok=True)
                        no_bg_img.save(no_bg_img_path)

                    print(f"{img_crop_out_path=}")
                    print(f"{original_img_out_path=}")
                    
# %%
