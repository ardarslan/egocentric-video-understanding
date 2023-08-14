import argparse
import cv2
from PIL import Image
import numpy as np
import importlib
import matplotlib.pyplot as plt
from matplotlib import animation
import os
from os.path import dirname, isdir, isfile
import pickle
import sys
import torch
from tqdm import tqdm

sys.path.append(dirname(dirname(__file__)))

from data_handling.video_reader import VideoReader
from data_handling.specific.ek100 import *
from utils.args import arg_dict_to_list
from utils.globals import *
from utils.exceptions import ToggleableException
from utils.io import read_pkl

os.chdir(E2FGVI_PATH)

from core.utils import to_tensors


def main(arg_dict=None):
    # export CUDA_VISIBLE_DEVICES=4,6 && python test.py --video=/mnt/scratch/agavryushin/Thesis/data/EK_e2fgvi_test_video_red/ --mask=/mnt/scratch/agavryushin/Thesis/data/EK_e2fgvi_test_mask_red/ --ckpt=/mnt/scratch/agavryushin/E2FGVI/release_model/E2FGVI-CVPR22.pth --model=e2fgvi --neighbor_stride=12 --step=25 --savefps=60

    start_time = int(time.time())
    parser = argparse.ArgumentParser()
    parser.add_argument("--generator_videos", action="append", type=str, default=None)
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument("--output_dir", type=str, default=INPAINTING_DATA_DIR)
    parser.add_argument("--set_size", action="store_true", default=False)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument(
        "--model", type=str, choices=["e2fgvi", "e2fgvi_hq"], default="e2fgvi"
    )
    parser.add_argument("--hos_version", type=str, default="egohos")
    parser.add_argument("--step", type=int, default=10)
    parser.add_argument("--num_ref", type=int, default=8)
    parser.add_argument("--neighbor_stride", type=int, default=12)
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

    if args.model == "e2fgvi":
        checkpoint_path = "release_model/E2FGVI-CVPR22.pth"
    elif args.model == "e2fgvi_hq":
        checkpoint_path = "release_model/E2FGVI-HQ-CVPR22.pth"

    ref_length = args.step  # ref_step
    num_ref = args.num_ref
    neighbor_stride = args.neighbor_stride

    # sample reference frames from the whole video
    def get_ref_index(f, neighbor_ids, length):
        ref_index = []
        if num_ref == -1:
            for i in range(0, length, ref_length):
                if i not in neighbor_ids:
                    ref_index.append(i)
        else:
            start_idx = max(0, f - ref_length * (num_ref // 2))
            end_idx = min(length, f + ref_length * (num_ref // 2))
            for i in range(start_idx, end_idx + 1, ref_length):
                if i not in neighbor_ids:
                    if len(ref_index) > num_ref:
                        break
                    ref_index.append(i)
        return ref_index

    # read frame-wise masks
    def read_mask(mpath, size):
        masks = []
        mnames = os.listdir(mpath)
        mnames.sort()
        for mp in mnames:
            m = Image.open(os.path.join(mpath, mp))
            m = m.resize(size, Image.NEAREST)
            m = np.array(m.convert("L"))
            m = np.array(m > 0).astype(np.uint8)
            m = cv2.dilate(
                m, cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)), iterations=4
            )
            masks.append(Image.fromarray(m * 255))
        return masks

    #  read frames from video
    def read_frame_from_videos(args):
        vname = args.video
        frames = []
        if args.use_mp4:
            vidcap = cv2.VideoCapture(vname)
            success, image = vidcap.read()
            count = 0
            while success:
                image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                frames.append(image)
                success, image = vidcap.read()
                count += 1
        else:
            lst = os.listdir(vname)
            lst.sort()
            fr_lst = [vname + "/" + name for name in lst]
            for fr in fr_lst:
                image = cv2.imread(fr)
                image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                frames.append(image)
        return frames

    # set up models
    device = args.device

    if args.model == "e2fgvi":
        size = (432, 240)
    elif args.width is not None or args.height is not None:
        size = (args.width, args.height)
    else:
        size = None

    net = importlib.import_module("model." + args.model)
    model = net.InpaintGenerator().to(device)
    data = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(data)
    print(f"Loading model from: {checkpoint_path}")
    model.eval()

    for video_id in args.generator_videos:
        # 1st pass: get track boundaries
        # gen = get_action_recognition_frame_gen(subsets=["val"], videos=[video_id])
        reader = VideoReader(
            get_video_path(video_id),
            get_extracted_frame_dir_path(video_id),
            assumed_fps=EK_ASSUMED_FPS,
        )

        range_obj = range(len(reader))

        print()
        print("Processing step 1/3: reading frames and masks")
        print()

        hos_version = args.hos_version

        frames = []
        frames_np = []
        masks = []
        for frame_idx in tqdm(range_obj):
            frame_id = fmt_frame(video_id, frame_idx)
            try:
                img_np = reader.get_frame(
                    frame_idx
                ).copy()  # avoid "negative stride" error
            except ToggleableException as ex:
                print(f"Could not read frame {frame_idx} of {video_id}:", ex)
                continue
            img = Image.fromarray(img_np)

            if (size[0] is not None and img.width != size[0]) or (
                size[1] is not None and img.height != size[1]
            ):
                img = img.resize(
                    (size[0] or img.width, size[1] or img.height), Image.BILINEAR
                )
                frames_np.append(np.array(img))
            else:
                frames_np.append(img_np)

            frames.append(img)

            # read mask

            mask = None
            path_hands = CHANNEL_FRAME_PATH_FUNCTS["hos_hands"](
                video_id, frame_idx, frame_id, hos_version
            )

            if isfile(path_hands):
                pkl = read_pkl(path_hands)

                if hos_version == "egohos":
                    mask = np.array(np.logical_or(pkl == 1, pkl == 2)).astype(np.uint8)
                else:
                    mask = np.zeros(frames_np[0].shape)
                    for cls, handside, mask_inst, box in zip(
                        pkl["instances"].pred_classes,
                        pkl["instances"].pred_handsides,
                        pkl["instances"].pred_masks,
                        pkl["instances"].pred_boxes,
                    ):
                        if cls == 0:  # 0: hand
                            if mask is None:
                                mask = mask_inst > 0
                            else:
                                mask |= mask_inst > 0

                    if mask is not None:
                        mask = mask.astype(np.uint8)

            if mask is None:
                mask = np.zeros(frames_np[0].shape)

            if (size[0] is not None and mask.shape[1] != size[0]) or (
                size[1] is not None and mask.shape[0] != size[1]
            ):
                img = Image.fromarray(mask)
                img = img.resize(size, Image.NEAREST)
                mask = np.array(img.convert("L"))

            mask = cv2.dilate(
                mask, cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)), iterations=4
            )
            masks.append(Image.fromarray(mask * 255))

            # !!!!!!!!!!
            if frame_idx >= 9000:
                break

        h = frames_np[0].shape[0]
        w = frames_np[0].shape[1]

        imgs = to_tensors()(frames).unsqueeze(0) * 2 - 1
        binary_masks = [
            np.expand_dims((np.array(m) != 0).astype(np.uint8), 2) for m in masks
        ]
        masks = to_tensors()(masks).unsqueeze(0)
        imgs, masks = imgs.to(device), masks.to(device)

        print()
        print("Processing step 2/3: inpainting")
        print()

        comp_frames = [None] * len(frames)

        for f in tqdm(range(0, len(frames), neighbor_stride)):
            neighbor_ids = [
                i
                for i in range(
                    max(0, f - neighbor_stride),
                    min(len(frames), f + neighbor_stride + 1),
                )
            ]
            ref_ids = get_ref_index(f, neighbor_ids, len(frames))
            selected_imgs = imgs[:1, neighbor_ids + ref_ids, :, :, :]
            selected_masks = masks[:1, neighbor_ids + ref_ids, :, :, :]
            with torch.no_grad():
                masked_imgs = selected_imgs * (1 - selected_masks)
                mod_size_h = 60
                mod_size_w = 108
                h_pad = (mod_size_h - h % mod_size_h) % mod_size_h
                w_pad = (mod_size_w - w % mod_size_w) % mod_size_w
                masked_imgs = torch.cat([masked_imgs, torch.flip(masked_imgs, [3])], 3)[
                    :, :, :, : h + h_pad, :
                ]
                masked_imgs = torch.cat([masked_imgs, torch.flip(masked_imgs, [4])], 4)[
                    :, :, :, :, : w + w_pad
                ]
                pred_imgs, _ = model(masked_imgs, len(neighbor_ids))
                pred_imgs = pred_imgs[:, :, :h, :w]
                pred_imgs = (pred_imgs + 1) / 2
                pred_imgs = pred_imgs.cpu().permute(0, 2, 3, 1).numpy() * 255
                for i in range(len(neighbor_ids)):
                    idx = neighbor_ids[i]
                    img = np.array(pred_imgs[i]).astype(np.uint8) * binary_masks[
                        idx
                    ] + frames[idx] * (1 - binary_masks[idx])
                    if comp_frames[idx] is None:
                        comp_frames[idx] = img
                    else:
                        comp_frames[idx] = (
                            comp_frames[idx].astype(np.float32) * 0.5
                            + img.astype(np.float32) * 0.5
                        )

        print()
        print("Processing step 3/3: saving inpainted images")
        print()

        video_output_dir = join(args.output_dir, f"e2fgvi_{args.hos_version}", video_id)
        os.makedirs(video_output_dir, exist_ok=True)
        for frame_idx, frame_np in tqdm(enumerate(comp_frames)):
            frame_id = fmt_frame(video_id, frame_idx)
            jpg_output_path = join(video_output_dir, f"{frame_id}.jpg")
            Image.fromarray(frame_np.astype(np.uint8)).save(jpg_output_path)

        print(f"Inpainted images saved to {video_output_dir}")


if __name__ == "__main__":
    main()
