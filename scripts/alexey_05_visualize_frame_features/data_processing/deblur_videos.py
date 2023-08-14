# %%

import argparse
import numpy as np
import os
from os.path import dirname, join
import pickle
from PIL import Image
import sys
import torch
from tqdm import tqdm


sys.path.append(dirname(dirname(__file__)))
from data_handling.specific.ek100 import *
from utils.args import arg_dict_to_list
from utils.globals import *

os.chdir(SHIFTNET_PATH)

from basicsr.models.archs.gshift_deblur1 import GShiftNet


# from https://github.com/dasongli1/Shift-Net/blob/main/inference/test_deblur.py
def numpy2tensor(input_seq, rgb_range=1.0):
    tensor_list = []
    for img in input_seq:
        img = np.array(img).astype("float64")
        np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))  # HWC -> CHW
        tensor = torch.from_numpy(np_transpose).float()  # numpy -> tensor
        tensor.mul_(rgb_range / 255)  # (0,255) -> (0,1)
        tensor_list.append(tensor)
    stacked = torch.stack(tensor_list).unsqueeze(0)
    return stacked


def main(arg_dict=None):
    SIZE_MUST_MOD = 4

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir", type=str, default=join(ROOT_PATH, "data", "EK_frames_deblurred")
    )
    parser.add_argument("--before_frames", type=int, default=10)
    parser.add_argument("--after_frames", type=int, default=10)

    parser.add_argument("--max_width", type=int, default=1280)  # 1280
    parser.add_argument("--max_height", type=int, default=720)  # 720

    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    args, _ = parser.parse_known_args(arg_dict_to_list(arg_dict))

    gen = get_action_recognition_frame_gen(
        subsets=["val"],
        add_deltas=(
            [-(i + 1) for i in range(args.before_frames)]
            + [(i + 1) for i in range(args.after_frames)]
        ),
    )

    # instantiate models

    with torch.no_grad():
        net = GShiftNet(future_frames=args.before_frames, past_frames=args.after_frames)
        net.load_state_dict(
            torch.load(
                join(SHIFTNET_PATH, "pretrained_models", "net_gopro_deblur.pth")
            )["params"]
        )
        net.half()
        net = net.to(args.device)
        net.eval()

        for frame_data in tqdm(gen):
            frame_id = frame_data["frame_id"]
            video_id = frame_data["video_id"]
            output_dir = join(args.output_dir, video_id)
            os.makedirs(output_dir, exist_ok=True)
            output_path = join(output_dir, f"{frame_id}.jpg")
            original_output_path = join(output_dir, f"{frame_id}_original.jpg")
            if os.path.isfile(output_path):
                continue

            h, w, c = frame_data["image"].shape
            new_h, new_w = h - h % SIZE_MUST_MOD, w - w % SIZE_MUST_MOD
            inputs = []
            for delta, img in frame_data["delta_images"].items():
                img_pil = Image.fromarray(img[:new_h, :new_w, :])
                img_pil.thumbnail((args.max_width, args.max_height))
                if delta == 0:
                    img_pil.save(original_output_path)
                inputs.append(np.array(img_pil))

            in_tensor = numpy2tensor(inputs).to(args.device)
            output = net(in_tensor.half())[0]
            output_array = (
                output.float().clamp(0, 1.0).permute(1, 2, 0).cpu().numpy() * 255
            )
            output_img = Image.fromarray(output_array.astype(np.uint8))
            output_img.save(output_path)

            print(f"Saved {output_path} (original: {original_output_path})")


if __name__ == "__main__":
    main()
