import os
import cv2
import argparse
from PIL import Image
from tqdm import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder_path", type=str)
    parser.add_argument("--output_folder_path", type=str)
    parser.add_argument("--max_width", type=int)
    parser.add_argument("--max_height", type=int)
    args = parser.parse_args()

    video_file_names = [
        video_file_name
        for video_file_name in os.listdir(args.input_folder_path)
        if video_file_name.split(".")[-1] == "mp4"
    ]

    os.makedirs(args.output_folder_path, exist_ok=True)

    for video_file_name in tqdm(video_file_names):
        input_video_file_path = os.path.join(args.input_folder_path, video_file_name)
        cap = cv2.VideoCapture(input_video_file_path)
        video_len = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(video_len // 2 - 1))
        success, mid_frame = cap.read()
        mid_frame = Image.fromarray(mid_frame[:, :, ::-1])
        mid_frame.thumbnail((args.max_width, args.max_height))
        mid_frame.save(
            os.path.join(
                args.output_folder_path, f"{video_file_name.split('.')[0]}.jpg"
            )
        )
        cap.release()
        del cap
