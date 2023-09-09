import os
import cv2


def extract_frames(clip_id: str, output_folder_path: str):
    if not os.path.exists(os.path.join(output_folder_path, clip_id, "end.txt")):
        cap = cv2.VideoCapture(
            os.path.join(os.environ["SCRATCH"], "ego4d_data/v2/clips", f"{clip_id}.mp4")
        )
        print(f"Extracting frames for clip with ID: {clip_id}...")
        os.system(f"rm -rf {os.path.join(output_folder_path, clip_id)}")
        os.makedirs(
            os.path.join(output_folder_path, clip_id),
            exist_ok=True,
        )
        success = True
        frame_id = 0
        while success:
            success, frame = cap.read()
            if not success:
                break
            cv2.imwrite(
                os.path.join(
                    output_folder_path,
                    clip_id,
                    str(frame_id).zfill(6) + ".jpg",
                ),
                frame,
                [
                    int(cv2.IMWRITE_JPEG_QUALITY),
                    80,
                ],
            )
            frame_id += 1
        with open(
            os.path.join(output_folder_path, clip_id, "end.txt"),
            "w",
        ) as writer:
            writer.write("\n")
    else:
        print(f"Using pre-extracted frames for clip with ID: {clip_id}.")
