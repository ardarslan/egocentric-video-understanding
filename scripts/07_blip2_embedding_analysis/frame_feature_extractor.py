import os
import cv2
import numpy as np
import pandas as pd

from typing import List


class FrameFeatureExtractor(object):
    def __init__(self, args):
        if args.frame_feature_name == "blip2_vqa":
            self.number_of_frames_per_input = 1
            self.target_fps = None
            self.window_center_frame_stride = 6
            self.column_names = [
                "frame_index",
                "question",
                "caption",
                "caption_sbert_embedding",
                "encoder_output"
            ]
        elif args.frame_feature_name == "video_blip":
            self.number_of_frames_per_input = 9
            self.target_fps = 4.5
            self.window_center_frame_stride = 9
            self.column_names = [
                "frame_index",
                "question",
                "caption",
                "caption_sbert_embedding",
                "encoder_output"
            ]

    def get_new_input(self, current_input_start_frame_index: int, cap: cv2.VideoCapture):
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        if self.number_of_frames_per_input > 1:
            cursor_stride_in_same_window = int(np.round(original_fps / float(self.target_fps)))
            current_input_center_frame_index = int(np.round(current_input_start_frame_index + original_fps / float(self.target_fps) * self.number_of_frames_per_input / 2))
        else:
            current_input_center_frame_index = current_input_start_frame_index

        current_input = {
            "frame_index": current_input_center_frame_index,
            "frames": []
        }
        current_cursor = current_input_start_frame_index
        for _ in range(self.number_of_frames_per_input):
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_cursor - 1)
            success, frame = cap.read() # (HWC, BGR)
            if self.number_of_frames_per_input > 1:
                current_cursor += cursor_stride_in_same_window
            if not success:
                return None, None
            current_input["frames"].append(frame)

        current_input_start_frame_index += self.window_center_frame_stride
        return current_input_start_frame_index, current_input

    def predictor_function(
        self, frame_index: int, frames: List[np.array]
    ):
        pass

    @staticmethod
    def save_results(
        clip_uid,
        results_list,
        output_folder_path,
        column_names,
        output_file_name,
    ):
        results_df = pd.DataFrame(
            data=results_list,
            columns=column_names,
        )

        os.makedirs(
            os.path.join(output_folder_path, clip_uid),
            exist_ok=True,
        )
        results_df.to_csv(
            os.path.join(
                output_folder_path, clip_uid, output_file_name
            ),
            index=False,
            sep="\t",
        )
