import os
import cv2
import torch
import numpy as np

from typing import List


class FrameFeatureExtractor(object):
    def __init__(self, args):
        pass
        # if args.frame_feature_name == "blip2_vqa":
        # self.number_of_frames_per_input = 1
        # self.window_center_frame_stride = 6
        # self.column_names = [
        #     "frame_index",
        #     "question",
        #     "caption",
        #     "caption_sbert_embedding",
        #     "encoder_output",
        # ]

    # elif args.frame_feature_name == "video_blip":
    #     self.number_of_frames_per_input = 9
    #     self.target_fps = 4.5
    #     self.window_center_frame_stride = 9
    #     self.column_names = [
    #         "frame_index",
    #         "question",
    #         "caption",
    #         "caption_sbert_embedding",
    #         "encoder_output"
    #     ]

    def get_new_input(self, current_embedding_index: int, cap: cv2.VideoCapture):
        # original_fps = cap.get(cv2.CAP_PROP_FPS)
        # if self.number_of_frames_per_input > 1:
        #     cursor_stride_in_same_window = int(
        #         np.round(original_fps / float(self.target_fps))
        #     )
        #     current_input_center_frame_index = int(
        #         np.round(
        #             current_input_start_frame_index
        #             + original_fps
        #             / float(self.target_fps)
        #             * self.number_of_frames_per_input
        #             / 2
        #         )
        #     )
        # else:

        frame_index = int(
            current_embedding_index * cap.get(cv2.CAP_PROP_FRAME_COUNT) / 1024.0
        )

        current_input = {
            "frame_index": frame_index,
            "frames": [],
        }

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index - 1)
        success, frame = cap.read()  # (HWC, BGR)
        # if self.number_of_frames_per_input > 1:
        #     current_cursor += cursor_stride_in_same_window
        if not success:
            return None, None

        current_input["frames"].append(frame)

        current_embedding_index += 1
        return current_embedding_index, current_input

    def predictor_function(self, frame_index: int, frames: List[np.array]):
        pass

    @staticmethod
    def save_results(
        caption_sbert_embeddings,
        encoder_outputs,
        output_folder_path,
        clip_uid,
    ):
        os.makedirs(
            os.path.join(output_folder_path, "caption_sbert_embeddings"),
            exist_ok=True,
        )

        os.makedirs(
            os.path.join(output_folder_path, "encoder_outputs"),
            exist_ok=True,
        )

        torch.save(
            caption_sbert_embeddings,
            os.path.join(
                output_folder_path, "caption_sbert_embeddings", clip_uid + ".pt"
            ),
        )

        torch.save(
            encoder_outputs,
            os.path.join(output_folder_path, "encoder_outputs", clip_uid + ".pt"),
        )
