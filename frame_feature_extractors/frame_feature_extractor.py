import os
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from typing import List


class FrameFeatureExtractor(ABC):
    def __init__(self):
        pass

    @staticmethod
    def get_inputs(cap, batch_size: int, frame_feature_name: str, output_subfolder_path: str, number_of_frames_per_video_part: int, stride_for_processing_frames: int, global_frame_index):
        if frame_feature_name == "ego_hos":
            if not os.path.exists(os.path.join(output_subfolder_path, "unidet_features.tsv")):
                raise Exception("To extract ego_hos features, unidet features should be extracted first.")
            if not os.path.exists(os.path.join(output_subfolder_path, "gsam_features.tsv")):
                raise Exception("To extract ego_hos features, gsam features should be extracted first.")
            unidet_features = pd.read_csv(os.path.join(output_subfolder_path, "unidet_features.tsv"), sep="\t")
            gsam_features = pd.read_csv(os.path.join(output_subfolder_path, "gsam_features.tsv"), sep="\t")
            unidet_features_batches = []
            gsam_features_batches = []

        cap_is_opened = True
        part_frame_index = 0

        frame_indices_batches = []
        frames_batches = []

        while cap_is_opened:
            frame_indices_batch, frames_batch = [], []
            if frame_feature_name == "ego_hos":
                unidet_features_batch, gsam_features_batch = [], []

            for i in range(batch_size * stride_for_processing_frames):
                if part_frame_index == number_of_frames_per_video_part:
                    cap_is_opened = False
                    break
                frame = cap.read()
                if frame is None:  # Reached the end of the video.
                    cap_is_opened = False
                    break
                else:  # We got a valid frame.
                    current_global_frame_index = global_frame_index.get_value()
                    if current_global_frame_index % stride_for_processing_frames != 0:  # The current frame will not be added to the current batch. But we update our counters.
                        part_frame_index += 1
                        global_frame_index.increment_value()
                        continue
                    else:  # The current frame will be added to the current batch.
                        frames_batch.append(frame)
                        frame_indices_batch.append(current_global_frame_index)
                        if frame_feature_name == "ego_hos":
                            relevant_unidet_feature_rows = [dict(row) for index, row in unidet_features[unidet_features["frame_index"] == global_frame_index.get_value()].iterrows()]
                            relevant_gsam_feature_rows = [dict(row) for index, row in gsam_features[gsam_features["frame_index"] == global_frame_index.get_value()].iterrows()]
                            unidet_features_batch.append(relevant_unidet_feature_rows)
                            gsam_features_batch.append(relevant_gsam_feature_rows)
                        global_frame_index.increment_value()
                        part_frame_index += 1

            if len(frames_batch) > 0:
                frame_indices_batches.append(frame_indices_batch)
                frames_batches.append(frames_batch)
                if frame_feature_name == "ego_hos":
                    unidet_features_batches.append(unidet_features_batch)
                    gsam_features_batches.append(gsam_features_batch)
        if frame_feature_name == "ego_hos":
            inputs = zip(frame_indices_batches, frames_batches, unidet_features_batches, gsam_features_batches)
        else:
            inputs = zip(frame_indices_batches, frames_batches)
        return inputs

    @abstractmethod
    def predictor_function(self, frame_indices_batch: List[int], frames_batch: List[np.array]):
        pass

    @staticmethod
    def save_results(input_video_file_path, results_list, output_folder_path, column_names, output_file_name):
        results_df = pd.DataFrame(data=[item for sublist in results_list for item in sublist], columns=column_names)

        input_video_file_name_wo_ext = input_video_file_path.split("/")[-1][:-4]

        os.makedirs(
            os.path.join(output_folder_path, input_video_file_name_wo_ext),
            exist_ok=True,
        )
        results_df.to_csv(
            os.path.join(output_folder_path, input_video_file_name_wo_ext, output_file_name),
            index=False,
            sep="\t",
        )
