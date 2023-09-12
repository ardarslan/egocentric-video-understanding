import os
import cv2
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from typing import List


class FrameFeatureExtractor(ABC):
    def __init__(self):
        pass

    @staticmethod
    def get_inputs(
        cap,
        batch_size: int,
        frame_feature_name: str,
        output_subfolder_path: str,
        frame_feature_extraction_stride: int,
        global_frame_index,
    ):
        if frame_feature_name == "ego_hos":
            if not os.path.exists(
                os.path.join(output_subfolder_path, "unidet_features.tsv")
            ):
                raise Exception(
                    "To extract ego_hos features, unidet features should be extracted first."
                )
            if not os.path.exists(
                os.path.join(output_subfolder_path, "gsam_features.tsv")
            ):
                raise Exception(
                    "To extract ego_hos features, gsam features should be extracted first."
                )
            unidet_features = pd.read_csv(
                os.path.join(output_subfolder_path, "unidet_features.tsv"), sep="\t"
            )
            gsam_features = pd.read_csv(
                os.path.join(output_subfolder_path, "gsam_features.tsv"), sep="\t"
            )
            unidet_features_batches = []
            gsam_features_batches = []

        success = True

        frame_indices_batches = []
        frames_batches = []

        current_frame_pos = 0

        while success:
            frame_indices_batch, frames_batch = [], []
            if frame_feature_name == "ego_hos":
                unidet_features_batch, gsam_features_batch = [], []

            for i in range(batch_size):
                if not success:
                    break
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_pos - 1)
                success, frame = cap.read()
                current_frame_pos += frame_feature_extraction_stride

                if (
                    frame is not None
                ):  # We got a valid frame. The current frame will be added to the current batch.
                    frames_batch.append(frame)
                    current_global_frame_index = global_frame_index.get_value()
                    frame_indices_batch.append(current_global_frame_index)
                    global_frame_index.increment_value(frame_feature_extraction_stride)

                    if frame_feature_name == "ego_hos":
                        relevant_unidet_feature_rows = [
                            dict(row)
                            for index, row in unidet_features[
                                unidet_features["frame_index"]
                                == current_global_frame_index
                            ].iterrows()
                        ]
                        relevant_gsam_feature_rows = [
                            dict(row)
                            for index, row in gsam_features[
                                gsam_features["frame_index"]
                                == current_global_frame_index
                            ].iterrows()
                        ]
                        unidet_features_batch.append(relevant_unidet_feature_rows)
                        gsam_features_batch.append(relevant_gsam_feature_rows)

            if len(frames_batch) > 0:
                frame_indices_batches.append(frame_indices_batch)
                frames_batches.append(frames_batch)
                if frame_feature_name == "ego_hos":
                    unidet_features_batches.append(unidet_features_batch)
                    gsam_features_batches.append(gsam_features_batch)

        if frame_feature_name == "ego_hos":
            inputs = list(
                zip(
                    frame_indices_batches,
                    frames_batches,
                    unidet_features_batches,
                    gsam_features_batches,
                )
            )
        else:
            inputs = list(zip(frame_indices_batches, frames_batches))
        return inputs

    @abstractmethod
    def predictor_function(
        self, frame_indices_batch: List[int], frames_batch: List[np.array]
    ):
        pass

    @staticmethod
    def save_results(
        input_video_file_path,
        results_list,
        output_folder_path,
        column_names,
        output_file_name,
    ):
        results_df = pd.DataFrame(
            data=[item for sublist in results_list for item in sublist],
            columns=column_names,
        )

        input_video_file_name_wo_ext = input_video_file_path.split("/")[-1][:-4]

        os.makedirs(
            os.path.join(output_folder_path, input_video_file_name_wo_ext),
            exist_ok=True,
        )
        results_df.to_csv(
            os.path.join(
                output_folder_path, input_video_file_name_wo_ext, output_file_name
            ),
            index=False,
            sep="\t",
        )
