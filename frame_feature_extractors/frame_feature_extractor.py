import os
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from imutils.video import FileVideoStream

from typing import List


class FrameFeatureExtractor(ABC):
    def __init__(self):
        pass

    @staticmethod
    def get_frame_indices_batches_and_frames_batches(input_video_file_path: str, batch_size: int):
        cap = FileVideoStream(input_video_file_path).start()
        cap_is_opened = True
        frame_index = 0

        frame_indices_batches = []
        frames_batches = []
        while cap_is_opened:
            frame_indices_batch, frames_batch = [], []
            for i in range(batch_size):
                frame = cap.read()
                if frame is not None:
                    frames_batch.append(frame)
                    frame_indices_batch.append(frame_index)
                    frame_index += 1
                else:
                    cap_is_opened = False
                    break
            if len(frames_batch) > 0:
                frame_indices_batches.append(frame_indices_batch)
                frames_batches.append(frames_batch)
        cap.stop()
        return frame_indices_batches, frames_batches

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
