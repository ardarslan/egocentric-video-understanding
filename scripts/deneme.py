import os
import cv2
import ray

ray.init(num_gpus=2, num_cpus=2)
import torch
import torch

torch.cuda.empty_cache()
import numpy as np
from extract_frame_features.frame_feature_extractor import (
    FrameFeatureExtractor,
)
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from typing import List
from PIL import Image
from extract_frame_features.constants import question_constant_mapping


@ray.remote(num_gpus=1)
class BLIP2VQAFrameFeatureExtractor(FrameFeatureExtractor):
    def __init__(self):
        super().__init__()
        self.processor = Blip2Processor.from_pretrained(
            os.path.join(os.environ["SCRATCH"], "mq_libs/blip2")
        )
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            os.path.join(os.environ["SCRATCH"], "mq_libs/blip2"),
            torch_dtype=torch.float16,
        )
        self.model.to("cuda")

    @property
    def column_names(self):
        return [
            "frame_index",
            "question",
            "blip2_answer",
        ]

    def predictor_function(
        self, frame_indices_batch: List[int], frames_batch: List[np.array]
    ):
        with torch.no_grad():
            all_results = []
            for question in question_constant_mapping.keys():
                if question == "asl":
                    continue
                preprocessed_frames_batch_dict = self.processor(
                    images=[
                        Image.fromarray(frame[:, :, ::-1]) for frame in frames_batch
                    ],
                    text=["Question: " + question + " Answer:"] * len(frames_batch),
                    return_tensors="pt",
                ).to("cuda", torch.float16)
                generated_ids = self.model.generate(**preprocessed_frames_batch_dict)
                blip2_answers = self.processor.batch_decode(
                    generated_ids, skip_special_tokens=True
                )
                blip2_answers = [blip2_answer.strip() for blip2_answer in blip2_answers]
                all_results.extend(
                    [
                        (
                            frame_index,
                            question,
                            blip2_answer,
                        )
                        for frame_index, blip2_answer in zip(
                            frame_indices_batch,
                            blip2_answers,
                        )
                    ]
                )
        return all_results


blip2vqa_frame_feature_extractor = BLIP2VQAFrameFeatureExtractor.remote()

frame_indices_batch = [10458]

input_video_file_path = os.path.join(
    os.environ["SCRATCH"],
    "ego4d_data/v2/clips",
    "02246bfe-dcef-465d-9aa5-47a2e71460dd.mp4",
)

cap = cv2.VideoCapture(input_video_file_path)
cap.set(cv2.CAP_PROP_POS_FRAMES, 10458 - 1)
_, frame = cap.read()
cap.release()
frames_batch = [frame]

results = blip2vqa_frame_feature_extractor.predictor_function.remote(
    frame_indices_batch=frame_indices_batch, frames_batch=frames_batch
)
Image.fromarray(frame[:, :, ::-1]).save("/home/aarslan/mq/image.png")
print(ray.get(results))

ray.shutdown()

