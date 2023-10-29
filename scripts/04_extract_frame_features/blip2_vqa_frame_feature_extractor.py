import os
import sys

# import ray   # CHANGEHERE
import torch
import numpy as np
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration

from frame_feature_extractor import FrameFeatureExtractor

from typing import List

sys.path.insert(
    0,
    os.path.join(
        os.environ["CODE"],
        "scripts/06_analyze_frame_features/03_map_label_dependency_parsing_features_and_blip2_answer_dependency_parsing_features/",
    ),
)

from constants import question_constant_mapping


# @ray.remote(num_gpus=1)   # CHANGEHERE
class BLIP2VQAFrameFeatureExtractor(FrameFeatureExtractor):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.processor = Blip2Processor.from_pretrained(
            os.path.join(os.environ["SCRATCH"], "mq_libs/blip2")
        )
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            os.path.join(os.environ["SCRATCH"], "mq_libs/blip2"),
            torch_dtype=torch.float16,
        )
        self.model.to(args.device)

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
                ).to(self.args.device, torch.float16)
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
