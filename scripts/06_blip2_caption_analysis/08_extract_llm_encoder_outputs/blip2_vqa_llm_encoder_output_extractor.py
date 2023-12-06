import os
import sys
import cv2
import torch
import argparse
import numpy as np
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from accelerate import init_empty_weights, infer_auto_device_map

from llm_encoder_output_extractor import LLMEncoderOutputExtractor

from typing import List

sys.path.insert(
    0,
    os.path.join(
        os.environ["CODE"],
        "scripts/06_analyze_frame_features/08_extract_llm_encoder_outputs/",
    ),
)


# @ray.remote(num_gpus=1)
class BLIP2VQALLMEncoderOutputExtractor(LLMEncoderOutputExtractor):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.processor = Blip2Processor.from_pretrained(
            os.path.join(os.environ["SCRATCH"], "mq_libs/blip2")
        )

        self.model = Blip2ForConditionalGeneration.from_pretrained(
            os.path.join(os.environ["SCRATCH"], "mq_libs/blip2"),
            torch_dtype=torch.float32,
        )

    @property
    def column_names(self):
        return [
            "frame_index",
            "question",
            "blip2_llm_encoder_output",
        ]

    def predictor_function(
        self, frame_indices_batch: List[int], frames_batch: List[np.array]
    ):
        question = "What is the person in this image doing?"
        with torch.no_grad():
            preprocessed_frames_batch_dict = self.processor(
                images=[Image.fromarray(frame[:, :, ::-1]) for frame in frames_batch],
                text=["Question: " + question + " Answer:"] * len(frames_batch),
                return_tensors="pt",
            ).to(self.args.device)
            blip2_vqa_llm_encoder_output = self.model.generate(
                **preprocessed_frames_batch_dict
            )
        return zip(frame_indices_batch, blip2_vqa_llm_encoder_output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Argument parser")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    blip2_vqa_llm_encoder_output_extractor = BLIP2VQALLMEncoderOutputExtractor(
        args=args
    )
    cap = cv2.VideoCapture(
        os.path.join(
            os.environ["SCRATCH"],
            "ego4d_data/v2/clips",
            "ffe2261f-b973-4fbd-8824-06f8334afdc5.mp4",
        )
    )
    _, first_frame = cap.read()
    _, second_frame = cap.read()
    cap.release()
    frames_batch = [first_frame, second_frame]
    frame_indices_batch = [0, 1]
    blip2_vqa_llm_encoder_output = (
        blip2_vqa_llm_encoder_output_extractor.predictor_function(
            frame_indices_batch=frame_indices_batch, frames_batch=frames_batch
        )
    )
