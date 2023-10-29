import os
import gc
import ray
import numpy as np
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch

from frame_feature_extractor import FrameFeatureExtractor

from typing import List


@ray.remote(num_gpus=1)
class BLIP2CaptioningFrameFeatureExtractor(FrameFeatureExtractor):
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
        return ["frame_index", "question", "answer"]

    def predictor_function(
        self, frame_indices_batch: List[int], frames_batch: List[np.array]
    ):
        with torch.no_grad():
            preprocessed_frames_batch_dict = self.processor(
                images=[Image.fromarray(frame[:, :, ::-1]) for frame in frames_batch],
                return_tensors="pt",
            ).to(self.args.device, torch.float16)
            generated_ids = self.model.generate(**preprocessed_frames_batch_dict)
            generated_texts = self.processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )
            generated_texts = [
                (frame_index, "Image Caption", generated_text.strip())
                for frame_index, generated_text in zip(
                    frame_indices_batch, generated_texts
                )
            ]
            del preprocessed_frames_batch_dict
            del frame_indices_batch
            gc.collect()
            torch.cuda.empty_cache()
            return generated_texts
