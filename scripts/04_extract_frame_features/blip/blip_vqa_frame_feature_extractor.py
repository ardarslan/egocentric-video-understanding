import gc

import ray
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

from blip.blip.blip_vqa import blip_vqa

from frame_feature_extractor import FrameFeatureExtractor

from typing import List


@ray.remote(num_gpus=1)
class BLIPVQAFrameFeatureExtractor(FrameFeatureExtractor):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.image_size = 384
        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (self.image_size, self.image_size),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )

        self.model = blip_vqa(
            pretrained=self.args.blip_vqa_model_file_path,
            image_size=self.image_size,
            vit="base",
        )
        self.model.eval()
        self.model = self.model.to(self.args.device)

        self.questions = [
            "What does the image describe?",
            "What is the person in this picture doing?",
            "What is happening in this picture?",
        ]

    @property
    def column_names(self):
        return ["frame_index", "question", "answer"]

    def predictor_function(
        self, frame_indices_batch: List[int], frames_batch: List[np.array]
    ):
        preprocessed_frames_batch = []
        for frame in frames_batch:
            frame = Image.fromarray(frame).convert("RGB")
            frame = self.transform(frame).unsqueeze(0).to(self.args.device)
            preprocessed_frames_batch.append(frame)
        preprocessed_frames_batch = torch.vstack(preprocessed_frames_batch)
        del frames_batch
        gc.collect()
        torch.cuda.empty_cache()

        predictions = []
        with torch.no_grad():
            for question in self.questions:
                answers = self.model(
                    preprocessed_frames_batch,
                    question,
                    train=False,
                    inference="generate",
                )
                for frame_index, answer in zip(frame_indices_batch, answers):
                    predictions.append((frame_index, question, answer))

        del preprocessed_frames_batch
        del frame_indices_batch
        gc.collect()
        torch.cuda.empty_cache()

        return predictions
