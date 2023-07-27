import gc

import ray
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

from blip.blip.blip import blip_decoder

from frame_feature_extractor import FrameFeatureExtractor

from typing import List


@ray.remote(num_gpus=1)
class BLIPCaptioningFrameFeatureExtractor(FrameFeatureExtractor):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.image_size = 384
        self.transform = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size), interpolation=InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ]
        )

        self.model = blip_decoder(pretrained=self.args.blip_captioning_model_file_path, image_size=self.image_size, vit="base")
        self.model.eval()
        self.model = self.model.to(self.args.device)

    @property
    def output_file_name(self):
        return "blip_captioning_features.tsv"

    @property
    def error_file_name(self):
        return "blip_captioning_errors.txt"

    @property
    def column_names(self):
        return ["frame_index", "question", "answer"]

    def predictor_function(self, frame_indices_batch: List[int], frames_batch: List[np.array]):
        preprocessed_frames_batch = []
        for frame in frames_batch:
            frame = Image.fromarray(frame).convert("RGB")
            frame = self.transform(frame).unsqueeze(0)
            preprocessed_frames_batch.append(frame)
        preprocessed_frames_batch = torch.vstack(preprocessed_frames_batch).to(self.args.device)
        del frames_batch
        gc.collect()
        torch.cuda.empty_cache()

        predictions = []
        with torch.no_grad():
            image_captions = self.model.generate(preprocessed_frames_batch, sample=False, num_beams=3, max_length=20, min_length=5)
            for frame_index, image_caption in zip(frame_indices_batch, image_captions):
                predictions.append((frame_index, "Image Caption", image_caption))
        del preprocessed_frames_batch

        return predictions
