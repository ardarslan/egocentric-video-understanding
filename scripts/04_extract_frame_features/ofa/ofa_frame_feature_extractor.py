import gc

import ray
import torch
import numpy as np
from PIL import Image
from transformers import OFATokenizer, OFAModel
from torchvision import transforms
from transformers.models.ofa.generate import sequence_generator

from frame_feature_extractor import FrameFeatureExtractor

from typing import List


@ray.remote(num_gpus=1)
class OFAFrameFeatureExtractor(FrameFeatureExtractor):
    def __init__(self, args):
        super().__init__()
        self.args = args
        mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        resolution = 256
        self.patch_resize_transform = transforms.Compose(
            [
                transforms.Resize(
                    (resolution, resolution), interpolation=Image.BICUBIC
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
        self.tokenizer = OFATokenizer.from_pretrained(args.ofa_model_file_path)
        self.model = OFAModel.from_pretrained(
            args.ofa_model_file_path, use_cache=False
        ).to(args.device)
        self.model.eval()
        self.questions = [
            "What does the image describe?",
            "What is the person in this picture doing?",
            "What is happening in this picture?",
        ]
        self.question_input_ids_mapping = dict(
            (
                question,
                self.tokenizer([question], return_tensors="pt")
                .input_ids.repeat(self.args.batch_size, 1)
                .to(self.args.device),
            )
            for question in self.questions
        )
        self.patch_masks = torch.tensor([True], device=args.device).repeat(
            self.args.batch_size, 1
        )
        self.generator = sequence_generator.SequenceGenerator(
            tokenizer=self.tokenizer,
            beam_size=5,
            max_len_b=16,
            min_len=0,
            no_repeat_ngram_size=3,
        ).to(self.args.device)

    @property
    def column_names(self):
        return ["frame_index", "question", "answer"]

    def predictor_function(
        self, frame_indices_batch: List[int], frames_batch: List[np.array]
    ):
        patch_images_batch = []
        for frame in frames_batch:
            patch_images = (
                self.patch_resize_transform(Image.fromarray(frame[:, :, ::-1]))
                .unsqueeze(0)
                .to(self.args.device)
            )
            patch_images_batch.append(patch_images)
        del frames_batch
        gc.collect()
        torch.cuda.empty_cache()
        patch_images_batch = torch.vstack(patch_images_batch)

        predictions = []
        for question in self.questions:
            input_ids = self.question_input_ids_mapping[question]
            data = {
                "net_input": {
                    "input_ids": input_ids,
                    "patch_images": patch_images_batch,
                    "patch_masks": self.patch_masks,
                }
            }
            data = self.generator.generate([self.model], data)
            data = [data[i][0]["tokens"] for i in range(len(data))]
            data = self.model.generate(
                input_ids,
                patch_images=patch_images_batch,
                num_beams=5,
                no_repeat_ngram_size=3,
            )
            data = self.tokenizer.batch_decode(data, skip_special_tokens=True)
            for frame_index, answer in zip(frame_indices_batch, data):
                predictions.append((frame_index, question, answer))
        del patch_images_batch
        del frame_indices_batch
        gc.collect()
        torch.cuda.empty_cache()
        return predictions
