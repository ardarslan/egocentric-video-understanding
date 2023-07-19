import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from transformers import OFATokenizer, OFAModel
from transformers.models.ofa.generate import sequence_generator

from typing import List, Tuple, Any


class OFAFrameFeatureExtractor(object):
    def __init__(self, args):
        self.batch_size = 4
        mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        resolution = 256
        self.patch_resize_transform = transforms.Compose(
            [
                transforms.Resize((resolution, resolution), interpolation=Image.BICUBIC),
                transforms.ToTensor(),
                lambda tensor: tensor.to(args.device),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
        self.tokenizer = OFATokenizer.from_pretrained(args.ofa_checkpoint_dir)
        self.model = OFAModel.from_pretrained(args.ofa_checkpoint_dir, use_cache=False).to(args.device)
        self.questions = [
            "What does the image describe?",
            "What is the person in this picture doing?",
            "What is happening in this picture?",
        ]
        self.question_input_ids_mapping = dict((question, self.tokenizer([question], return_tensors="pt").input_ids.repeat(self.batch_size, 1).to(args.device)) for question in self.questions)
        self.patch_masks = torch.tensor([True], device=args.device).repeat(self.batch_size, 1).to(args.device)
        self.generator = sequence_generator.SequenceGenerator(
            tokenizer=self.tokenizer,
            beam_size=5,
            max_len_b=16,
            min_len=0,
            no_repeat_ngram_size=3,
        ).to(args.device)
        self.column_names = ["frame_index", "question", "answer"]
        self.file_name_wo_ext = "ofa_features"

    def extract_frame_features(self, frame_indices: List[int], frames: List[np.array]) -> List[Tuple[Any, ...]]:
        """
        frames[0]: in BGR format. The model expects RGB, so we do frame[:, :, ::-1].
        """
        patch_images = torch.vstack([self.patch_resize_transform(Image.fromarray(frame[:, :, ::-1])).unsqueeze(0) for frame in frames])
        features = []
        for question in self.questions:
            input_ids = self.question_input_ids_mapping[question]

            data = {
                "net_input": {
                    "input_ids": input_ids,
                    "patch_images": patch_images,
                    "patch_masks": self.patch_masks,
                }
            }
            gen_output = self.generator.generate([self.model], data)
            gen = [gen_output[i][0]["tokens"] for i in range(len(gen_output))]
            gen = self.model.generate(input_ids, patch_images=patch_images, num_beams=5, no_repeat_ngram_size=3)
            answers = self.tokenizer.batch_decode(gen, skip_special_tokens=True)
            features.extend([(frame_indices[i], question, answers[i]) for i in range(len(frame_indices))])
        return features
