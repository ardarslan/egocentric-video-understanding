import os
import cv2
import json
import torch
import pickle
import random
import argparse
import numpy as np
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration

from typing import Any, Dict, List


class BLIP2VQAFineTuningDataset(object):
    def __init__(self, args: argparse.Namespace, split: str):
        self.clip_ids = []
        with open(args.annotations_file_path, "rb") as reader:
            annotations = json.load(reader)
            for clip_id in annotations.keys():
                if annotations[clip_id]["subset"] == split:
                    self.clip_ids.append(clip_id)
        self.current_clip_index = 0
        self.current_cap = cv2.VideoCapture(
            os.path.join(
                os.environ["SCRATCH"],
                "ego4d_data/v2/clips",
                self.clip_ids[self.current_clip_index] + ".mp4",
            )
        )
        self.current_frame_id = 0
        with open(
            os.path.join(
                os.environ["SCRATCH"],
                "ego4d_data/v2/analysis_data/ground_truth_labels/ground_truth_labels.pickle",
            ),
            "rb",
        ) as reader:
            self.ground_truth_label_indices = pickle.load(reader)

        with open(
            os.path.join(
                os.environ["CODE"],
                "scripts/06_analyze_frame_features/02_map_label_dependency_parsing_features_and_blip2_answer_dependency_parsing_features/label_phrase_mapping.json",
            ),
            "r",
        ) as reader:
            self.label_phrase_mapping = json.load(reader)

        self.distinct_ground_truth_labels = sorted(
            list(self.label_phrase_mapping.keys())
        ) + ["background"]
        self.batch_size = args.batch_size
        self.device = args.device
        self.prompt = args.prompt
        self.current_batch = []
        self.processor = Blip2Processor.from_pretrained(
            os.path.join(os.environ["SCRATCH"], "mq_libs/blip2")
        )

    def collate_fn(self, batch: List[Dict[str, Any]]):
        result = {}
        for sample in batch:
            for key, value in sample.items():
                if key not in result.keys():
                    result[key] = [value]
                else:
                    result[key].append(value)
        for key in result.keys():
            result[key] = torch.vstack(result[key]).to(self.device)
        return result

    def get_current_sample_encodings(
        self, current_frame: np.ndarray, current_label_phrase: str
    ):
        input_encoding = self.processor(
            images=[Image.fromarray(current_frame[:, :, ::-1])],
            text=[self.prompt],
            padding="max_length",
            return_tensors="pt",
        )
        output_encoding = self.processor.tokenizer(
            [current_label_phrase],
            padding=True,
            return_tensors="pt",
        )

        return {
            "input_ids": input_encoding["input_ids"],
            "pixel_values": input_encoding["pixel_values"],
            "labels": output_encoding["input_ids"],
        }

    def get_one_sample(self):
        random_clip_id_index = random.randint(len(self.clip_ids))
        random_clip_id = self.clip_ids[random_clip_id_index]
        current_cap =
        self.success, current_frame = self.current_cap.read()
        current_label_indices = self.ground_truth_label_indices[
            self.clip_ids[self.current_clip_index]
        ][self.current_frame_id]
        current_labels = [
            self.distinct_ground_truth_labels[current_label_index]
            for current_label_index in current_label_indices
        ]
        if current_labels[0] == "background":
            return self.get_current_sample_encodings(
                current_frame=current_frame, current_label_phrase=""
            )
        else:
            current_label_phrases = [
                self.label_phrase_mapping[current_label]
                for current_label in current_labels
            ]
            for current_label_phrase in current_label_phrases:
                return self.get_current_sample_encodings(
                    current_frame=current_frame,
                    current_label_phrase=current_label_phrase,
                )
        self.current_clip_index += 1
        if self.current_clip_index < len(self.clip_ids):
            self.current_cap = cv2.VideoCapture(
                os.path.join(
                    os.environ["SCRATCH"],
                    "ego4d_data/v2/clips",
                    self.clip_ids[self.current_clip_index] + ".mp4",
                )
            )
            self.current_frame_id = 0
            self.success, current_frame = self.current_cap.read()
            current_label_indices = self.ground_truth_label_indices[
                self.clip_ids[self.current_clip_index]
            ][self.current_frame_id]
            current_labels = [
                self.distinct_ground_truth_labels[current_label_index]
                for current_label_index in current_label_indices
            ]
            current_label_phrases = [
                self.label_phrase_mapping[current_label]
                for current_label in current_labels
            ]
            for current_label_phrase in current_label_phrases:
                return self.get_current_sample_encodings(
                    current_frame=current_frame,
                    current_label_phrase=current_label_phrase,
                )
        else:
            return None

    def __iter__(self):
        while True:
            current_sample = self.get_one_sample()
            if current_sample is None:
                if len(self.current_batch) > 0:
                    yield self.collate_fn(self.current_batch)
                else:
                    return
            else:
                self.current_batch.append(current_sample)
                if len(self.current_batch) == self.batch_size:
                    yield self.collate_fn(self.current_batch)
                    self.current_batch = []
                else:
                    continue


parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="cuda:4")
parser.add_argument(
    "--prompt", type=str, default="What is the person in this image doing?"
)
parser.add_argument(
    "--annotations_file_path",
    type=str,
    default=os.path.join(
        os.environ["CODE"],
        "scripts/07_reproduce_baseline_results/data/ego4d/ego4d_clip_annotations_v3.json",
    ),
)
parser.add_argument("--num_epochs", type=int, default=50)
parser.add_argument("--batch_size", type=int, default=4)
args = parser.parse_args()

model = Blip2ForConditionalGeneration.from_pretrained(
    os.path.join(os.environ["SCRATCH"], "mq_libs/blip2"),
    torch_dtype=torch.float16,
)
model.to(args.device)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

for epoch in range(args.num_epochs):
    model.train()
    iftd_train = BLIP2VQAFineTuningDataset(args=args, split="train")
    total_train_loss = 0.0
    total_train_sample_count = 0.0
    for train_batch in iftd_train:
        optimizer.zero_grad()
        outputs = model(
            input_ids=train_batch["input_ids"],
            pixel_values=train_batch["pixel_values"],
            labels=train_batch["labels"],
        )
        train_loss = outputs.loss
        total_train_loss += train_loss.item() * len(train_batch["input_ids"])
        total_train_sample_count += float(len(train_batch["input_ids"]))
        train_loss.backward()
        optimizer.step()

    model.eval()
    iftd_val = BLIP2VQAFineTuningDataset(args=args, split="val")
    total_val_loss = 0.0
    total_val_sample_count = 0.0
    with torch.no_grad():
        for val_batch in iftd_val:
            outputs = model(
                input_ids=val_batch["input_ids"],
                pixel_values=val_batch["pixel_values"],
                labels=val_batch["labels"],
            )
            val_loss = outputs.loss
            total_val_loss += val_loss.item() * len(val_batch["input_ids"])
            total_val_sample_count += float(len(val_batch["input_ids"]))

    print(
        f"Epoch: {str(epoch).zfill(2)} | Train Loss: {np.round(total_train_loss / total_train_sample_count, 2)} | Val Loss: {np.round(total_val_loss / total_val_sample_count, 2)}"
    )
