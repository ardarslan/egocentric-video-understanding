import os
import json
import torch
import random
import argparse
import numpy as np
from PIL import Image
from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from transformers import Blip2Processor, Blip2ForConditionalGeneration

from typing import List


@pipeline_def
def video_pipe(file_list):
    video, label = fn.readers.video(
        device="cpu",
        file_list=file_list,
        sequence_length=1,
        shard_id=0,
        num_shards=1,
        random_shuffle=True,
        initial_fill=1024,
        image_type=types.RGB,
        dtype=types.FLOAT,
        file_list_frame_num=True,
        enable_frame_num=False,
        enable_timestamps=False,
        file_list_include_preceding_frame=True,
    )
    return video, label


class BLIP2VQAFineTuningDataset(object):
    def __init__(self, args: argparse.Namespace, split: str):
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

        self.device = args.device
        self.batch_size = args.batch_size
        self.prompt = args.prompt
        self.processor = Blip2Processor.from_pretrained(
            os.path.join(os.environ["SCRATCH"], "mq_libs/blip2")
        )
        self.split = split
        self.file_list = os.path.join(
            os.environ["SCRATCH"],
            "ego4d_data/v2/analysis_data/ground_truth_labels",
            f"{self.split}_ground_truth_labels.txt",
        )
        self.pipe = video_pipe(
            batch_size=args.batch_size,
            num_threads=args.num_data_reader_threads,
            device_id=0,
            file_list=self.file_list,
        )
        self.pipe.build()

    def get_random_label_phrase(self, label_phrases: List[str]):
        random_index = random.randint(len(label_phrases))
        random_label_phrase = label_phrases[random_index]
        return random_label_phrase

    def get_batch(self):
        images, label_indices = self.pipe.run()
        input_encoding = self.processor(
            images=[Image.fromarray(np.array(image)) for image in images],
            text=[self.prompt] * self.batch_size,
            padding="max_length",
            return_tensors="pt",
        )
        labels = [
            self.distinct_ground_truth_labels[label_index[0]]
            for label_index in label_indices
        ]
        label_phrases = [
            self.get_random_label_phrase(self.label_phrase_mapping[label])
            for label in labels
        ]
        output_encoding = self.processor.tokenizer(
            label_phrases,
            padding=True,
            return_tensors="pt",
        )
        return {
            "input_ids": input_encoding["input_ids"].to(self.device),
            "pixel_values": input_encoding["pixel_values"].to(self.device),
            "labels": output_encoding["input_ids"].to(self.device),
        }


parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="cuda:7")
parser.add_argument("--num_data_reader_threads", type=int, default=8)
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
parser.add_argument("--num_batches_in_one_epoch", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=4)
args = parser.parse_args()

model = Blip2ForConditionalGeneration.from_pretrained(
    os.path.join(os.environ["SCRATCH"], "mq_libs/blip2"),
    torch_dtype=torch.float32,
)
model.to(args.device)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

iftd_train = BLIP2VQAFineTuningDataset(args=args, split="train")
iftd_val = BLIP2VQAFineTuningDataset(args=args, split="val")

for epoch in range(args.num_epochs):
    total_train_loss = 0.0
    total_train_sample_count = 0.0
    model.train()
    for _ in range(args.num_batches_in_one_epoch):
        optimizer.zero_grad()
        train_batch = iftd_train.get_batch()
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

    total_val_loss = 0.0
    total_val_sample_count = 0.0
    model.eval()
    with torch.no_grad():
        for _ in range(args.num_batches_in_one_epoch):
            val_batch = iftd_val.get_batch()
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
