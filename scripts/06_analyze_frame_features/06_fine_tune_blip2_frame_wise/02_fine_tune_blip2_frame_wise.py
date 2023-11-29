import os
import gc
import cv2
import json
import torch
import pickle
import argparse
import numpy as np
from PIL import Image
from pathlib import Path

from accelerate import init_empty_weights, infer_auto_device_map
from transformers import (
    Blip2Processor,
    Blip2ForConditionalGeneration,
    Blip2PreTrainedModel,
)

from typing import List


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

        self.batch_size = args.batch_size
        self.prompt = args.prompt
        self.processor = Blip2Processor.from_pretrained(
            os.path.join(os.environ["SCRATCH"], "mq_libs/blip2")
        )
        with open(
            os.path.join(
                os.environ["SCRATCH"],
                "ego4d_data/v2/analysis_data/ground_truth_labels/ground_truth_labels.pickle",
            ),
            "rb",
        ) as reader:
            self.ground_truth_labels = pickle.load(reader)

        self.clip_ids = []

        with open(args.annotations_file_path, "r") as reader:
            annotations = json.load(reader)
            for clip_id in annotations.keys():
                if annotations[clip_id]["subset"] == split:
                    self.clip_ids.append(clip_id)

        self.split = split

    def get_random_label_phrase(self, label_phrases: List[str]):
        if label_phrases is None:
            return ""
        else:
            random_index = np.random.randint(len(label_phrases))
            random_label_phrase = label_phrases[random_index]
            return random_label_phrase

    def get_random_file_name(self):
        random_idx = np.random.randint(len(self.clip_ids))
        random_ground_truth_label_key = self.clip_ids[random_idx]
        return random_ground_truth_label_key + ".mp4"

    def get_random_label_indices_and_images(self):
        random_label_indices = []
        random_images = []
        for _ in range(self.batch_size):
            random_file_name = self.get_random_file_name()
            random_clip_id = random_file_name.split(".")[0]
            if not os.path.exists(
                os.path.join(
                    os.environ["SCRATCH"], "ego4d_data/v2/clips", random_file_name
                )
            ):
                raise Exception("This clip does not exist.")

            random_cap = cv2.VideoCapture(
                os.path.join(
                    os.environ["SCRATCH"], "ego4d_data/v2/clips", random_file_name
                )
            )
            number_of_frames = random_cap.get(cv2.CAP_PROP_FRAME_COUNT)
            random_frame_index = np.random.randint(number_of_frames)
            random_cap.set(cv2.CAP_PROP_POS_FRAMES, random_frame_index - 1)
            _, random_frame = random_cap.read()
            random_cap.release()
            current_random_label_indices = self.ground_truth_labels[random_clip_id][
                random_frame_index
            ]
            current_random_label_indices_index = np.random.randint(
                len(current_random_label_indices)
            )
            current_random_label_index = current_random_label_indices[
                current_random_label_indices_index
            ]
            random_label_indices.append(current_random_label_index)
            random_images.append(Image.fromarray(random_frame[:, :, ::-1]))
        return random_label_indices, random_images

    def get_batch(self):
        random_label_indices, random_images = self.get_random_label_indices_and_images()
        input_encoding = self.processor(
            images=random_images,
            text=[self.prompt] * self.batch_size,
            padding="max_length",
            return_tensors="pt",
        )
        labels = [
            self.distinct_ground_truth_labels[int(label_index)]
            for label_index in random_label_indices
        ]
        label_phrases = [
            self.get_random_label_phrase(self.label_phrase_mapping.get(label, None))
            for label in labels
        ]
        output_encoding = self.processor.tokenizer(
            label_phrases,
            padding=True,
            return_tensors="pt",
        )
        return {
            "input_ids": input_encoding["input_ids"].to("cuda"),
            "pixel_values": input_encoding["pixel_values"].to("cuda"),
            "labels": output_encoding["input_ids"].to("cuda"),
        }


parser = argparse.ArgumentParser()
# parser.add_argument("--device", type=str, default="cuda")
parser.add_argument(
    "--prompt", type=str, default="What is the person in this image doing?"
)
parser.add_argument(
    "--annotations_file_path",
    type=str,
    default=os.path.join(
        os.environ["CODE"],
        "scripts/07_reproduce_mq_experiments/data/ego4d/ego4d_clip_annotations_v3.json",
    ),
)
parser.add_argument(
    "--best_model_file_path",
    type=str,
    default=os.path.join(
        os.environ["SCRATCH"],
        "ego4d_data/v2/analysis_data/best_fine_tuned_blip2_model/best_fine_tuned_blip2_model.pt",
    ),
)
parser.add_argument("--num_epochs", type=int, default=50)
parser.add_argument("--num_batches_in_one_epoch", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--model_dtype", default=torch.float32)
args = parser.parse_args()

if not os.path.exists(args.best_model_file_path):
    os.makedirs(str(Path(args.best_model_file_path).parent), exist_ok=True)

with init_empty_weights():
    model = Blip2ForConditionalGeneration.from_pretrained(
        os.path.join(os.environ["SCRATCH"], "mq_libs/blip2"),
        torch_dtype=torch.float32,
    )

device_map = infer_auto_device_map(
    model,
    no_split_module_classes=Blip2PreTrainedModel._no_split_modules,
    dtype=args.model_dtype,
)

del model
gc.collect()

model = Blip2ForConditionalGeneration.from_pretrained(
    os.path.join(os.environ["SCRATCH"], "mq_libs/blip2"),
    device_map=device_map,
    torch_dtype=args.model_dtype,
)
model.train()

for name, layer in model.named_modules():
    print(name, layer.requires_grad)

# optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# iftd_train = BLIP2VQAFineTuningDataset(args=args, split="train")
# iftd_val = BLIP2VQAFineTuningDataset(args=args, split="val")

# best_val_loss = np.inf

# for epoch in range(1, args.num_epochs + 1):
#     print(f"Running epoch {epoch}...")
#     total_train_loss = 0.0
#     total_train_sample_count = 0.0
#     model.train()
#     for _ in tqdm(range(args.num_batches_in_one_epoch)):
#         optimizer.zero_grad()
#         train_batch = iftd_train.get_batch()
#         outputs = model(
#             input_ids=train_batch["input_ids"],
#             pixel_values=train_batch["pixel_values"],
#             labels=train_batch["labels"],
#         )
#         train_loss = outputs.loss
#         total_train_loss += train_loss.item() * len(train_batch["input_ids"])
#         total_train_sample_count += float(len(train_batch["input_ids"]))
#         train_loss.backward()
#         optimizer.step()

#     total_val_loss = 0.0
#     total_val_sample_count = 0.0
#     model.eval()
#     with torch.no_grad():
#         for _ in range(args.num_batches_in_one_epoch):
#             val_batch = iftd_val.get_batch()
#             outputs = model(
#                 input_ids=val_batch["input_ids"],
#                 pixel_values=val_batch["pixel_values"],
#                 labels=val_batch["labels"],
#             )
#             val_loss = outputs.loss
#             if val_loss < best_val_loss:
#                 best_val_loss = val_loss
#                 torch.save(model.state_dict(), args.best_model_file_path)

#             total_val_loss += val_loss.item() * len(val_batch["input_ids"])
#             total_val_sample_count += float(len(val_batch["input_ids"]))

#     print(
#         f"Epoch: {str(epoch).zfill(2)} | Train Loss: {np.round(total_train_loss / total_train_sample_count, 2)} | Val Loss: {np.round(total_val_loss / total_val_sample_count, 2)}"
#     )

# del model
# gc.collect()

# model = Blip2ForConditionalGeneration.from_pretrained(
#     os.path.join(os.environ["SCRATCH"], "mq_libs/blip2"),
#     device_map=device_map,
#     torch_dtype=args.model_dtype,
# )

# optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# iftd_train = BLIP2VQAFineTuningDataset(args=args, split="train")
# iftd_val = BLIP2VQAFineTuningDataset(args=args, split="val")

# best_val_loss = np.inf

# for epoch in range(1, args.num_epochs + 1):
#     print(f"Running epoch {epoch}...")
#     total_train_loss = 0.0
#     total_train_sample_count = 0.0
#     model.train()
#     for _ in tqdm(range(args.num_batches_in_one_epoch)):
#         optimizer.zero_grad()
#         train_batch = iftd_train.get_batch()
#         outputs = model(
#             input_ids=train_batch["input_ids"],
#             pixel_values=train_batch["pixel_values"],
#             labels=train_batch["labels"],
#         )
#         train_loss = outputs.loss
#         total_train_loss += train_loss.item() * len(train_batch["input_ids"])
#         total_train_sample_count += float(len(train_batch["input_ids"]))
#         train_loss.backward()
#         optimizer.step()

#     total_val_loss = 0.0
#     total_val_sample_count = 0.0
#     model.eval()
#     with torch.no_grad():
#         for _ in range(args.num_batches_in_one_epoch):
#             val_batch = iftd_val.get_batch()
#             outputs = model(
#                 input_ids=val_batch["input_ids"],
#                 pixel_values=val_batch["pixel_values"],
#                 labels=val_batch["labels"],
#             )
#             val_loss = outputs.loss
#             if val_loss < best_val_loss:
#                 best_val_loss = val_loss
#                 torch.save(model.state_dict(), args.best_model_file_path)

#             total_val_loss += val_loss.item() * len(val_batch["input_ids"])
#             total_val_sample_count += float(len(val_batch["input_ids"]))

#     print(
#         f"Epoch: {str(epoch).zfill(2)} | Train Loss: {np.round(total_train_loss / total_train_sample_count, 2)} | Val Loss: {np.round(total_val_loss / total_val_sample_count, 2)}"
#     )
