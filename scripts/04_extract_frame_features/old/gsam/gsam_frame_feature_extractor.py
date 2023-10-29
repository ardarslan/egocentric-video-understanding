import gc

import torch
import numpy as np
from PIL import Image

import ray
import torchvision
import torchvision.transforms as TS

import gsam.gsam.GroundingDINO.groundingdino.datasets.transforms as T
from gsam.gsam.GroundingDINO.groundingdino.models import build_model
from gsam.gsam.GroundingDINO.groundingdino.util.slconfig import SLConfig
from gsam.gsam.GroundingDINO.groundingdino.util.utils import (
    clean_state_dict,
    get_phrases_from_posmap,
)
from gsam.gsam.Tag2Text.models import tag2text
from gsam.gsam.Tag2Text import inference_ram

from frame_feature_extractor import FrameFeatureExtractor

from typing import List


@ray.remote(num_gpus=1, num_cpus=1)
class GSAMFrameFeatureExtractor(FrameFeatureExtractor):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.grounding_model = self.load_model(
            self.args.gsam_grounding_config_file_path,
            self.args.gsam_grounding_model_file_path,
            device=self.args.device,
        )
        self.pil_transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        self.ram_model = tag2text.ram(
            pretrained=self.args.gsam_ram_model_file_path, image_size=384, vit="swin_l"
        )
        self.ram_model.eval()
        self.ram_model = self.ram_model.to(self.args.device)

        self.ram_normalize = TS.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        self.ram_transform = TS.Compose(
            [TS.Resize((384, 384)), TS.ToTensor(), self.ram_normalize]
        )

    @property
    def column_names(self):
        return [
            "frame_index",
            "detection_index",
            "x_top_left",
            "y_top_left",
            "x_bottom_right",
            "y_bottom_right",
            "text_label",
            "detection_score",
        ]

    def load_model(self, model_config_path, model_checkpoint_path, device):
        args = SLConfig.fromfile(model_config_path)
        args.device = device
        model = build_model(args)
        checkpoint = torch.load(model_checkpoint_path)
        model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        model.eval()
        model.to(device)
        return model

    def get_grounding_output(
        self, model, image, caption, box_threshold, text_threshold, device="cpu"
    ):
        caption = caption.lower()
        caption = caption.strip()
        if not caption.endswith("."):
            caption = caption + "."
        with torch.no_grad():
            outputs = model(image[None], captions=[caption])
        logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
        boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
        logits.shape[0]

        # filter output
        logits_filt = logits.clone()
        boxes_filt = boxes.clone()
        filt_mask = logits_filt.max(dim=1)[0] > box_threshold
        logits_filt = logits_filt[filt_mask]  # num_filt, 256
        boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
        logits_filt.shape[0]

        # get phrase
        tokenizer = model.tokenizer
        tokenized = tokenizer(caption)
        # build pred
        text_labels = []
        scores = []
        for logit, box in zip(logits_filt, boxes_filt):
            text_label = get_phrases_from_posmap(
                logit > text_threshold, tokenized, tokenizer
            )
            text_labels.append(text_label)
            scores.append(float(logit.max().item()))
        return boxes_filt, scores, text_labels

    def predictor_function(
        self, frame_indices_batch: List[int], frames_batch: List[np.array]
    ):
        predictions = []
        for current_frame_index, frame_bgr_hwc_np in zip(
            frame_indices_batch, frames_batch
        ):
            frame_rgb_hwc_np = frame_bgr_hwc_np[:, :, ::-1]
            frame_rgb_hwc_pil = Image.fromarray(frame_rgb_hwc_np)
            frame_rgb_chw_torch, _ = self.pil_transform(frame_rgb_hwc_pil, None)
            frame_rgb_chw_torch = frame_rgb_chw_torch.to(self.args.device)
            frame_rgb_hwc_pil_resized = frame_rgb_hwc_pil.resize((384, 384))
            frame_rgb_hwc_pil_resized = (
                self.ram_transform(frame_rgb_hwc_pil_resized)
                .unsqueeze(0)
                .to(self.args.device)
            )
            ram_predictions = inference_ram.inference(
                frame_rgb_hwc_pil_resized, self.ram_model
            )
            ram_predictions = ram_predictions[0].replace(" |", ",")
            box_coordinates, detection_scores, text_labels = self.get_grounding_output(
                self.grounding_model,
                frame_rgb_chw_torch,
                ram_predictions,
                self.args.gsam_box_threshold,
                self.args.gsam_text_threshold,
                device=self.args.device,
            )
            size = frame_rgb_hwc_pil.size
            H, W = size[1], size[0]
            for i in range(box_coordinates.size(0)):
                box_coordinates[i] = box_coordinates[i] * torch.Tensor([W, H, W, H])
                box_coordinates[i][:2] -= box_coordinates[i][2:] / 2
                box_coordinates[i][2:] += box_coordinates[i][:2]

            box_coordinates = box_coordinates.cpu()
            nms_idx = (
                torchvision.ops.nms(
                    box_coordinates,
                    torch.tensor(detection_scores),
                    self.args.gsam_iou_threshold,
                )
                .numpy()
                .tolist()
            )
            box_coordinates = box_coordinates[nms_idx]
            text_labels = [text_labels[idx] for idx in nms_idx]
            detection_scores = [detection_scores[idx] for idx in nms_idx]

            for current_detection_index, (
                current_box_coordinates,
                current_text_label,
                current_detection_score,
            ) in enumerate(zip(box_coordinates, text_labels, detection_scores)):
                current_box_coordinates = current_box_coordinates.tolist()
                predictions.append(
                    (
                        current_frame_index,
                        current_detection_index,
                        current_box_coordinates[0],
                        current_box_coordinates[1],
                        current_box_coordinates[2],
                        current_box_coordinates[3],
                        current_text_label,
                        current_detection_score,
                    )
                )
        del frame_indices_batch
        del frames_batch
        gc.collect()
        torch.cuda.empty_cache()
        return predictions
