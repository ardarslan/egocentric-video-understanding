import os
import gc
import json

from frame_feature_extractor import FrameFeatureExtractor

import ray
import torch
import numpy as np
from detectron2.modeling import build_model
import detectron2.data.transforms as T
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from unidet.unidet.config import add_unidet_config

from typing import List


@ray.remote(num_gpus=1, num_cpus=1)
class UnidetFrameFeatureExtractor(FrameFeatureExtractor):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.cfg = self._setup_cfg(
            config_file_path=args.unidet_config_file_path,
            confidence_threshold=args.unidet_confidence_threshold,
            model_file_path=args.unidet_model_file_path,
            device=args.device,
        )
        unified_label_file = json.load(
            open(self.cfg.MULTI_DATASET.UNIFIED_LABEL_FILE, "r")
        )
        self.classes = [
            "{}".format([xx for xx in x["name"].split("_") if xx != ""][0])
            for x in unified_label_file["categories"]
        ]
        self.model = build_model(self.cfg)
        self.model.eval()
        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(self.cfg.MODEL.WEIGHTS)
        self.aug = T.ResizeShortestEdge(
            [self.cfg.INPUT.MIN_SIZE_TEST, self.cfg.INPUT.MIN_SIZE_TEST],
            self.cfg.INPUT.MAX_SIZE_TEST,
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

    def _setup_cfg(
        self,
        config_file_path,
        confidence_threshold,
        model_file_path,
        device,
    ):
        # load config from file and command-line arguments
        cfg = get_cfg()
        add_unidet_config(cfg)
        cfg.merge_from_file(config_file_path)
        cfg.MODEL.DEVICE = device
        cfg.MODEL.WEIGHTS = model_file_path
        # Set score_threshold for builtin models
        cfg.MULTI_DATASET.UNIFIED_LABEL_FILE = os.path.join(
            os.environ["CODE"],
            "scripts/04_extract_frame_features/unidet/datasets/label_spaces",
            cfg.MULTI_DATASET.UNIFIED_LABEL_FILE,
        )
        cfg.MODEL.RETINANET.SCORE_THRESH_TEST = confidence_threshold
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
        cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = (
            confidence_threshold
        )
        cfg.freeze()
        return cfg

    def predictor_function(
        self, frame_indices_batch: List[int], frames_batch: List[np.array]
    ):
        preprocessed_frames_batch = []
        for frame in frames_batch:
            frame = frame[:, :, ::-1]  # BGR->RGB
            frame = self.aug.get_transform(frame).apply_image(frame)
            frame = frame.astype("float32").transpose(2, 0, 1)  # HWC->CHW
            frame = torch.tensor(frame, device=self.args.device)
            preprocessed_frames_batch.append(
                {"image": frame, "height": frame.shape[1], "width": frame.shape[2]}
            )

        with torch.no_grad():
            predictions = self.model(preprocessed_frames_batch)

        del preprocessed_frames_batch
        gc.collect()
        torch.cuda.empty_cache()

        postprocessed_predictions = []
        for frame_index, current_frame_detections in zip(
            frame_indices_batch, predictions
        ):
            try:
                num_detections = len(current_frame_detections["instances"])
            except NotImplementedError:
                continue
            for detection_index in range(num_detections):
                detection = current_frame_detections["instances"][detection_index]
                (
                    x_top_left,
                    y_top_left,
                    x_bottom_right,
                    y_bottom_right,
                ) = detection.pred_boxes.tensor.cpu().numpy()[0]
                detection_score = detection.scores.cpu().numpy()[0]
                predicted_class_index = detection.pred_classes.cpu().numpy()[0]
                text_label = self.classes[predicted_class_index]
                postprocessed_predictions.append(
                    (
                        frame_index,
                        detection_index,
                        x_top_left,
                        y_top_left,
                        x_bottom_right,
                        y_bottom_right,
                        text_label,
                        detection_score,
                    )
                )
        del predictions
        del frame_indices_batch
        gc.collect()
        torch.cuda.empty_cache()
        return postprocessed_predictions
