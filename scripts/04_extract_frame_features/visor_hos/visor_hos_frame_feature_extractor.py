import ray
import numpy as np
import torch

from visor_hos.visor_hos.data.datasets.epick import register_epick_instances
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
import detectron2.data.transforms as T
from detectron2.config import get_cfg
from detectron2.projects import point_rend
from detectron2.structures import Instances

from frame_feature_extractor import FrameFeatureExtractor

from typing import List, Tuple


@ray.remote(num_gpus=1)
class VisorHOSFrameFeatureExtractor(FrameFeatureExtractor):
    def __init__(self, args):
        super().__init__()
        self.args = args
        version = "datasets/epick_visor_coco_hos"
        register_epick_instances(
            "epick_visor_2022_val_hos",
            {},
            f"{version}/annotations/val.json",
            f"{version}/val",
        )
        self.classes = ["hand", "object"]

        self.cfg = self._setup_cfg(
            config_file_path=args.visor_hos_config_file_path,
            confidence_threshold=args.visor_hos_confidence_threshold,
            model_file_path=args.visor_hos_model_file_path,
            device=args.device,
        )
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
        cfg = get_cfg()
        point_rend.add_pointrend_config(cfg)
        cfg.merge_from_file(config_file_path)
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
        cfg.MODEL.POINT_HEAD.NUM_CLASSES = 2
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
        cfg.MODEL.WEIGHTS = model_file_path
        cfg.MODEL.DEVICE = device
        return cfg

    def _get_offset(self, h_bbox_xyxy, o_bbox_xyxy):
        """
        Calculate offset from hand to object bbox, using xyxy bbox annotation.
        """
        h_center = [
            int((h_bbox_xyxy[0] + h_bbox_xyxy[2]) / 2),
            int((h_bbox_xyxy[1] + h_bbox_xyxy[3]) / 2),
        ]
        o_center = [
            int((o_bbox_xyxy[0] + o_bbox_xyxy[2]) / 2),
            int((o_bbox_xyxy[1] + o_bbox_xyxy[3]) / 2),
        ]
        # offset: [vx, vy, magnitute]
        scalar = 1000
        vec = np.array([o_center[0] - h_center[0], o_center[1] - h_center[1]]) / scalar
        norm = np.linalg.norm(vec)
        unit_vec = vec / norm
        offset = [unit_vec[0], unit_vec[1], norm]
        return offset

    def _get_center(self, box):
        box = box.pred_boxes.tensor.cpu().detach().numpy()[0]
        x0, y0, x1, y1 = box
        center = [int((x0 + x1) / 2), int((y0 + y1) / 2)]
        return center

    def _get_incontact_obj(self, h_box, offset, obj_preds):
        """
        Find in-contact object for hand that is predicted as in-contact.
        """
        h_center = self._get_center(h_box)
        scalar = 1000
        offset_vec = [offset[0] * offset[2] * scalar, offset[1] * offset[2] * scalar]
        pred_o_center = [h_center[0] + offset_vec[0], h_center[1] + offset_vec[1]]

        # choose from obj_preds
        dist_ls = []
        for i in range(len(obj_preds)):
            o_center = self._get_center(obj_preds[i])
            dist = np.linalg.norm(np.array(o_center) - np.array(pred_o_center))
            dist_ls.append(dist)

        if len(dist_ls) == 0:
            return []
        else:
            o_ind = np.argmin(np.array(dist_ls))
            return obj_preds[int(o_ind)]

    def _postprocessing_function(self, predictions):
        preds = predictions["instances"].to("cpu")
        if len(preds) == 0:
            return predictions
        # separate hand, obj preds
        hand_preds = preds[preds.pred_classes == 0]
        obj_preds = preds[preds.pred_classes == 1]

        if len(obj_preds) == 0:
            return {"instances": hand_preds}

        # find incontact obj
        incontact_obj = []
        updated_hand_preds = []
        for i in range(len(hand_preds)):
            box = hand_preds[i].pred_boxes.tensor.cpu().detach().numpy()[0]
            contact = hand_preds[i].pred_contacts.cpu().detach().numpy()[0]
            offset = hand_preds[i].pred_offsets.cpu().detach().numpy()[0]
            # if incontact
            if int(np.argmax(contact)):
                obj = self._get_incontact_obj(hand_preds[i], offset, obj_preds)
                if isinstance(obj, Instances):
                    incontact_obj.append(obj)
                    new = Instances(hand_preds[i].image_size)
                    for field in hand_preds[i]._fields:
                        if field == "pred_offsets":
                            new.set(
                                field,
                                torch.Tensor(
                                    [
                                        self._get_offset(
                                            box,
                                            obj.pred_boxes.tensor.cpu()
                                            .detach()
                                            .numpy()[0],
                                        )
                                    ]
                                ),
                            )
                        else:
                            new.set(field, hand_preds[i].get(field))
                    updated_hand_preds.append(new)
            else:
                updated_hand_preds.append(hand_preds[i])

        if len(incontact_obj) > 0:
            incontact_obj.extend(updated_hand_preds)
            ho = Instances.cat(incontact_obj)
        else:
            if len(updated_hand_preds) > 0:
                ho = Instances.cat(updated_hand_preds)
            else:
                ho = Instances(preds[0].image_size)

        return {"instances": ho}

    def predictor_function(
        self, frame_indices_batch: List[int], frames_batch: List[np.array]
    ):
        predictions = []
        for frame_index, frame in zip(frame_indices_batch, frames_batch):
            frame = frame[:, :, ::-1]  # BGR->RGB
            frame = self.aug.get_transform(frame).apply_image(frame)
            frame = frame.astype("float32").transpose(2, 0, 1)  # HWC->CHW
            frame = torch.tensor(frame, device=self.args.device)
            frame = {"image": frame, "height": frame.shape[1], "width": frame.shape[2]}
            with torch.no_grad():
                frame = self._postprocessing_function(self.model([frame])[0])
            try:
                num_detections = len(frame["instances"])
            except NotImplementedError:
                continue
            for detection_index in range(num_detections):
                detection = frame["instances"][detection_index]
                (
                    x_top_left,
                    y_top_left,
                    x_bottom_right,
                    y_bottom_right,
                ) = detection.pred_boxes.tensor.cpu().numpy()[0]
                detection_score = detection.scores.cpu().numpy()[0]
                predicted_class_index = detection.pred_classes.cpu().numpy()[0]
                if predicted_class_index != 1:
                    continue
                text_label = self.classes[predicted_class_index]
                predictions.append(
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

        return predictions
