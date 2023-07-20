import torch
import numpy as np

from visor_hos.hos.data.datasets.epick import register_epick_instances
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.projects import point_rend
from detectron2.data import MetadataCatalog
from detectron2.structures import Instances

from typing import List, Tuple, Any


class VisorHOSFrameFeatureExtractor(object):
    def __init__(self, args):
        version = "datasets/epick_visor_coco_hos"
        register_epick_instances("epick_visor_2022_val_hos", {}, f"{version}/annotations/val.json", f"{version}/val")
        MetadataCatalog.get("epick_visor_2022_val_hos").thing_classes = ["hand", "object"]
        self.metadata = MetadataCatalog.get("epick_visor_2022_val_hos")
        cfg = get_cfg()
        point_rend.add_pointrend_config(cfg)
        cfg.merge_from_file(args.visor_hos_config_file)
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
        cfg.MODEL.POINT_HEAD.NUM_CLASSES = 2
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        cfg.MODEL.WEIGHTS = args.visor_hos_model_file
        self.visor_hos_predictor = DefaultPredictor(cfg)
        self.column_names = [
            "frame_index",
            "detection_index",
            "x_top_left",
            "y_top_left",
            "x_bottom_right",
            "y_bottom_right",
            "text_label",
            "detection_score",
        ]
        self.file_name_wo_ext = "visor_hos_features"

    def hos_postprocessing(self, predictions):
        """
        Use predicted offsets to associate hand and its in-contact obj.
        """
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
                obj = self.get_incontact_obj(hand_preds[i], offset, obj_preds)
                if isinstance(obj, Instances):
                    incontact_obj.append(obj)
                    new = Instances(hand_preds[i].image_size)
                    for field in hand_preds[i]._fields:
                        if field == "pred_offsets":
                            new.set(
                                field,
                                torch.Tensor(
                                    [
                                        self.get_offset(
                                            box,
                                            obj.pred_boxes.tensor.cpu().detach().numpy()[0],
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

    def get_offset(self, h_bbox_xyxy, o_bbox_xyxy):
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

    def get_incontact_obj(self, h_box, offset, obj_preds):
        """
        Find in-contact object for hand that is predicted as in-contact.
        """
        h_center = self.get_center(h_box)
        scalar = 1000
        offset_vec = [offset[0] * offset[2] * scalar, offset[1] * offset[2] * scalar]
        pred_o_center = [h_center[0] + offset_vec[0], h_center[1] + offset_vec[1]]

        # choose from obj_preds
        dist_ls = []
        for i in range(len(obj_preds)):
            o_center = self.get_center(obj_preds[i])
            dist = np.linalg.norm(np.array(o_center) - np.array(pred_o_center))
            dist_ls.append(dist)

        if len(dist_ls) == 0:
            return []
        else:
            o_ind = np.argmin(np.array(dist_ls))
            return obj_preds[int(o_ind)]

    def get_center(self, box):
        box = box.pred_boxes.tensor.cpu().detach().numpy()[0]
        x0, y0, x1, y1 = box
        center = [int((x0 + x1) / 2), int((y0 + y1) / 2)]
        return center

    def extract_frame_features(self, frame_index: int, frame: np.array) -> List[Tuple[Any, ...]]:
        # """frames[0] is in BGR format."""
        detections = self.hos_postprocessing(self.visor_hos_predictor(frame))["instances"]
        features = []
        for current_detection_index in range(len(detections)):
            current_detection = detections[current_detection_index]
            predicted_class = int(current_detection.pred_classes.numpy()[0])
            text_label = self.metadata.get("thing_classes", None)[predicted_class]
            if text_label == "hand":
                continue
            detection_score = float(current_detection.scores.numpy()[0])
            predicted_box_coordinates = current_detection.pred_boxes.tensor.tolist()[0]
            x_top_left, y_top_left, x_bottom_right, y_bottom_right = (
                predicted_box_coordinates[0],
                predicted_box_coordinates[1],
                predicted_box_coordinates[2],
                predicted_box_coordinates[3],
            )
            features.append(
                (
                    frame_index,
                    current_detection_index,
                    x_top_left,
                    y_top_left,
                    x_bottom_right,
                    y_bottom_right,
                    text_label,
                    detection_score,
                )
            )
        return features
