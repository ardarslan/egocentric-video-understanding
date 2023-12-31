import gc

import ray
import torch
import numpy as np
from mmseg.apis import init_segmentor
from mmseg.datasets.pipelines import Compose
from mmcv.parallel import collate, scatter

from frame_feature_extractor import FrameFeatureExtractor
from ego_hos.utils import LoadImage

from typing import List, Dict, Any
from PIL import Image


@ray.remote(num_gpus=1, num_cpus=1)
class EgoHOSFrameFeatureExtractor(FrameFeatureExtractor):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.seg_twohands_model = init_segmentor(
            args.ego_hos_seg_twohands_config_file_path,
            args.ego_hos_seg_twohands_model_file_path,
            device=self.args.device,
        )

        self.twohands_to_cb_model = init_segmentor(
            args.ego_hos_twohands_to_cb_config_file_path,
            args.ego_hos_twohands_to_cb_model_file_path,
            device=self.args.device,
        )

        self.twohands_cb_to_obj2_model = init_segmentor(
            args.ego_hos_twohands_cb_to_obj2_config_file_path,
            args.ego_hos_twohands_cb_to_obj2_model_file_path,
            device=self.args.device,
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
            "object_label",
        ]

    def inference_segmentor(model, img, previous_results):
        """Inference image(s) with the segmentor.

        Args:
            model (nn.Module): The loaded segmentor.
            imgs (str/ndarray or list[str/ndarray]): Either image files or loaded
                images.

        Returns:
            (list[Tensor]): The segmentation result.
        """
        cfg = model.cfg
        device = next(model.parameters()).device  # model device
        # build the data pipeline
        test_pipeline = [LoadImage()] + cfg.data.test.pipeline[1:]
        test_pipeline = Compose(test_pipeline)
        # prepare data
        data = dict(img=img)
        data = test_pipeline(data)
        data = collate([data], samples_per_gpu=1)
        if next(model.parameters()).is_cuda:
            # scatter to specified GPU
            data = scatter(data, [device])[0]
        else:
            data["img_metas"] = [i.data[0] for i in data["img_metas"]]

        if "additional_channel" in cfg.keys():
            data["img_metas"][0][0]["additional_channel"] = cfg["additional_channel"]
        if "twohands_dir" in cfg.keys():
            data["img_metas"][0][0]["twohands_dir"] = cfg["twohands_dir"]
        if "cb_dir" in cfg.keys():
            data["img_metas"][0][0]["cb_dir"] = cfg["cb_dir"]

        with torch.no_grad():
            result = model(
                return_loss=False,
                rescale=True,
                previous_results=previous_results,
                **data
            )
        return result

    def predictor_function(
        self,
        frame_indices_batch: List[int],
        frames_batch: List[np.array],
        gsam_features_batch: List[List[Dict[str, Any]]],
    ):
        predictions = []
        for frame_index, frame, gsam_feature_rows in zip(
            frame_indices_batch,
            frames_batch,
            gsam_features_batch,
        ):
            seg_twohands_result = self.inference_segmentor(
                self.seg_twohands_model, frame, previous_results={}
            )[0].astype(np.uint8)

            twohands_to_cb_result = self.inference_segmentor(
                self.twohands_to_cb_model,
                frame,
                previous_results={
                    "seg_twohands_result": Image.fromarray(seg_twohands_result)
                },
            )[0].astype(np.uint8)

            twohands_cb_to_obj2_result = self.inference_segmentor(
                self.twohands_cb_to_obj2_model,
                frame,
                previous_results={
                    "seg_twohands_result": Image.fromarray(seg_twohands_result),
                    "twohands_to_cb_result": Image.fromarray(twohands_to_cb_result),
                },
            )[0]

            left_hand_first_order_object_pixels = np.zeros(frame.shape)
            left_hand_first_order_object_pixels[twohands_cb_to_obj2_result == 1] = 1

            right_hand_first_order_object_pixels = np.zeros(frame.shape)
            right_hand_first_order_object_pixels[twohands_cb_to_obj2_result == 2] = 1

            both_hands_first_order_object_pixels = np.zeros(frame.shape)
            both_hands_first_order_object_pixels[twohands_cb_to_obj2_result == 3] = 1

            left_hand_second_order_object_pixels = np.zeros(frame.shape)
            left_hand_second_order_object_pixels[twohands_cb_to_obj2_result == 4] = 1

            right_hand_second_order_object_pixels = np.zeros(frame.shape)
            right_hand_second_order_object_pixels[twohands_cb_to_obj2_result == 5] = 1

            both_hands_second_order_object_pixels = np.zeros(frame.shape)
            both_hands_second_order_object_pixels[twohands_cb_to_obj2_result == 6] = 1

            diagonal_length = frame.shape[0] * np.sqrt(2)

            object_label_object_pixels_mapping = {
                "left_hand_first_order": left_hand_first_order_object_pixels,
                "right_hand_first_order": right_hand_first_order_object_pixels,
                "both_hands_first_order": both_hands_first_order_object_pixels,
                "left_hand_second_order": left_hand_second_order_object_pixels,
                "right_hand_second_order": right_hand_second_order_object_pixels,
                "both_hands_second_order": both_hands_second_order_object_pixels,
            }

            for gsam_feature_row in gsam_feature_rows:
                x_top_left = np.round(gsam_feature_row["x_top_left"]).astype(np.int32)
                y_top_left = np.round(gsam_feature_row["y_top_left"]).astype(np.int32)
                x_bottom_right = np.round(gsam_feature_row["x_bottom_right"]).astype(
                    np.int32
                )
                y_bottom_right = np.round(gsam_feature_row["y_bottom_right"]).astype(
                    np.int32
                )

                current_bounding_box_pixels = np.zeros(frame.shape)
                current_bounding_box_pixels[
                    y_top_left:y_bottom_right, x_top_left:x_bottom_right
                ] = 1

                for (
                    object_label,
                    object_pixels,
                ) in object_label_object_pixels_mapping.items():
                    number_of_object_pixels_inside_the_bounding_box = (
                        object_pixels * current_bounding_box_pixels
                    ).sum()
                    if (
                        number_of_object_pixels_inside_the_bounding_box
                        < diagonal_length
                    ):
                        continue
                    number_of_object_pixels_outside_the_bounding_box = (
                        object_pixels * (1 - current_bounding_box_pixels)
                    ).sum()

                    if (
                        number_of_object_pixels_inside_the_bounding_box
                        / (
                            float(number_of_object_pixels_outside_the_bounding_box)
                            + 1e-6
                        )
                        >= 0.25
                    ):
                        predictions.append(
                            (
                                frame_index,
                                gsam_feature_row["detection_index"],
                                x_top_left,
                                y_top_left,
                                x_bottom_right,
                                y_bottom_right,
                                gsam_feature_row["text_label"],
                                gsam_feature_row["detection_score"],
                                object_label,
                            )
                        )
        del frame_indices_batch
        del frames_batch
        del gsam_features_batch
        gc.collect()
        torch.cuda.empty_cache()
        return predictions
