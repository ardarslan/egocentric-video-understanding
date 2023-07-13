# Code adapted from 
import argparse
import glob
import multiprocessing as mp
import os
import time
import cv2
import sys
import tqdm
from PIL import Image
import pandas as pd
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import _create_text_labels
from utils import UnifiedVisualizationDemo, setup_cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin models")
    parser.add_argument(
        "--config_file",
        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--input_folder_path", help="A directory to read input videos.")
    parser.add_argument(
        "--output_folder_path",
        help="A directory to save output frames."
    )
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=0.4, # Unidet uses 0.5 and TransFusion uses 0.4.
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    # args = get_parser().parse_args()
    args = argparse.Namespace(**{"config_file": "configs/Unified_learned_OCIM_R50_6x+2x.yaml",
                                 "input_folder_path": "../../sample_data/inputs",
                                 "output_folder_path": "../../sample_data/outputs",
                                 "opts": ["MODEL.WEIGHTS", "models/Unified_learned_OCIM_R50_6x+2x.pth"],
                                 "confidence_threshold": 0.4})
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = UnifiedVisualizationDemo(cfg)

    for current_input_video_file_name in os.listdir(args.input_folder_path):
        if current_input_video_file_name[-4:] != ".mp4":
            continue
        current_input_video_file_path = os.path.join(args.input_folder_path, current_input_video_file_name)
        current_cap = cv2.VideoCapture(current_input_video_file_path)

        current_input_video_file_name_wo_ext = current_input_video_file_name[:-4]

        os.makedirs(os.path.join(args.output_folder_path, current_input_video_file_name_wo_ext), exist_ok=True)

        current_detections = []

        for frame_index, (frame, predictions, visualization) in tqdm.tqdm(enumerate(demo.run_on_video(current_cap)), total=int(current_cap.get(cv2.CAP_PROP_FRAME_COUNT))):
            if frame_index == 0:
                Image.fromarray(frame[:, :, ::-1]).save(os.path.join(args.output_folder_path, current_input_video_file_name_wo_ext, f"{str(frame_index).zfill(6)}_input.png"))
                Image.fromarray(visualization[:, :, ::-1]).save(os.path.join(args.output_folder_path, current_input_video_file_name_wo_ext, f"{str(frame_index).zfill(6)}_output.png"))
            instances = predictions["instances"]

            for instance_index in range(len(instances)):
                instance = instances[instance_index]
                score = float(instance.scores.cpu().numpy())
                predicted_class = int(instance.pred_classes.cpu().numpy())
                # text_label = _create_text_labels(predicted_class, score, metadata.get("thing_classes", None))
                predicted_box_coordinates = instance.pred_boxes.tensor.cpu().tolist()[0]
                x_top_left, y_top_left, x_bottom_right, y_bottom_right = predicted_box_coordinates[0], predicted_box_coordinates[1], predicted_box_coordinates[2], predicted_box_coordinates[3]
                current_detections.append((frame_index, instance_index, x_top_left, y_top_left, x_bottom_right, y_bottom_right, text_label, score))

        current_cap.release()
        current_detections_df = pd.DataFrame(current_detections, columns=["frame_index", "instance_index", "x_top_left", "y_top_left", "x_bottom_right", "y_bottom_right", "text_label", "score"])
        current_detections_df.to_csv(os.path.join(args.output_folder_path, current_input_video_file_name_wo_ext, "unidet_detections.tsv"), sep="\t", index=False)
