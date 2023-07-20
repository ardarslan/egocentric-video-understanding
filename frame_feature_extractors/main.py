import os
import argparse

import ray
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from unidet.unidet_feature_extractor import UnidetFrameFeatureExtractor
from ofa.ofa_feature_extractor import OFAFrameFeatureExtractor
from visor_hos.visor_hos_feature_extractor import VisorHOSFrameFeatureExtractor


def get_parser():
    parser = argparse.ArgumentParser(description="Argument parser")

    parser.add_argument("--feature_name", type=str, choices=["unidet", "ofa", "visor-hos"], default="unidet")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--num_devices", type=int, default=4)
    parser.add_argument(
        "--input_folder_path",
        type=str,
        default="/srv/beegfs02/scratch/aarslan_data/data/ego4d_data/v2/full_scale",
        help="The directory of the input videos.",
    )
    parser.add_argument(
        "--output_folder_path",
        type=str,
        default="/srv/beegfs02/scratch/aarslan_data/data/ego4d_data/v2/frame_features",
        help="The directory of the output videos.",
    )

    # ofa configs
    parser.add_argument(
        "--ofa_checkpoint_dir",
        type=str,
        default="/scratch/aarslan/mq_libs/OFA-huge",
        help="The directory of the checkpoint of the pre-trained OFA-huge model.",
    )

    # unidet configs
    parser.add_argument(
        "--unidet_config_file",
        default="/home/aarslan/mq/frame_feature_extractors/unidet/configs/Unified_learned_OCIM_R50_6x+2x.yaml",
        type=str,
    )
    parser.add_argument("--unidet_confidence_threshold", type=float, default=0.4)
    parser.add_argument(
        "--unidet_opts",
        type=list,
        default=["MODEL.WEIGHTS", "/home/aarslan/mq/frame_feature_extractors/unidet/models/Unified_learned_OCIM_R50_6x+2x.pth", "MODEL.DEVICE", "cuda"],
        nargs=argparse.REMAINDER,
    )

    # visor-hos configs
    parser.add_argument("--visor_hos_config_file", default="/home/aarslan/mq/frame_feature_extractors/visor_hos/configs/hos_pointrend_rcnn_R_50_FPN_1x.yaml", type=str)
    parser.add_argument("--visor_hos_model_file", default="/home/aarslan/mq/frame_feature_extractors/visor_hos/models/model_final_hos.pth", type=str)

    return parser


class FeatureExtractor(object):
    def __init__(self, args):
        self.feature_name = args.feature_name
        if self.feature_name == "unidet":
            self.feature_extractor = UnidetFrameFeatureExtractor(args=args)
        elif self.feature_name == "ofa":
            self.feature_extractor = OFAFrameFeatureExtractor(args=args)
        elif self.feature_name == "visor-hos":
            self.feature_extractor = VisorHOSFrameFeatureExtractor(args=args)
        else:
            raise Exception(f"{self.feature_name} is not a valid feature name.")

    def frame_from_video(self, cap):
        while cap.isOpened():
            success, frame = cap.read()
            if success:
                yield frame
            else:
                break

    @ray.remote(num_gpus=1)
    def extract_frame_features(self, frame_index: int, frame: np.array):
        return self.feature_extractor.extract_frame_features(frame_index=frame_index, frame=frame)

    def extract_features(self, input_video_file_path: str, output_folder_path: str):
        cap = cv2.VideoCapture(input_video_file_path)
        frame_generator = self.frame_from_video(cap)
        features = [ray.get([self.extract_frame_features.remote(frame_index, frame) for frame_index, frame in enumerate(frame_generator)])]
        features_df = pd.DataFrame(
            data=features,
            columns=self.feature_extractor.column_names,
        )
        features_df.to_csv(
            os.path.join(
                output_folder_path,
                self.feature_extractor.file_name_wo_ext + ".tsv",
            ),
            sep="\t",
            index=False,
        )
        cap.release()


if __name__ == "__main__":
    args = get_parser().parse_args()
    if args.device == "cuda":
        ray.init(num_gpus=args.num_devices)
    elif args.device == "cpu":
        ray.init(num_cpus=args.num_devices)

    feature_extractor = FeatureExtractor(args=args)

    for file_name in tqdm(list(os.listdir(args.input_folder_path))):
        if file_name[-4:] != ".mp4":
            continue
        file_name_wo_ext = file_name[:-4]
        current_input_video_file_path = os.path.join(args.input_folder_path, file_name)
        current_output_folder_path = os.path.join(args.output_folder_path, file_name_wo_ext)
        if os.path.exists(os.path.join(current_output_folder_path, feature_extractor.file_name_wo_ext + ".tsv")):
            continue
        os.makedirs(current_output_folder_path, exist_ok=True)
        feature_extractor.extract_features(
            input_video_file_path=current_input_video_file_path,
            output_folder_path=current_output_folder_path,
        )
    ray.shutdown()
