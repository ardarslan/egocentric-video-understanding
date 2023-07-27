import os
import argparse
import traceback

import ray

from utils import get_frame_feature_extractor, get_column_names, get_output_file_name, get_error_file_name
from frame_feature_extractor import FrameFeatureExtractor


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Argument parser")

    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--num_devices", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--frame_feature_name", type=str, choices=["visor_hos", "unidet", "ofa", "blip_captioning", "blip_vqa", "gsam"], default="gsam")

    parser.add_argument("--unidet_confidence_threshold", type=float, default=0.4)
    parser.add_argument(
        "--unidet_model_file_path",
        type=str,
        default="/scratch/aarslan/mq_pretrained_models/unidet/Unified_learned_OCIM_R50_6x+2x.pth",
    )
    parser.add_argument(
        "--unidet_config_file_path",
        type=str,
        default="/home/aarslan/mq/frame_feature_extractors/unidet/configs/Unified_learned_OCIM_R50_6x+2x.yaml",
    )

    parser.add_argument("--visor_hos_confidence_threshold", type=float, default=0.4)
    parser.add_argument(
        "--visor_hos_model_file_path",
        type=str,
        default="/scratch/aarslan/mq_pretrained_models/visor_hos/model_final_hos.pth",
    )
    parser.add_argument(
        "--visor_hos_config_file_path",
        type=str,
        default="/home/aarslan/mq/frame_feature_extractors/visor_hos/configs/hos_pointrend_rcnn_R_50_FPN_1x.yaml",
    )

    parser.add_argument(
        "--ofa_model_file_path",
        type=str,
        default="/scratch/aarslan/mq_pretrained_models/ofa",
    )

    parser.add_argument(
        "--blip_captioning_model_file_path",
        type=str,
        default="/scratch/aarslan/mq_pretrained_models/blip/model_base_capfilt_large.pth",
    )
    parser.add_argument(
        "--blip_vqa_model_file_path",
        type=str,
        default="/scratch/aarslan/mq_pretrained_models/blip/model_base_vqa_capfilt_large.pth",
    )

    parser.add_argument(
        "--gsam_grounding_config_file_path",
        type=str,
        default="/home/aarslan/mq/frame_feature_extractors/gsam/gsam/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        help="path to config file",
    )
    parser.add_argument("--gsam_grounding_model_file_path", type=str, default="/scratch/aarslan/mq_pretrained_models/gsam/groundingdino_swint_ogc.pth", help="path to checkpoint file")
    parser.add_argument("--gsam_ram_model_file_path", type=str, default="/scratch/aarslan/mq_pretrained_models/gsam/ram_swin_large_14m.pth", help="path to checkpoint file")
    parser.add_argument("--gsam_box_threshold", type=float, default=0.25, help="box threshold")
    parser.add_argument("--gsam_text_threshold", type=float, default=0.2, help="text threshold")
    parser.add_argument("--gsam_iou_threshold", type=float, default=0.5, help="iou threshold")

    parser.add_argument(
        "--input_folder_path",
        type=str,
        default="/home/aarslan/mq/sample_data/inputs",
    )
    parser.add_argument(
        "--output_folder_path",
        type=str,
        default="/home/aarslan/mq/sample_data/outputs",
    )
    parser.add_argument(
        "--error_folder_path",
        type=str,
        default="/home/aarslan/mq/error_files/",
    )

    args = parser.parse_args()

    os.makedirs(args.error_folder_path, exist_ok=True)

    ray.init(num_gpus=args.num_devices)

    frame_feature_extractor_pool = ray.util.ActorPool([get_frame_feature_extractor(args=args) for _ in range(args.num_devices)])
    column_names = get_column_names(args=args)
    output_file_name = get_output_file_name(args=args)
    error_file_name = get_error_file_name(args=args)

    for input_video_file_name in os.listdir(args.input_folder_path):
        try:
            input_video_file_path = os.path.join(args.input_folder_path, input_video_file_name)
            frame_indices_batches, frames_batches = FrameFeatureExtractor.get_frame_indices_batches_and_frames_batches(input_video_file_path=input_video_file_path, batch_size=args.batch_size)
            results_list = frame_feature_extractor_pool.map(
                lambda frame_feature_extractor, inputs: frame_feature_extractor.predictor_function.remote(inputs[0], inputs[1]), list(zip(frame_indices_batches, frames_batches))[:2]
            )
            FrameFeatureExtractor.save_results(
                input_video_file_path=input_video_file_path, results_list=results_list, output_folder_path=args.output_folder_path, column_names=column_names, output_file_name=output_file_name
            )
        except Exception as e:
            print(f"Error with file {input_video_file_name}: \n {traceback.format_exc()}")
            e = "-" * 100
            print(e)
            with open(os.path.join(args.error_folder_path, error_file_name), "a") as error_file_writer:
                error_file_writer.write(f"Error with file {input_video_file_name}: \n {traceback.format_exc()}")
                error_file_writer.write(e)

    ray.shutdown()
