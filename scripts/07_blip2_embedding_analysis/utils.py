from blip2_vqa_frame_feature_extractor import BLIP2VQAFrameFeatureExtractor
from video_blip_frame_feature_extractor import VideoBLIPFrameFeatureExtractor


def get_frame_feature_extractor(args):
    if args.frame_feature_name == "blip2_vqa":
        frame_feature_extractor = BLIP2VQAFrameFeatureExtractor
    elif args.frame_feature_name == "video_blip":
        frame_feature_extractor = VideoBLIPFrameFeatureExtractor
    else:
        raise Exception(f"{args.frame_feature_name} is not a valid frame feature name.")
    return frame_feature_extractor(args=args)


def get_output_file_name(args):
    return args.frame_feature_name + "_features.tsv"


def get_error_file_name(args):
    return args.frame_feature_name + "_errors.tsv"
