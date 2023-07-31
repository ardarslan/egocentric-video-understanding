from unidet.unidet_frame_feature_extractor import UnidetFrameFeatureExtractor
from ofa.ofa_frame_feature_extractor import OFAFrameFeatureExtractor
from visor_hos.visor_hos_frame_feature_extractor import VisorHOSFrameFeatureExtractor
from blip.blip_captioning_frame_feature_extractor import BLIPCaptioningFrameFeatureExtractor
from blip.blip_vqa_frame_feature_extractor import BLIPVQAFrameFeatureExtractor
from gsam.gsam_frame_feature_extractor import GSAMFrameFeatureExtractor
from ego_hos.ego_hos_frame_feature_extractor import EgoHOSFrameFeatureExtractor


def get_frame_feature_extractor(args):
    if args.frame_feature_name == "unidet":
        remote_function = UnidetFrameFeatureExtractor.remote
    elif args.frame_feature_name == "visor_hos":
        remote_function = VisorHOSFrameFeatureExtractor.remote
    elif args.frame_feature_name == "ofa":
        remote_function = OFAFrameFeatureExtractor.remote
    elif args.frame_feature_name == "blip_captioning":
        remote_function = BLIPCaptioningFrameFeatureExtractor.remote
    elif args.frame_feature_name == "blip_vqa":
        remote_function = BLIPVQAFrameFeatureExtractor.remote
    elif args.frame_feature_name == "gsam":
        remote_function = GSAMFrameFeatureExtractor.remote
    elif args.frame_feature_name == "ego_hos":
        remote_function = EgoHOSFrameFeatureExtractor.remote
    else:
        raise Exception(f"{args.frame_feature_name} is not a valid frame feature name.")
    return remote_function(args=args)


def get_output_file_name(args):
    return args.frame_feature_name + "_features.tsv"


def get_error_file_name(args):
    return args.frame_feature_name + "_errors.tsv"


def get_column_names(args):
    if args.frame_feature_name == "unidet":
        column_names = UnidetFrameFeatureExtractor.column_names
    elif args.frame_feature_name == "visor_hos":
        column_names = VisorHOSFrameFeatureExtractor.column_names
    elif args.frame_feature_name == "ofa":
        column_names = OFAFrameFeatureExtractor.column_names
    elif args.frame_feature_name == "blip_captioning":
        column_names = BLIPCaptioningFrameFeatureExtractor.column_names
    elif args.frame_feature_name == "blip_vqa":
        column_names = BLIPVQAFrameFeatureExtractor.column_names
    elif args.frame_feature_name == "gsam":
        column_names = GSAMFrameFeatureExtractor.column_names
    elif args.frame_feature_name == "ego_hos":
        column_names = EgoHOSFrameFeatureExtractor.column_names
    else:
        raise Exception(f"{args.frame_feature_name} is not a valid frame feature name.")
    return column_names
