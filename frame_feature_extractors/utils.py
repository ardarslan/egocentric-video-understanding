from unidet.unidet_frame_feature_extractor import UnidetFrameFeatureExtractor
from ofa.ofa_frame_feature_extractor import OFAFrameFeatureExtractor
from visor_hos.visor_hos_frame_feature_extractor import VisorHOSFrameFeatureExtractor


def get_frame_feature_extractor(args):
    if args.frame_feature_name == "unidet":
        return UnidetFrameFeatureExtractor.remote(args=args)
    elif args.frame_feature_name == "visor_hos":
        return VisorHOSFrameFeatureExtractor.remote(args=args)
    elif args.frame_feature_name == "ofa":
        return OFAFrameFeatureExtractor.remote(args=args)
    else:
        raise Exception(f"{args.frame_feature_name} is not a valid frame feature name.")


def get_output_file_name(args):
    if args.frame_feature_name == "unidet":
        return UnidetFrameFeatureExtractor.output_file_name
    elif args.frame_feature_name == "visor_hos":
        return VisorHOSFrameFeatureExtractor.output_file_name
    elif args.frame_feature_name == "ofa":
        return OFAFrameFeatureExtractor.output_file_name
    else:
        raise Exception(f"{args.frame_feature_name} is not a valid frame feature name.")


def get_error_file_name(args):
    if args.frame_feature_name == "unidet":
        return UnidetFrameFeatureExtractor.error_file_name
    elif args.frame_feature_name == "visor_hos":
        return VisorHOSFrameFeatureExtractor.error_file_name
    elif args.frame_feature_name == "ofa":
        return OFAFrameFeatureExtractor.error_file_name
    else:
        raise Exception(f"{args.frame_feature_name} is not a valid frame feature name.")


def get_column_names(args):
    if args.frame_feature_name == "unidet":
        return UnidetFrameFeatureExtractor.column_names
    elif args.frame_feature_name == "visor_hos":
        return VisorHOSFrameFeatureExtractor.column_names
    elif args.frame_feature_name == "ofa":
        return OFAFrameFeatureExtractor.column_names
    else:
        raise Exception(f"{args.frame_feature_name} is not a valid frame feature name.")
