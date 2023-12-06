from blip2_vqa_llm_encoder_output_extractor import BLIP2VQALLMEncoderOutputExtractor

# from video_blip_llm_encoder_output_extractor import VideoBLIPLLMEncoderOutputExtractor


def get_llm_encoder_output_extractor(args):
    if args.frame_feature_name == "blip2_vqa":
        remote_function = BLIP2VQALLMEncoderOutputExtractor.remote
    # elif args.frame_feature_name == "video_blip":
    # remote_function = VideoBLIPLLMEncoderOutputExtractor.remote
    else:
        raise Exception(f"{args.frame_feature_name} is not a valid frame feature name.")
    return remote_function(args=args)


def get_output_file_name(args):
    return args.frame_feature_name + "_llm_encoder_output.tsv"


def get_error_file_name(args):
    return args.frame_feature_name + "_errors.tsv"


def get_column_names(args):
    if args.frame_feature_name == "blip2_vqa":
        column_names = BLIP2VQALLMEncoderOutputExtractor.column_names
    # elif args.frame_feature_name == "video_blip":
    #     column_names = VideoBLIPLLMEncoderOutputExtractor.column_names
    else:
        raise Exception(f"{args.frame_feature_name} is not a valid frame feature name.")
    return column_names


class GlobalFrameIndex(object):
    def __init__(self):
        self.value = 0

    def get_value(self):
        return self.value

    def increment_value(self, global_frame_index):
        self.value += global_frame_index
