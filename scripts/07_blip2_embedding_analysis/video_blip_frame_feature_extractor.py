import os
import torch
import numpy as np
from transformers import Blip2Processor
from eilev.model.utils import process
from sentence_transformers import SentenceTransformer
from frame_feature_extractor import FrameFeatureExtractor
from eilev.model.v1 import VideoBlipForConditionalGeneration

from typing import List


class VideoBLIPFrameFeatureExtractor(FrameFeatureExtractor):
    def __init__(self, args):
        super().__init__(args=args)
        self.processor = Blip2Processor.from_pretrained(
            os.path.join(os.environ["SCRATCH"], "mq_libs/video-blip-opt-2.7b-ego4d")
        )

        if torch.cuda.device_count() > 1:
            device_map = {
                "query_tokens": 0,
                "vision_model": 0,
                "language_model": 1,
                "language_projection": 1,
                "qformer": 0,
            }
        else:
            device_map = "auto"

        self.video_blip_model = VideoBlipForConditionalGeneration.from_pretrained(
            os.path.join(os.environ["SCRATCH"], "mq_libs/video-blip-opt-2.7b-ego4d"),
            device_map=device_map,
            torch_dtype=torch.float16,
        )
        self.sbert_model = SentenceTransformer("sentence-transformers/all-distilroberta-v1", device="cuda:0")
        self.question = "What is the camera wearer doing?"

    def predictor_function(
        self, frame_index: int, frames: List[np.array] # (T, H, W, C), np.uint8, BGR, 1 fps
    ):
        if len(frames) > 10:
            raise Exception("Number of frames should be at most 10.")
        with torch.no_grad():
            # sample a frame every 30 frames, i.e. 1 fps. We assume the video is 30 fps for now.
            frames = torch.tensor(np.stack([frame[:, :, ::-1] for frame in frames], axis=0), device="cuda") # (T, H, W, C), torch.int, BGR, 1 fps
            frames = frames.permute((3, 0, 1, 2)) # (C, T, H, W), torch.int, RGB, 1 fps
            model_input = process(self.processor, video=frames, text=self.question).to("cuda")
            results = self.video_blip_model.generate(
                **model_input,
                num_beams=4,
                max_new_tokens=128,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.5,
                do_sample=True,
            )
            caption_token_ids = results.pop("caption_token_ids")
            results["caption"] = self.processor.batch_decode(caption_token_ids, skip_special_tokens=True)[
                0
            ].strip()
            results["caption_sbert_embedding"] = self.sbert_model.encode([results["caption"]])[0].ravel().tolist()
            results["frame_index"] = frame_index
            return (results["frame_index"], self.question, results["caption"], results["caption_sbert_embedding"], results["language_model_input"], results["first_word_first_layer_hidden_state"], results["first_word_last_layer_hidden_state"])
