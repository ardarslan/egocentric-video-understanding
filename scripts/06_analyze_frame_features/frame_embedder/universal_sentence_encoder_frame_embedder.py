import tensorflow_hub

import numpy as np
from frame_embedder.frame_embedder import FrameEmbedder

from typing import Dict, List


class UniversalSentenceEncoderFrameEmbedder(FrameEmbedder):
    @classmethod
    def __init__(
        cls, train_blip2_answer_word_label_mapping: Dict[str, float], unify_words: bool
    ):
        super().__init__(
            train_blip2_answer_word_label_mapping=train_blip2_answer_word_label_mapping,
            unify_words=unify_words,
        )
        cls.model = tensorflow_hub.load(
            "https://tfhub.dev/google/universal-sentence-encoder/4"
        )
        cls.embedding_size = 512
        cls.embedder_name = "universal_sentence_encoder"

    @classmethod
    def get_embedding_per_frame(cls, blip2_answers: List[str], blip2_words: List[str]):
        frame_embedding = cls.model(blip2_answers)
        frame_embedding = np.hstack(
            [
                np.array(frame_embedding[0]),
                np.array(frame_embedding[1]),
                np.array(frame_embedding[2]),
            ]
        )
        return frame_embedding
