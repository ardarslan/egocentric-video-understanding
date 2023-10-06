import numpy as np
from sentence_transformers import SentenceTransformer
from frame_embedder.frame_embedder import FrameEmbedder

from typing import Dict, List


class SentenceTransformerFrameEmbedder(FrameEmbedder):
    @classmethod
    def __init__(
        cls, train_blip2_answer_word_label_mapping: Dict[str, float], unify_words: bool
    ):
        super().__init__(
            train_blip2_answer_word_label_mapping=train_blip2_answer_word_label_mapping,
            unify_words=unify_words,
        )
        cls.model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
        cls.embedding_size = 384
        cls.embedder_name = "sbert"

    @classmethod
    def get_embedding_per_frame(cls, blip2_answers: List[str], blip2_words: List[str]):
        frame_embedding = cls.model.encode(blip2_answers)
        frame_embedding = np.hstack(
            [
                np.array(frame_embedding[0]),
                np.array(frame_embedding[1]),
                np.array(frame_embedding[2]),
            ]
        )
        return frame_embedding
