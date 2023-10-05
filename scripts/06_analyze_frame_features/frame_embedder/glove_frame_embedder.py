from typing import Dict, List
import torchtext.vocab as vocab
from frame_embedder.frame_embedder import FrameEmbedder


class GloveFrameEmbedder(FrameEmbedder):
    @classmethod
    def __init__(
        cls, train_blip2_answer_word_label_mapping: Dict[str, float], unify_words: bool
    ):
        super().__init__(
            train_blip2_answer_word_label_mapping=train_blip2_answer_word_label_mapping,
            unify_words=unify_words,
        )
        cls.glove_embeddings = vocab.GloVe(name="6B", dim=100)
        cls.embedding_size = 100
        cls.embedder_name = "glove"

    @classmethod
    def get_embedding_per_frame(cls, blip2_answers: List[str], blip2_words: List[str]):
        frame_embedding = None
        if cls.unify_words:
            blip2_words = set(blip2_words)
        for word in blip2_words:
            try:
                word_weight = cls.train_blip2_answer_word_label_mapping[word]
                word_embedding = cls.glove_embeddings[word]
            except Exception as e:
                print(e)
                continue

            if frame_embedding is None:
                frame_embedding = word_weight * word_embedding
            else:
                frame_embedding += word_weight * word_embedding
        return frame_embedding
