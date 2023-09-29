import numpy as np

from frame_embedder.frame_embedder import FrameEmbedder

from typing import Dict, List


class OneHotFrameEmbedder(FrameEmbedder):
    @classmethod
    def __init__(
        cls, train_blip2_answer_word_label_mapping: Dict[str, float], unify_words: bool
    ):
        super().__init__(
            train_blip2_answer_word_label_mapping=train_blip2_answer_word_label_mapping,
            unify_words=unify_words,
        )
        cls.vocabulary = list(train_blip2_answer_word_label_mapping.keys())
        cls.embedding_size = len(cls.vocabulary)
        cls.embedder_name = "one_hot"

    @classmethod
    def get_embedding_per_frame(cls, blip2_answers: List[str], blip2_words: List[str]):
        words = cls.process_per_frame_blip2_answers(blip2_answers)
        frame_embedding = None
        if cls.unify_words:
            words = set(words)
        for word in words:
            try:
                word_weight = cls.train_blip2_answer_word_label_mapping[word]
            except Exception as e:
                print(e)
                continue
            word_embedding = np.zeros(len(cls.vocabulary))
            word_embedding[cls.vocabulary.index(word)] = 1
            if frame_embedding is None:
                frame_embedding = word_weight * word_embedding
            else:
                frame_embedding += word_weight * word_embedding
        return frame_embedding
