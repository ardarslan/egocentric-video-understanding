import os
import gensim

from frame_embedder.frame_embedder import FrameEmbedder

from typing import Dict, List


class Word2VecFrameEmbedder(FrameEmbedder):
    @classmethod
    def __init__(
        cls, train_blip2_answer_word_label_mapping: Dict[str, float], unify_words: bool
    ):
        super().__init__(
            train_blip2_answer_word_label_mapping=train_blip2_answer_word_label_mapping,
            unify_words=unify_words,
        )
        cls.word2vec_embeddings = gensim.models.KeyedVectors.load_word2vec_format(
            os.path.join(
                os.environ["SCRATCH"],
                "mq_libs/word2vec",
                "GoogleNews-vectors-negative300.bin",
            ),
            binary=True,
        )
        cls.embedding_size = 300
        cls.embedder_name = "word2vec"

    @classmethod
    def get_embedding_per_frame(cls, blip2_answers: List[str], blip2_words: List[str]):
        frame_embedding = None
        if cls.unify_words:
            blip2_words = set(blip2_words)
        for blip2_word in blip2_words:
            try:
                word_weight = cls.train_blip2_answer_word_label_mapping[blip2_word]
                word_embedding = cls.word2vec_embeddings[blip2_word]
            except Exception as e:
                print(e)
                continue

            if frame_embedding is None:
                frame_embedding = word_weight * word_embedding
            else:
                frame_embedding += word_weight * word_embedding
        return frame_embedding
