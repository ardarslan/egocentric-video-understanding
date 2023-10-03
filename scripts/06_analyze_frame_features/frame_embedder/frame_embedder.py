import spacy

from typing import Dict, List


class FrameEmbedder(object):
    nlp = spacy.load("en_core_web_lg")

    @classmethod
    def __init__(
        cls, train_blip2_answer_word_label_mapping: Dict[str, float], unify_words: bool
    ):
        cls.train_blip2_answer_word_label_mapping = (
            train_blip2_answer_word_label_mapping
        )
        cls.unify_words = unify_words

    @classmethod
    def process_per_frame_blip2_answers(cls, blip2_answers: Dict[str, str]):
        words = []
        docs = [cls.nlp(blip2_answer) for blip2_answer in blip2_answers.values()]
        for doc in docs:
            words.extend(
                [
                    token.lemma_.lower()
                    for token in doc
                    if (token.lemma_.isalpha())
                    and (not token.is_stop)
                    and (token.text != "no_answer")
                ]
            )
        return words

    @classmethod
    def process_per_clip_blip2_answers(
        cls, clip_id: str, frame_id_blip2_answers_mapping: Dict[int, Dict[str, str]]
    ):
        frame_id_words_mapping = {}
        for frame_id, blip2_answers_mapping in frame_id_blip2_answers_mapping.items():
            frame_id_words_mapping[
                frame_id
            ] = FrameEmbedder.process_per_frame_blip2_answers(blip2_answers_mapping)
        return clip_id, frame_id_words_mapping

    @classmethod
    def get_embedding_per_frame(cls, blip2_answers: List[str], blip2_words: List[str]):
        pass

    @classmethod
    def get_embedding_per_clip(
        cls,
        clip_id: str,
        frame_id_blip2_answers_mapping: Dict[int, List[str]],
        frame_id_blip2_words_mapping: Dict[int, List[str]],
    ):
        frame_id_embedding_mapping = {}
        for frame_id, blip2_answers in frame_id_blip2_answers_mapping.items():
            blip2_words = frame_id_blip2_words_mapping[frame_id]
            current_embedding = cls.get_embedding_per_frame(
                blip2_answers=blip2_answers, blip2_words=blip2_words
            )
            frame_id_embedding_mapping[frame_id] = current_embedding
        return clip_id, frame_id_embedding_mapping
