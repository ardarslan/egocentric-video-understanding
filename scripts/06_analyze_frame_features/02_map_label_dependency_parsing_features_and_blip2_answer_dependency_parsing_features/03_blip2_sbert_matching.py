import os
import json
import pickle
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

from sentence_transformers import SentenceTransformer

from typing import List


def calculate_sbert_cosine_similarity(
    blip2_sbert_embedding: List[np.array], label_sbert_embedding: List[np.array]
):
    cosine_similarity = (
        np.dot(blip2_sbert_embedding, label_sbert_embedding)
        / (
            np.linalg.norm(blip2_sbert_embedding)
            * np.linalg.norm(label_sbert_embedding)
        )
        + 1.0
    ) / 2.0
    return cosine_similarity


def get_label_index_label_phrase_sbert_embeddings_mapping(
    label_phrase_mapping_file_path: str, model: SentenceTransformer
):
    with open(label_phrase_mapping_file_path, "r") as reader:
        label_phrase_mapping = json.load(reader)

    distinct_ground_truth_labels = sorted(list(label_phrase_mapping.keys()))

    label_index_label_phrase_sbert_embeddings_mapping = dict()

    for label, label_phrases in label_phrase_mapping.items():
        label_index = distinct_ground_truth_labels.index(label)
        label_index_label_phrase_sbert_embeddings_mapping[label_index] = []
        for label_phrase in label_phrases:
            label_index_label_phrase_sbert_embeddings_mapping[label_index].append(
                model.encode([label_phrase])[0]
            )

    return label_index_label_phrase_sbert_embeddings_mapping


def get_blip2_answer_sbert_embedding(blip2_answer: str, model: SentenceTransformer):
    sbert_embedding = model.encode([blip2_answer])[0]
    return sbert_embedding


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backbone",
        type=str,
        choices=[
            "sentence-transformers/paraphrase-MiniLM-L6-v2",
            "sentence-transformers/all-distilroberta-v1",
        ],
        default="sentence-transformers/all-distilroberta-v1",
    )
    parser.add_argument(
        "--label_phrase_mapping_file_path",
        type=str,
        default=os.path.join(
            os.environ["CODE"],
            "scripts/06_analyze_frame_features/02_map_label_dependency_parsing_features_and_blip2_answer_dependency_parsing_features",
            "label_phrase_mapping.json",
        ),
    )
    parser.add_argument(
        "--quarter_index", type=int, choices=[0, 1, 2, 3], required=True
    )
    parser.add_argument(
        "--input_file_path",
        type=str,
        default=os.path.join(
            os.environ["SCRATCH"],
            "ego4d_data/v2/analysis_data/dependency_parsing_results",
            "dependency_parsing_results.pickle",
        ),
    )
    parser.add_argument(
        "--output_parent_folder_path",
        type=str,
        default=os.path.join(
            os.environ["SCRATCH"],
            "ego4d_data/v2/analysis_data",
        ),
    )
    args = parser.parse_args()

    model = SentenceTransformer(args.backbone, device="cuda")

    with open(
        args.input_file_path,
        "rb",
    ) as reader:
        clip_id_frame_id_blip2_answers_mapping = pickle.load(reader)

    output_folder_path = (
        args.output_parent_folder_path
        + "/"
        + "blip2_sbert_matching_"
        + args.backbone.split("/")[-1]
        + "_predictions"
    )
    os.makedirs(output_folder_path, exist_ok=True)

    label_index_label_phrase_sbert_embeddings_mapping = (
        get_label_index_label_phrase_sbert_embeddings_mapping(
            label_phrase_mapping_file_path=args.label_phrase_mapping_file_path,
            model=model,
        )
    )

    number_of_clips = len(clip_id_frame_id_blip2_answers_mapping.keys())
    if args.quarter_index == 0:
        start_index = 0
        end_index = number_of_clips // 4
    elif args.quarter_index == 1:
        start_index = number_of_clips // 4
        end_index = 2 * (number_of_clips // 4)
    elif args.quarter_index == 2:
        start_index = 2 * (number_of_clips // 4)
        end_index = 3 * (number_of_clips // 4)
    elif args.quarter_index == 3:
        start_index = 3 * (number_of_clips // 4)
        end_index = number_of_clips
    current_clip_id_frame_id_blip2_answers_list = sorted(
        list(clip_id_frame_id_blip2_answers_mapping.items()), key=lambda x: x[0]
    )[start_index:end_index]

    for clip_id, frame_id_blip2_answers_mapping in tqdm(
        current_clip_id_frame_id_blip2_answers_list
    ):
        current_clip_id_frame_id_predicted_label_indices_and_scores_dictionary_matching = {
            clip_id: dict()
        }
        for frame_id, blip2_answers_mapping in frame_id_blip2_answers_mapping.items():
            current_clip_id_frame_id_predicted_label_indices_and_scores_dictionary_matching[
                clip_id
            ][
                frame_id
            ] = dict()
            for (
                blip2_question_index,
                blip2_answer_verb_noun_tool_pairs,
            ) in blip2_answers_mapping.items():
                current_clip_id_frame_id_predicted_label_indices_and_scores_dictionary_matching[
                    clip_id
                ][
                    frame_id
                ][
                    blip2_question_index
                ] = dict()
                blip2_answer = blip2_answer_verb_noun_tool_pairs[0]
                if pd.isnull(blip2_answer):
                    for (
                        label_index
                    ) in label_index_label_phrase_sbert_embeddings_mapping.keys():
                        current_clip_id_frame_id_predicted_label_indices_and_scores_dictionary_matching[
                            clip_id
                        ][
                            frame_id
                        ][
                            blip2_question_index
                        ][
                            label_index
                        ] = []
                else:
                    blip2_answer_sbert_embedding = get_blip2_answer_sbert_embedding(
                        blip2_answer=blip2_answer, model=model
                    )
                    for (
                        label_index
                    ) in label_index_label_phrase_sbert_embeddings_mapping.keys():
                        label_phrase_sbert_embeddings = (
                            label_index_label_phrase_sbert_embeddings_mapping[
                                label_index
                            ]
                        )
                        current_clip_id_frame_id_predicted_label_indices_and_scores_dictionary_matching[
                            clip_id
                        ][
                            frame_id
                        ][
                            blip2_question_index
                        ][
                            label_index
                        ] = []
                        for (
                            label_phrase_sbert_embedding
                        ) in label_phrase_sbert_embeddings:
                            current_clip_id_frame_id_predicted_label_indices_and_scores_dictionary_matching[
                                clip_id
                            ][
                                frame_id
                            ][
                                blip2_question_index
                            ][
                                label_index
                            ].append(
                                calculate_sbert_cosine_similarity(
                                    blip2_sbert_embedding=blip2_answer_sbert_embedding,
                                    label_sbert_embedding=label_phrase_sbert_embedding,
                                )
                            )

        with open(
            os.path.join(output_folder_path, clip_id + ".pickle"), "wb"
        ) as writer:
            pickle.dump(
                current_clip_id_frame_id_predicted_label_indices_and_scores_dictionary_matching,
                writer,
            )
