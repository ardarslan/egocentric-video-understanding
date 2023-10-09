import os
import json
import pickle
import argparse
import numpy as np
from tqdm import tqdm

from sentence_transformers import SentenceTransformer

from typing import Dict, List, Tuple

import constants

sbert = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L6-v2")


def calculate_sbert_cosine_similarities(
    blip2_sbert_embeddings: List[np.array],
    label_sbert_embeddings: List[np.array],
    similarity_type: int,
):
    cosine_similarities = []
    for blip2_sbert_embedding in blip2_sbert_embeddings:
        for label_sbert_embedding in label_sbert_embeddings:
            cosine_similarity = (np.dot(blip2_sbert_embedding, label_sbert_embedding) / (np.linalg.norm(blip2_sbert_embedding) * np.linalg.norm(label_sbert_embedding)) + 1.0) / 2.0
            cosine_similarities.append((similarity_type, cosine_similarity))
    return cosine_similarities


def nontemporal_sbert_embedding_for_given_clip(
    clip_id: str,
    frame_id_blip2_question_answer_verb_noun_tool_pairs_mapping: Dict[str, List[Tuple[str, str, str]]],
    label_verb_noun_tool_mapping: Dict[str, List[Tuple[str, str, str]]],
):
    label_verb_noun_tool_sbert_embeddings_mapping = dict()
    label_verb_noun_sbert_embeddings_mapping = dict()
    label_verb_tool_sbert_embeddings_mapping = dict()
    label_verb_sbert_embeddings_mapping = dict()

    distinct_ground_truth_labels = list(label_verb_noun_tool_mapping.keys())

    for label, label_verb_noun_tools in label_verb_noun_tool_mapping.items():
        label_index = distinct_ground_truth_labels.index(label)
        if label_index not in label_verb_noun_tool_sbert_embeddings_mapping.keys():
            label_verb_noun_tool_sbert_embeddings_mapping[label_index] = []
        if label_index not in label_verb_noun_sbert_embeddings_mapping.keys():
            label_verb_noun_sbert_embeddings_mapping[label_index] = []
        if label_index not in label_verb_tool_sbert_embeddings_mapping.keys():
            label_verb_tool_sbert_embeddings_mapping[label_index] = []
        if label_index not in label_verb_sbert_embeddings_mapping.keys():
            label_verb_sbert_embeddings_mapping[label_index] = []

        for label_verb_noun_tool in label_verb_noun_tools:
            label_verb = label_verb_noun_tool[0]
            label_noun = label_verb_noun_tool[1]
            label_tool = label_verb_noun_tool[2]
            if label_noun == "NaN":
                label_noun = "something"
            if label_tool == "NaN":
                label_tool = "a tool"

            label_sbert_embeddings = sbert.encode(
                [
                    f"{label_verb} {label_noun} using {label_tool}",
                    f"{label_verb} {label_noun}",
                    f"{label_verb} using {label_tool}",
                    f"{label_verb}",
                ]
            )

            label_verb_noun_tool_sbert_embeddings_mapping[label_index].append(label_sbert_embeddings[0])
            label_verb_noun_sbert_embeddings_mapping[label_index].append(label_sbert_embeddings[1])
            label_verb_tool_sbert_embeddings_mapping[label_index].append(label_sbert_embeddings[2])
            label_verb_sbert_embeddings_mapping[label_index].append(label_sbert_embeddings[3])

    frame_id_blip2_answer_sbert_embeddings_mapping = dict()
    frame_id_blip2_verb_noun_tool_sbert_embeddings_mapping = dict()
    frame_id_blip2_verb_noun_sbert_embeddings_mapping = dict()
    frame_id_blip2_verb_tool_sbert_embeddings_mapping = dict()
    frame_id_blip2_verb_sbert_embeddings_mapping = dict()

    for (
        frame_id,
        blip2_question_answer_verb_noun_tool_pairs_mapping,
    ) in frame_id_blip2_question_answer_verb_noun_tool_pairs_mapping.items():
        frame_id_blip2_answer_sbert_embeddings_mapping[frame_id] = []
        frame_id_blip2_verb_noun_tool_sbert_embeddings_mapping[frame_id] = []
        frame_id_blip2_verb_noun_sbert_embeddings_mapping[frame_id] = []
        frame_id_blip2_verb_tool_sbert_embeddings_mapping[frame_id] = []
        frame_id_blip2_verb_sbert_embeddings_mapping[frame_id] = []
        for (
            _,
            blip2_answer_verb_noun_tool_pairs,
        ) in blip2_question_answer_verb_noun_tool_pairs_mapping.items():
            for index, blip2_answer_verb_noun_tool_pair in enumerate(blip2_answer_verb_noun_tool_pairs[1]):
                blip2_verb = blip2_answer_verb_noun_tool_pair[0]
                blip2_noun = blip2_answer_verb_noun_tool_pair[1]
                blip2_tool = blip2_answer_verb_noun_tool_pair[2]
                if blip2_noun == "NaN":
                    blip2_noun = "something"
                if blip2_tool == "NaN":
                    blip2_tool = "a tool"

                if index == 0:
                    blip2_answer = blip2_answer_verb_noun_tool_pairs[0]
                    blip2_sbert_embeddings = sbert.encode(
                        [
                            f"{blip2_verb} {blip2_noun} using {blip2_tool}",
                            f"{blip2_verb} {blip2_noun}",
                            f"{blip2_verb} using {blip2_tool}",
                            f"{blip2_verb}",
                            f"{blip2_answer}",
                        ]
                    )
                    frame_id_blip2_verb_noun_tool_sbert_embeddings_mapping[frame_id].append(blip2_sbert_embeddings[0])
                    frame_id_blip2_verb_noun_sbert_embeddings_mapping[frame_id].append(blip2_sbert_embeddings[1])
                    frame_id_blip2_verb_tool_sbert_embeddings_mapping[frame_id].append(blip2_sbert_embeddings[2])
                    frame_id_blip2_verb_sbert_embeddings_mapping[frame_id].append(blip2_sbert_embeddings[3])
                    frame_id_blip2_answer_sbert_embeddings_mapping[frame_id].append(blip2_sbert_embeddings[4])
                else:
                    blip2_sbert_embeddings = sbert.encode(
                        [
                            f"{blip2_verb} {blip2_noun} using {blip2_tool}",
                            f"{blip2_verb} {blip2_noun}",
                            f"{blip2_verb} using {blip2_tool}",
                            f"{blip2_verb}",
                        ]
                    )
                    frame_id_blip2_verb_noun_tool_sbert_embeddings_mapping[frame_id].append(blip2_sbert_embeddings[0])
                    frame_id_blip2_verb_noun_sbert_embeddings_mapping[frame_id].append(blip2_sbert_embeddings[1])
                    frame_id_blip2_verb_tool_sbert_embeddings_mapping[frame_id].append(blip2_sbert_embeddings[2])
                    frame_id_blip2_verb_sbert_embeddings_mapping[frame_id].append(blip2_sbert_embeddings[3])

    frame_id_predicted_label_indices_and_scores = dict()
    for frame_id in frame_id_blip2_question_answer_verb_noun_tool_pairs_mapping.keys():
        frame_id_predicted_label_indices_and_scores[frame_id] = dict()

        blip2_answer_sbert_embeddings = frame_id_blip2_answer_sbert_embeddings_mapping[frame_id]
        blip2_verb_noun_tool_sbert_embeddings = frame_id_blip2_verb_noun_tool_sbert_embeddings_mapping[frame_id]
        blip2_verb_noun_sbert_embeddings = frame_id_blip2_verb_noun_sbert_embeddings_mapping[frame_id]
        blip2_verb_tool_sbert_embeddings = frame_id_blip2_verb_tool_sbert_embeddings_mapping[frame_id]
        blip2_verb_sbert_embeddings = frame_id_blip2_verb_sbert_embeddings_mapping[frame_id]

        for label_index in range(len(distinct_ground_truth_labels) + 1):
            if label_index == len(distinct_ground_truth_labels):
                for (
                    _,
                    blip2_answer_verb_noun_tool_pairs,
                ) in blip2_question_answer_verb_noun_tool_pairs_mapping.items():
                    blip2_verb_noun_tool_pairs = blip2_answer_verb_noun_tool_pairs[1]
                    if len(blip2_verb_noun_tool_pairs) == 0:
                        frame_id_predicted_label_indices_and_scores[frame_id][label_index] = 1.0
                    else:
                        frame_id_predicted_label_indices_and_scores[frame_id][label_index] = 0.0
            else:
                frame_id_predicted_label_indices_and_scores[frame_id][label_index] = []

                label_verb_noun_tool_sbert_embeddings = label_verb_noun_tool_sbert_embeddings_mapping[label_index]
                label_verb_noun_sbert_embeddings = label_verb_noun_sbert_embeddings_mapping[label_index]
                label_verb_tool_sbert_embeddings = label_verb_tool_sbert_embeddings_mapping[label_index]
                label_verb_sbert_embeddings = label_verb_sbert_embeddings_mapping[label_index]

                frame_id_predicted_label_indices_and_scores[frame_id][label_index].extend(
                    calculate_sbert_cosine_similarities(
                        blip2_sbert_embeddings=blip2_answer_sbert_embeddings,
                        label_sbert_embeddings=label_verb_noun_tool_sbert_embeddings,
                        similarity_type=constants.BLIP2_ANSWER_LABEL_VERB_NOUN_TOOL,
                    )
                )

                frame_id_predicted_label_indices_and_scores[frame_id][label_index].extend(
                    calculate_sbert_cosine_similarities(
                        blip2_sbert_embeddings=blip2_answer_sbert_embeddings,
                        label_sbert_embeddings=label_verb_noun_sbert_embeddings,
                        similarity_type=constants.BLIP2_ANSWER_LABEL_VERB_NOUN,
                    )
                )

                frame_id_predicted_label_indices_and_scores[frame_id][label_index].extend(
                    calculate_sbert_cosine_similarities(
                        blip2_sbert_embeddings=blip2_answer_sbert_embeddings,
                        label_sbert_embeddings=label_verb_tool_sbert_embeddings,
                        similarity_type=constants.BLIP2_ANSWER_LABEL_VERB_TOOL,
                    )
                )

                frame_id_predicted_label_indices_and_scores[frame_id][label_index].extend(
                    calculate_sbert_cosine_similarities(
                        blip2_sbert_embeddings=blip2_answer_sbert_embeddings,
                        label_sbert_embeddings=label_verb_sbert_embeddings,
                        similarity_type=constants.BLIP2_ANSWER_LABEL_VERB,
                    )
                )

                frame_id_predicted_label_indices_and_scores[frame_id][label_index].extend(
                    calculate_sbert_cosine_similarities(
                        blip2_sbert_embeddings=blip2_verb_noun_tool_sbert_embeddings,
                        label_sbert_embeddings=label_verb_noun_tool_sbert_embeddings,
                        similarity_type=constants.BLIP2_VERB_NOUN_TOOL_LABEL_VERB_NOUN_TOOL,
                    )
                )

                frame_id_predicted_label_indices_and_scores[frame_id][label_index].extend(
                    calculate_sbert_cosine_similarities(
                        blip2_sbert_embeddings=blip2_verb_noun_sbert_embeddings,
                        label_sbert_embeddings=label_verb_noun_sbert_embeddings,
                        similarity_type=constants.BLIP2_VERB_NOUN_LABEL_VERB_NOUN,
                    )
                )

                frame_id_predicted_label_indices_and_scores[frame_id][label_index].extend(
                    calculate_sbert_cosine_similarities(
                        blip2_sbert_embeddings=blip2_verb_tool_sbert_embeddings,
                        label_sbert_embeddings=label_verb_tool_sbert_embeddings,
                        similarity_type=constants.BLIP2_VERB_TOOL_LABEL_VERB_TOOL,
                    )
                )

                frame_id_predicted_label_indices_and_scores[frame_id][label_index].extend(
                    calculate_sbert_cosine_similarities(
                        blip2_sbert_embeddings=blip2_verb_sbert_embeddings,
                        label_sbert_embeddings=label_verb_sbert_embeddings,
                        similarity_type=constants.BLIP2_VERB_LABEL_VERB,
                    )
                )

        # Normalize scores per frame so that their sum is equal to 1.0.
        # sum_scores = 0.0
        # for label_index in range(len(distinct_ground_truth_labels)):
        #     sum_scores += frame_id_predicted_label_indices_and_scores[frame_id][
        #         label_index
        #     ]
        # for label_index in range(len(distinct_ground_truth_labels)):
        #     frame_id_predicted_label_indices_and_scores[frame_id][
        #         label_index
        #     ] = frame_id_predicted_label_indices_and_scores[frame_id][
        #         label_index
        #     ] / float(
        #         sum_scores
        #     )

    return clip_id, frame_id_predicted_label_indices_and_scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--label_verb_noun_tool_mapping_file_path",
        type=str,
        default=os.path.join(
            os.environ["CODE"],
            "scripts/06_analyze_frame_features/03_map_label_dependency_parsing_features_and_blip2_answer_dependency_parsing_features_nontemporal",
            "label_verb_noun_tool_mapping.json",
        ),
    )
    parser.add_argument(
        "--clip_id_frame_id_verb_noun_tool_pairs_mapping_file_path",
        type=str,
        default=os.path.join(
            os.environ["SCRATCH"],
            "ego4d_data/v2/analysis_data",
            "clip_id_frame_id_verb_noun_tool_pairs_mapping.pickle",
        ),
    )
    parser.add_argument(
        "--output_path_name_wo_ext",
        type=str,
        default=os.path.join(
            os.environ["SCRATCH"],
            "ego4d_data/v2/analysis_data",
            "clip_id_frame_id_predicted_label_indices_and_scores_nontemporal_sbert_embedding",
        ),
    )
    args = parser.parse_args()

    with open(args.label_verb_noun_tool_mapping_file_path, "r") as reader:
        label_verb_noun_tool_mapping = json.load(reader)

    with open(
        args.clip_id_frame_id_verb_noun_tool_pairs_mapping_file_path,
        "rb",
    ) as reader:
        clip_id_frame_id_verb_noun_tool_pairs_mapping = pickle.load(reader)

    current_clip_id_frame_id_predicted_label_indices_and_scores_nontemporal_sbert_embedding = dict()
    clip_counter = 0
    file_name_counter = 0
    for clip_id, frame_id_verb_noun_tool_pairs_mapping in clip_id_frame_id_verb_noun_tool_pairs_mapping.items():
        current_clip_id_frame_id_predicted_label_indices_and_scores_nontemporal_sbert_embedding[clip_id] = nontemporal_sbert_embedding_for_given_clip(
            clip_id=clip_id, frame_id_blip2_question_answer_verb_noun_tool_pairs_mapping=frame_id_verb_noun_tool_pairs_mapping, label_verb_noun_tool_mapping=label_verb_noun_tool_mapping
        )
        clip_counter += 1
        if clip_counter % 100 == 0:
            with open(args.output_path_name_wo_ext + "_" + str(file_name_counter).zfill(3) + ".pickle", "wb") as writer:
                pickle.dump(current_clip_id_frame_id_predicted_label_indices_and_scores_nontemporal_sbert_embedding, writer)
            current_clip_id_frame_id_predicted_label_indices_and_scores_nontemporal_sbert_embedding = dict()
            file_name_counter += 1
