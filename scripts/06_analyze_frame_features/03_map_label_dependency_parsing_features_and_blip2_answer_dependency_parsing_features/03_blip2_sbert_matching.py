import os
import json
import pickle
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

from sentence_transformers import SentenceTransformer

from typing import List, Tuple

import constants

sbert = SentenceTransformer(
    "sentence-transformers/paraphrase-MiniLM-L6-v2", device="cuda"
)


def calculate_sbert_cosine_similarities(
    blip2_sbert_embeddings: List[np.array],
    label_sbert_embeddings: List[np.array],
    similarity_type: int,
):
    cosine_similarities = []
    for blip2_sbert_embedding in blip2_sbert_embeddings:
        for label_sbert_embedding in label_sbert_embeddings:
            cosine_similarity = (
                np.dot(blip2_sbert_embedding, label_sbert_embedding)
                / (
                    np.linalg.norm(blip2_sbert_embedding)
                    * np.linalg.norm(label_sbert_embedding)
                )
                + 1.0
            ) / 2.0
            cosine_similarities.append((similarity_type, cosine_similarity))
    return cosine_similarities


def get_label_index_dependency_parsing_feature_sbert_embeddings_mapping(
    label_verb_noun_tool_mapping_file_path: str,
):
    with open(label_verb_noun_tool_mapping_file_path, "r") as reader:
        label_verb_noun_tool_mapping = json.load(reader)

    distinct_ground_truth_labels = sorted(list(label_verb_noun_tool_mapping.keys()))

    label_index_dependency_parsing_feature_sbert_embeddings_mapping = dict()

    for label, label_verb_noun_tools in label_verb_noun_tool_mapping.items():
        label_index = distinct_ground_truth_labels.index(label)
        label_index_dependency_parsing_feature_sbert_embeddings_mapping[
            label_index
        ] = dict()
        label_index_dependency_parsing_feature_sbert_embeddings_mapping[label_index][
            "verb"
        ] = []
        label_index_dependency_parsing_feature_sbert_embeddings_mapping[label_index][
            "noun"
        ] = []
        label_index_dependency_parsing_feature_sbert_embeddings_mapping[label_index][
            "verb_noun"
        ] = []
        label_index_dependency_parsing_feature_sbert_embeddings_mapping[label_index][
            "verb_tool"
        ] = []
        label_index_dependency_parsing_feature_sbert_embeddings_mapping[label_index][
            "verb_noun_tool"
        ] = []
        for label_verb_noun_tool in label_verb_noun_tools:
            label_verb = label_verb_noun_tool[0]
            label_noun = label_verb_noun_tool[1]
            label_tool = label_verb_noun_tool[2]
            if label_noun == "NaN":
                # Both noun and tool are NaN, so only encode verb.
                if label_tool == "NaN":
                    sbert_embeddings = sbert.encode([label_verb])
                    label_index_dependency_parsing_feature_sbert_embeddings_mapping[
                        label_index
                    ]["verb"].append(sbert_embeddings[0])
                # Only noun is NaN. So we encode both verb, and verb_using_tool.
                else:
                    sbert_embeddings = sbert.encode(
                        [label_verb, f"{label_verb}_using_{label_tool}"]
                    )

                    label_index_dependency_parsing_feature_sbert_embeddings_mapping[
                        label_index
                    ]["verb"].append(sbert_embeddings[0])
                    label_index_dependency_parsing_feature_sbert_embeddings_mapping[
                        label_index
                    ]["verb_tool"].append(sbert_embeddings[1])
            else:
                # Noun is not NaN. Tool is NaN. Then we encode (verb, noun, verb_noun)
                if label_tool == "NaN":
                    sbert_embeddings = sbert.encode(
                        [label_verb, label_noun, f"{label_verb}_{label_noun}"]
                    )

                    label_index_dependency_parsing_feature_sbert_embeddings_mapping[
                        label_index
                    ]["verb"].append(sbert_embeddings[0])
                    label_index_dependency_parsing_feature_sbert_embeddings_mapping[
                        label_index
                    ]["noun"].append(sbert_embeddings[1])
                    label_index_dependency_parsing_feature_sbert_embeddings_mapping[
                        label_index
                    ]["verb_noun"].append(sbert_embeddings[2])
                # Noun is not NaN, Tool is not NaN. Then we encode (verb, noun, verb_noun, verb_tool, verb_noun_tool)
                else:
                    sbert_embeddings = sbert.encode(
                        [
                            label_verb,
                            label_noun,
                            f"{label_verb}_{label_noun}",
                            f"{label_verb}_using_{label_tool}",
                            f"{label_verb}_{label_noun}_using_{label_tool}",
                        ]
                    )
                    label_index_dependency_parsing_feature_sbert_embeddings_mapping[
                        label_index
                    ]["verb"].append(sbert_embeddings[0])
                    label_index_dependency_parsing_feature_sbert_embeddings_mapping[
                        label_index
                    ]["noun"].append(sbert_embeddings[1])
                    label_index_dependency_parsing_feature_sbert_embeddings_mapping[
                        label_index
                    ]["verb_noun"].append(sbert_embeddings[2])
                    label_index_dependency_parsing_feature_sbert_embeddings_mapping[
                        label_index
                    ]["verb_tool"].append(sbert_embeddings[3])
                    label_index_dependency_parsing_feature_sbert_embeddings_mapping[
                        label_index
                    ]["verb_noun_tool"].append(sbert_embeddings[4])
    return label_index_dependency_parsing_feature_sbert_embeddings_mapping


def get_blip2_dependency_parsing_feature_sbert_embeddings_mapping(
    blip2_answer: str, five_best_dependency_parsing_features: List[Tuple[str, str]]
):
    blip2_dependency_parsing_feature_sbert_embeddings_mapping = {
        "answer": [],
        "verb_noun_tool": [],
        "verb_noun": [],
        "verb_tool": [],
        "verb": [],
        "noun": [],
    }

    sentences_to_encode = [blip2_answer] + [
        dependency_parsing_feature
        for _, dependency_parsing_feature in five_best_dependency_parsing_features
    ]
    dependency_parsing_feature_types = ["answer"] + [
        dependency_parsing_feature_type
        for dependency_parsing_feature_type, _ in five_best_dependency_parsing_features
    ]
    sbert_embeddings = sbert.encode(sentences_to_encode)

    for dependency_parsing_feature_type, sbert_embedding in zip(
        dependency_parsing_feature_types, sbert_embeddings
    ):
        blip2_dependency_parsing_feature_sbert_embeddings_mapping[
            dependency_parsing_feature_type
        ].append(sbert_embedding)

    return blip2_dependency_parsing_feature_sbert_embeddings_mapping


def get_five_best_dependency_parsing_features(
    blip2_verb_noun_tools: List[Tuple[str, str, str]]
):
    dependency_parsing_features_and_scores = []
    for blip2_verb_noun_tool in blip2_verb_noun_tools:
        blip2_verb = blip2_verb_noun_tool[0]
        blip2_noun = blip2_verb_noun_tool[1]
        blip2_tool = blip2_verb_noun_tool[2]
        if blip2_verb != "NaN" and blip2_noun != "NaN" and blip2_tool != "NaN":
            dependency_parsing_feature = f"{blip2_verb} {blip2_noun} using {blip2_tool}"
            score = 5
            dependency_parsing_features_and_scores.append(
                ("verb_noun_tool", dependency_parsing_feature, score)
            )
        elif blip2_verb != "NaN" and blip2_noun != "NaN" and blip2_tool == "NaN":
            dependency_parsing_feature = f"{blip2_verb} {blip2_noun}"
            score = 4
            dependency_parsing_features_and_scores.append(
                ("verb_noun", dependency_parsing_feature, score)
            )
        elif blip2_verb != "NaN" and blip2_noun == "NaN" and blip2_tool != "NaN":
            dependency_parsing_feature = f"{blip2_verb} using {blip2_tool}"
            score = 3
            dependency_parsing_features_and_scores.append(
                ("verb_tool", dependency_parsing_feature, score)
            )
        elif blip2_verb != "NaN" and blip2_noun == "NaN" and blip2_tool == "NaN":
            dependency_parsing_feature = f"{blip2_verb}"
            score = 2
            dependency_parsing_features_and_scores.append(
                ("verb", dependency_parsing_feature, score)
            )
        elif blip2_verb == "NaN" and blip2_noun != "NaN" and blip2_tool == "NaN":
            dependency_parsing_feature = f"{blip2_noun}"
            score = 1
            dependency_parsing_features_and_scores.append(
                ("noun", dependency_parsing_feature, score)
            )
    five_best_dependency_parsing_features_and_scores = sorted(
        dependency_parsing_features_and_scores, key=lambda x: -x[-1]
    )[:5]
    five_best_dependency_parsing_features = [
        (dependency_parsing_feature_type, dependency_parsing_feature)
        for dependency_parsing_feature_type, dependency_parsing_feature, _ in five_best_dependency_parsing_features_and_scores
    ]
    return five_best_dependency_parsing_features


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--label_verb_noun_tool_mapping_file_path",
        type=str,
        default=os.path.join(
            os.environ["CODE"],
            "scripts/06_analyze_frame_features/03_map_label_dependency_parsing_features_and_blip2_answer_dependency_parsing_features",
            "label_verb_noun_tool_mapping.json",
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
        "--output_folder_path",
        type=str,
        default=os.path.join(
            os.environ["SCRATCH"],
            "ego4d_data/v2/analysis_data",
            "blip2_sbert_matching_predictions",
        ),
    )
    args = parser.parse_args()

    with open(
        args.input_file_path,
        "rb",
    ) as reader:
        clip_id_frame_id_verb_noun_tool_pairs_mapping = pickle.load(reader)

    os.makedirs(args.output_folder_path, exist_ok=True)

    label_index_dependency_parsing_feature_sbert_embeddings_mapping = get_label_index_dependency_parsing_feature_sbert_embeddings_mapping(
        label_verb_noun_tool_mapping_file_path=args.label_verb_noun_tool_mapping_file_path
    )

    number_of_clips = len(clip_id_frame_id_verb_noun_tool_pairs_mapping.keys())
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
    current_clip_id_frame_id_verb_noun_tool_pairs_list = sorted(
        list(clip_id_frame_id_verb_noun_tool_pairs_mapping.items()), key=lambda x: x[0]
    )[start_index:end_index]

    current_clip_id_frame_id_predicted_label_indices_and_scores_dictionary_matching = (
        dict()
    )
    for clip_id, frame_id_verb_noun_tool_pairs_mapping in tqdm(
        current_clip_id_frame_id_verb_noun_tool_pairs_list
    ):
        if os.path.exists(os.path.join(args.output_folder_path, clip_id + ".pickle")):
            continue

        current_clip_id_frame_id_predicted_label_indices_and_scores_dictionary_matching[
            clip_id
        ] = dict()
        for (
            frame_id,
            blip2_question_answer_verb_noun_tool_pairs_mapping,
        ) in frame_id_verb_noun_tool_pairs_mapping.items():
            current_clip_id_frame_id_predicted_label_indices_and_scores_dictionary_matching[
                clip_id
            ][
                frame_id
            ] = dict()
            for (
                blip2_question_index,
                blip2_answer_verb_noun_tool_pairs,
            ) in blip2_question_answer_verb_noun_tool_pairs_mapping.items():
                (
                    blip2_answer,
                    blip2_verb_noun_tools,
                ) = blip2_answer_verb_noun_tool_pairs

                if pd.isnull(blip2_answer):
                    current_clip_id_frame_id_predicted_label_indices_and_scores_dictionary_matching[
                        clip_id
                    ][
                        frame_id
                    ][
                        blip2_question_index
                    ] = dict()
                    continue
                else:
                    five_best_dependency_parsing_features = (
                        get_five_best_dependency_parsing_features(
                            blip2_verb_noun_tools=blip2_verb_noun_tools
                        )
                    )
                    blip2_dependency_parsing_feature_sbert_embeddings_mapping = get_blip2_dependency_parsing_feature_sbert_embeddings_mapping(
                        blip2_answer=blip2_answer,
                        five_best_dependency_parsing_features=five_best_dependency_parsing_features,
                    )

                    current_clip_id_frame_id_predicted_label_indices_and_scores_dictionary_matching[
                        clip_id
                    ][
                        frame_id
                    ][
                        blip2_question_index
                    ] = dict()

                    for (
                        label_index,
                        label_dependency_parsing_feature_sbert_embeddings_mapping,
                    ) in (
                        label_index_dependency_parsing_feature_sbert_embeddings_mapping.items()
                    ):
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
                            blip2_dependency_parsing_feature,
                            blip2_sbert_embeddings,
                        ) in (
                            blip2_dependency_parsing_feature_sbert_embeddings_mapping.items()
                        ):
                            for (
                                label_dependency_parsing_feature,
                                label_sbert_embeddings,
                            ) in (
                                label_dependency_parsing_feature_sbert_embeddings_mapping.items()
                            ):
                                if (
                                    blip2_dependency_parsing_feature == "answer"
                                    and (
                                        label_dependency_parsing_feature
                                        == "verb_noun_tool"
                                        or label_dependency_parsing_feature
                                        == "verb_noun"
                                        or label_dependency_parsing_feature
                                        == "verb_tool"
                                        or label_dependency_parsing_feature == "verb"
                                        or label_dependency_parsing_feature == "noun"
                                    )
                                ) or (
                                    blip2_dependency_parsing_feature
                                    == label_dependency_parsing_feature
                                ):
                                    pass
                                else:
                                    continue

                                blip2_dependency_parsing_feature_label_dependency_parsing_feature = f"blip2_{blip2_dependency_parsing_feature}_label_{label_dependency_parsing_feature}"
                                cosine_similarities = calculate_sbert_cosine_similarities(
                                    blip2_sbert_embeddings=blip2_sbert_embeddings,
                                    label_sbert_embeddings=label_sbert_embeddings,
                                    similarity_type=constants.blip2_dependency_parsing_feature_label_dependency_parsing_feature_mapping[
                                        blip2_dependency_parsing_feature_label_dependency_parsing_feature
                                    ],
                                )
                                current_clip_id_frame_id_predicted_label_indices_and_scores_dictionary_matching[
                                    clip_id
                                ][
                                    frame_id
                                ][
                                    blip2_question_index
                                ][
                                    label_index
                                ].extend(
                                    cosine_similarities
                                )
        with open(
            os.path.join(args.output_folder_path, clip_id + ".pickle"), "wb"
        ) as writer:
            pickle.dump(
                current_clip_id_frame_id_predicted_label_indices_and_scores_dictionary_matching,
                writer,
            )
        current_clip_id_frame_id_predicted_label_indices_and_scores_dictionary_matching = (
            dict()
        )
