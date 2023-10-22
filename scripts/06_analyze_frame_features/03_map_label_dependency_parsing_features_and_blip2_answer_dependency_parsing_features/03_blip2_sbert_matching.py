import os
import json
import pickle
import argparse
import numpy as np
from tqdm import tqdm

from sentence_transformers import SentenceTransformer

from typing import List, Tuple

import constants

sbert = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L6-v2", device="cuda")


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


def get_label_index_dependency_parsing_feature_sbert_embeddings_mapping(label_verb_noun_tool_mapping_file_path: str):
    with open(label_verb_noun_tool_mapping_file_path, "r") as reader:
        label_verb_noun_tool_mapping = json.load(reader)

    distinct_ground_truth_labels = sorted(list(label_verb_noun_tool_mapping.keys()))

    label_index_dependency_parsing_feature_sbert_embeddings_mapping = dict()

    for label, label_verb_noun_tools in label_verb_noun_tool_mapping.items():
        label_index = distinct_ground_truth_labels.index(label)
        label_index_dependency_parsing_feature_sbert_embeddings_mapping[label_index] = dict()
        label_index_dependency_parsing_feature_sbert_embeddings_mapping[label_index]["verb"] = []
        label_index_dependency_parsing_feature_sbert_embeddings_mapping[label_index]["verb_noun"] = []
        label_index_dependency_parsing_feature_sbert_embeddings_mapping[label_index]["verb_tool"] = []
        label_index_dependency_parsing_feature_sbert_embeddings_mapping[label_index]["verb_noun_tool"] = []
        for label_verb_noun_tool in label_verb_noun_tools:
            label_verb = label_verb_noun_tool[0]
            label_noun = label_verb_noun_tool[1]
            label_tool = label_verb_noun_tool[2]
            if label_noun == "NaN":
                # Both noun and tool are NaN, so only encode verb.
                if label_tool == "NaN":
                    sbert_embeddings = sbert.encode([label_verb])
                    label_index_dependency_parsing_feature_sbert_embeddings_mapping[label_index]["verb"].append(sbert_embeddings[0])
                # Only noun is NaN. So we encode both verb, and verb_using_tool.
                else:
                    sbert_embeddings = sbert.encode([label_verb, f"{label_verb}_using_{label_tool}"])

                    label_index_dependency_parsing_feature_sbert_embeddings_mapping[label_index]["verb"].append(sbert_embeddings[0])
                    label_index_dependency_parsing_feature_sbert_embeddings_mapping[label_index]["verb_tool"].append(sbert_embeddings[1])
            else:
                # Noun is not NaN. Tool is NaN. Then we encode (verb, noun, verb_noun)
                if label_tool == "NaN":
                    sbert_embeddings = sbert.encode([label_verb, f"{label_verb}_{label_noun}"])

                    label_index_dependency_parsing_feature_sbert_embeddings_mapping[label_index]["verb"].append(sbert_embeddings[0])
                    label_index_dependency_parsing_feature_sbert_embeddings_mapping[label_index]["verb_noun"].append(sbert_embeddings[1])
                # Noun is not NaN, Tool is not NaN. Then we encode (verb, noun, verb_noun, verb_tool, verb_noun_tool)
                else:
                    sbert_embeddings = sbert.encode([label_verb, f"{label_verb}_{label_noun}", f"{label_verb}_using_{label_tool}", f"{label_verb}_{label_noun}_using_{label_tool}"])
                    label_index_dependency_parsing_feature_sbert_embeddings_mapping[label_index]["verb"].append(sbert_embeddings[0])
                    label_index_dependency_parsing_feature_sbert_embeddings_mapping[label_index]["verb_noun"].append(sbert_embeddings[1])
                    label_index_dependency_parsing_feature_sbert_embeddings_mapping[label_index]["verb_tool"].append(sbert_embeddings[2])
                    label_index_dependency_parsing_feature_sbert_embeddings_mapping[label_index]["verb_noun_tool"].append(sbert_embeddings[3])
    return label_index_dependency_parsing_feature_sbert_embeddings_mapping


def get_blip2_dependency_parsing_feature_sbert_embeddings_mapping(blip2_answer: str, blip2_verb_noun_tools: List[Tuple[str, str, str]]):
    blip2_dependency_parsing_feature_sbert_embeddings_mapping = dict()
    blip2_dependency_parsing_feature_sbert_embeddings_mapping["answer"] = []
    blip2_dependency_parsing_feature_sbert_embeddings_mapping["verb"] = []
    blip2_dependency_parsing_feature_sbert_embeddings_mapping["verb_noun"] = []
    blip2_dependency_parsing_feature_sbert_embeddings_mapping["verb_tool"] = []
    blip2_dependency_parsing_feature_sbert_embeddings_mapping["verb_noun_tool"] = []
    for blip2_verb_noun_tool in blip2_verb_noun_tools:
        blip2_verb = blip2_verb_noun_tool[0]
        blip2_noun = blip2_verb_noun_tool[1]
        blip2_tool = blip2_verb_noun_tool[2]
        if blip2_noun == "NaN":
            # Both noun and tool are NaN, so only encode verb.
            if blip2_tool == "NaN":
                sbert_embeddings = sbert.encode([blip2_answer, blip2_verb])
                blip2_dependency_parsing_feature_sbert_embeddings_mapping["answer"].append(sbert_embeddings[0])
                blip2_dependency_parsing_feature_sbert_embeddings_mapping["verb"].append(sbert_embeddings[1])
            # Only noun is NaN. So we encode both verb, and verb_using_tool.
            else:
                sbert_embeddings = sbert.encode([blip2_answer, blip2_verb, f"{blip2_verb}_using_{blip2_tool}"])
                blip2_dependency_parsing_feature_sbert_embeddings_mapping["answer"].append(sbert_embeddings[0])
                blip2_dependency_parsing_feature_sbert_embeddings_mapping["verb"].append(sbert_embeddings[1])
                blip2_dependency_parsing_feature_sbert_embeddings_mapping["verb_tool"].append(sbert_embeddings[2])
        else:
            # Noun is not NaN. Tool is NaN. Then we encode (verb, noun, verb_noun)
            if blip2_tool == "NaN":
                sbert_embeddings = sbert.encode([blip2_answer, blip2_verb, blip2_noun, f"{blip2_verb}_{blip2_noun}"])
                blip2_dependency_parsing_feature_sbert_embeddings_mapping["answer"].append(sbert_embeddings[0])
                blip2_dependency_parsing_feature_sbert_embeddings_mapping["verb"].append(sbert_embeddings[1])
                blip2_dependency_parsing_feature_sbert_embeddings_mapping["verb_noun"].append(sbert_embeddings[2])
            # Noun is not NaN, Tool is not NaN. Then we encode (verb, noun, verb_noun, verb_tool, verb_noun_tool)
            else:
                sbert_embeddings = sbert.encode(
                    [blip2_answer, blip2_verb, blip2_noun, f"{blip2_verb}_{blip2_noun}", f"{blip2_verb}_using_{blip2_tool}", f"{blip2_verb}_{blip2_noun}_using_{blip2_tool}"]
                )
                blip2_dependency_parsing_feature_sbert_embeddings_mapping["answer"].append(sbert_embeddings[0])
                blip2_dependency_parsing_feature_sbert_embeddings_mapping["verb"].append(sbert_embeddings[1])
                blip2_dependency_parsing_feature_sbert_embeddings_mapping["verb_noun"].append(sbert_embeddings[2])
                blip2_dependency_parsing_feature_sbert_embeddings_mapping["verb_tool"].append(sbert_embeddings[3])
                blip2_dependency_parsing_feature_sbert_embeddings_mapping["verb_noun_tool"].append(sbert_embeddings[4])
    return blip2_dependency_parsing_feature_sbert_embeddings_mapping


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

    current_clip_id_frame_id_predicted_label_indices_and_scores_dictionary_matching = dict()
    clip_counter = 0
    for clip_id, frame_id_verb_noun_tool_pairs_mapping in tqdm(sorted(list(clip_id_frame_id_verb_noun_tool_pairs_mapping.items()), key=lambda x: x[0])):
        if os.path.exists(os.path.join(args.output_folder_path, str(clip_counter).zfill(5) + ".pickle")):
            continue

        current_clip_id_frame_id_predicted_label_indices_and_scores_dictionary_matching[clip_id] = dict()
        for frame_id, blip2_question_answer_verb_noun_tool_pairs_mapping in frame_id_verb_noun_tool_pairs_mapping.items():
            current_clip_id_frame_id_predicted_label_indices_and_scores_dictionary_matching[clip_id][frame_id] = dict()
            for blip2_question_index, blip2_answer_verb_noun_tool_pairs in blip2_question_answer_verb_noun_tool_pairs_mapping.items():
                current_clip_id_frame_id_predicted_label_indices_and_scores_dictionary_matching[clip_id][frame_id][blip2_question_index] = dict()
                for label_index, label_dependency_parsing_feature_sbert_embeddings_mapping in label_index_dependency_parsing_feature_sbert_embeddings_mapping.items():
                    current_clip_id_frame_id_predicted_label_indices_and_scores_dictionary_matching[clip_id][frame_id][blip2_question_index][label_index] = []
                    for label_dependency_parsing_feature, label_sbert_embeddings in label_dependency_parsing_feature_sbert_embeddings_mapping.items():
                        blip2_answer, blip2_verb_noun_tools = blip2_answer_verb_noun_tool_pairs
                        blip2_dependency_parsing_feature_sbert_embeddings_mapping = get_blip2_dependency_parsing_feature_sbert_embeddings_mapping(
                            blip2_answer=blip2_answer, blip2_verb_noun_tools=blip2_verb_noun_tools
                        )
                        for blip2_dependency_parsing_feature, blip2_sbert_embeddings in blip2_dependency_parsing_feature_sbert_embeddings_mapping.items():
                            if (
                                blip2_dependency_parsing_feature == "answer"
                                and (
                                    label_dependency_parsing_feature == "verb_noun_tool"
                                    or label_dependency_parsing_feature == "verb_noun"
                                    or label_dependency_parsing_feature == "verb_tool"
                                    or label_dependency_parsing_feature == "verb"
                                )
                            ) or (blip2_dependency_parsing_feature == label_dependency_parsing_feature):
                                pass
                            else:
                                continue

                            blip2_dependency_parsing_feature_label_dependency_parsing_feature = f"blip2_{blip2_dependency_parsing_feature}_label_{label_dependency_parsing_feature}"
                            cosine_similarities = calculate_sbert_cosine_similarities(
                                blip2_sbert_embeddings=blip2_sbert_embeddings,
                                label_sbert_embeddings=label_sbert_embeddings,
                                similarity_type=constants.blip2_dependency_parsing_feature_label_dependency_parsing_feature_mapping[blip2_dependency_parsing_feature_label_dependency_parsing_feature],
                            )
                            current_clip_id_frame_id_predicted_label_indices_and_scores_dictionary_matching[clip_id][frame_id][blip2_question_index][label_index].extend(cosine_similarities)
        clip_counter += 1
        with open(os.path.join(args.output_folder_path, str(clip_counter).zfill(5) + ".pickle"), "wb") as writer:
            pickle.dump(current_clip_id_frame_id_predicted_label_indices_and_scores_dictionary_matching, writer)
        current_clip_id_frame_id_predicted_label_indices_and_scores_dictionary_matching = dict()
