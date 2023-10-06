import os
import cv2
import json
import pickle
import numpy as np
from tqdm import tqdm
from pqdm.processes import pqdm
from typing import Dict, List, Tuple
from sklearn.metrics import f1_score, precision_score, recall_score

from sentence_transformers import SentenceTransformer

with open(os.path.join(os.environ["CODE"], "scripts/06_analyze_frame_features/label_verb_noun_tool_mapping.json"), "r") as reader:
    label_verb_noun_tools_mapping = json.load(reader)

with open(os.path.join(os.environ["SCRATCH"], "ego4d_data/v2/analysis_data", "clip_id_frame_id_blip2_verb_noun_tool_pair_mapping.pickle"), "rb") as reader:
    clip_id_frame_id_blip2_verb_noun_tool_pair_mapping = pickle.load(reader)

with open(os.path.join(os.environ["SCRATCH"], "ego4d_data/v2/analysis_data", "analysis_data.pickle"), "rb") as reader:
    clip_id_frame_id_ground_truth_labels_mapping = pickle.load(reader)["clip_id_frame_id_labels_mapping"]

sbert = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2')

distinct_ground_truth_labels = sorted(list(label_verb_noun_tools_mapping.keys()))

clip_id_frame_id_ground_truth_label_indices_mapping = dict()
for clip_id, frame_id_ground_truth_labels_mapping in clip_id_frame_id_ground_truth_labels_mapping.items():
    if list(clip_id_frame_id_ground_truth_labels_mapping[clip_id][0])[0] == "no_annotation":
        continue
    clip_id_frame_id_ground_truth_label_indices_mapping[clip_id] = dict()
    for frame_id, ground_truth_labels in frame_id_ground_truth_labels_mapping.items():
        clip_id_frame_id_ground_truth_label_indices_mapping[clip_id][frame_id] = []
        for ground_truth_label in ground_truth_labels:
            if ground_truth_label == "background": # handle in map_blip2_answers_to_verb_noun_tool_pairs.py so that such a label never comes.
                continue
            clip_id_frame_id_ground_truth_label_indices_mapping[clip_id][frame_id].append(distinct_ground_truth_labels.index(ground_truth_label))
del clip_id_frame_id_ground_truth_labels_mapping


def evaluate_predictions(clip_id_frame_id_predicted_label_indices_mapping: Dict[str, Dict[str, List[str]]], clip_id_frame_id_ground_truth_label_indices_mapping: Dict[str, Dict[str, List[str]]]):
    predicted_label_one_hot_vectors = []
    ground_truth_label_one_hot_vectors = []

    for clip_id, frame_id_ground_truth_label_indices_mapping in tqdm(list(clip_id_frame_id_ground_truth_label_indices_mapping.items())):
        for frame_id, ground_truth_label_indices in frame_id_ground_truth_label_indices_mapping.items():
            predicted_label_indices = clip_id_frame_id_predicted_label_indices_mapping[clip_id][int(frame_id // 6 * 6)]
            predicted_labels_one_hot_vector = np.zeros(len(distinct_ground_truth_labels) + 1)
            ground_truth_labels_one_hot_vector = np.zeros(len(distinct_ground_truth_labels) + 1)

            if len(predicted_label_indices) == 0:
                predicted_labels_one_hot_vector[-1] = 1
            else:
                for predicted_label_index in predicted_label_indices:
                    predicted_labels_one_hot_vector[predicted_label_index] = 1

            if len(ground_truth_label_indices) == 0:
                ground_truth_labels_one_hot_vector[-1] = 1
            else:
                for ground_truth_label_index in ground_truth_label_indices:
                    ground_truth_labels_one_hot_vector[ground_truth_label_index] = 1

            predicted_label_one_hot_vectors.append(predicted_labels_one_hot_vector)
            ground_truth_label_one_hot_vectors.append(ground_truth_labels_one_hot_vector)
    # f1_score_per_label = f1_score(y_true=ground_truth_label_one_hot_vectors, y_pred=predicted_label_one_hot_vectors, average=None)
    weighted_f1 = f1_score(y_true=ground_truth_label_one_hot_vectors, y_pred=predicted_label_one_hot_vectors, average="weighted", zero_division=0)
    # precision_per_label = precision_score(y_true=ground_truth_label_one_hot_vectors, y_pred=predicted_label_one_hot_vectors, average=None)
    weighted_precision = precision_score(y_true=ground_truth_label_one_hot_vectors, y_pred=predicted_label_one_hot_vectors, average="weighted", zero_division=0)
    # recall_per_label = recall_score(y_true=ground_truth_label_one_hot_vectors, y_pred=predicted_label_one_hot_vectors, average=None)
    weighted_recall = recall_score(y_true=ground_truth_label_one_hot_vectors, y_pred=predicted_label_one_hot_vectors, average="weighted", zero_division=0)
    return weighted_f1, weighted_precision, weighted_recall


def calculate_sbert_cosine_similarities(blip2_sbert_embeddings: List[np.array], label_sbert_embeddings: List[np.array]):
    max_cosine_similarity = 0.0
    for blip2_sbert_embedding in blip2_sbert_embeddings:
        for label_sbert_embedding in label_sbert_embeddings:
            current_cosine_similarity = (np.dot(blip2_sbert_embedding, label_sbert_embedding)/(np.linalg.norm(blip2_sbert_embedding) * np.linalg.norm(label_sbert_embedding)) + 1.0) / 2.0
            max_cosine_similarity = max(max_cosine_similarity, current_cosine_similarity)
    return max_cosine_similarity

    
def nontemporal_dictionary_matching_for_given_clip(clip_id: str, frame_id_blip2_question_answer_verb_noun_tool_pairs_mapping: Dict[str, List[Tuple[str, str, str]]], label_verb_noun_tools_mapping: Dict[str, List[Tuple[str, str, str]]]):
    frame_id_predicted_label_indices_and_scores = dict()

    for frame_id, blip2_question_answer_verb_noun_tool_pairs_mapping in frame_id_blip2_question_answer_verb_noun_tool_pairs_mapping.items():
        frame_id_predicted_label_indices_and_scores[frame_id] = dict()
        for label_index in range(len(distinct_ground_truth_labels)):
            if label_index not in frame_id_predicted_label_indices_and_scores[frame_id].keys():
                frame_id_predicted_label_indices_and_scores[frame_id][label_index] = 0.0
            
            label = distinct_ground_truth_labels[label_index]
            label_verb_noun_tools = label_verb_noun_tools_mapping[label]

            for label_verb_noun_tool in label_verb_noun_tools:
                label_verb = label_verb_noun_tool[0].replace(" ", "")
                label_noun = label_verb_noun_tool[1].replace(" ", "")
                label_tool = label_verb_noun_tool[2].replace(" ", "")
                for blip2_question, blip2_answer_verb_noun_tool_pairs in blip2_question_answer_verb_noun_tool_pairs_mapping.items():
                    blip2_answer = blip2_answer_verb_noun_tool_pairs[0]
                    blip2_verb_noun_tool_pairs = blip2_answer_verb_noun_tool_pairs[1]
                    for blip2_verb_noun_tool_pair in blip2_verb_noun_tool_pairs:
                        blip2_verb = blip2_verb_noun_tool_pair[0].replace(" ", "")
                        blip2_noun = blip2_verb_noun_tool_pair[1].replace(" ", "")
                        blip2_tool = blip2_verb_noun_tool_pair[2].replace(" ", "")
                        if label_verb == blip2_verb and label_noun == blip2_noun and label_tool == blip2_tool:
                            frame_id_predicted_label_indices_and_scores[frame_id][label_index] = max(frame_id_predicted_label_indices_and_scores[frame_id][label_index], 1.00)
                        elif label_verb == blip2_verb and label_noun == blip2_noun:
                            frame_id_predicted_label_indices_and_scores[frame_id][label_index] = max(frame_id_predicted_label_indices_and_scores[frame_id][label_index], 0.75)
                        elif label_verb == blip2_verb and label_tool == blip2_tool:
                            frame_id_predicted_label_indices_and_scores[frame_id][label_index] = max(frame_id_predicted_label_indices_and_scores[frame_id][label_index], 0.50)
                        elif label_verb == blip2_verb:
                            frame_id_predicted_label_indices_and_scores[frame_id][label_index] = max(frame_id_predicted_label_indices_and_scores[frame_id][label_index], 0.25)

        # Normalize scores per frame so that their sum is equal to 1.0.
        sum_scores = 0.0
        for label_index in range(len(distinct_ground_truth_labels)):
            sum_scores += frame_id_predicted_label_indices_and_scores[frame_id][label_index]
        for label_index in range(len(distinct_ground_truth_labels)):
            frame_id_predicted_label_indices_and_scores[frame_id][label_index] = frame_id_predicted_label_indices_and_scores[frame_id][label_index] / float(sum_scores)

    return clip_id, frame_id_predicted_label_indices_and_scores


def nontemporal_sbert_embedding_for_given_clip(clip_id: str, frame_id_blip2_question_answer_verb_noun_tool_pairs_mapping: Dict[str, List[Tuple[str, str, str]]], label_verb_noun_tools_mapping: Dict[str, List[Tuple[str, str, str]]]):
    label_verb_noun_tool_sbert_embeddings_mapping = dict()
    label_verb_noun_sbert_embeddings_mapping = dict()
    label_verb_tool_sbert_embeddings_mapping = dict()
    label_verb_sbert_embeddings_mapping = dict()

    for label, label_verb_noun_tools in label_verb_noun_tools_mapping.items():
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

            label_sbert_embeddings = sbert.encode([
                f"{label_verb} {label_noun} using {label_tool}",
                f"{label_verb} {label_noun}",
                f"{label_verb} using {label_tool}",
                f"{label_verb}"
            ])

            label_verb_noun_tool_sbert_embeddings_mapping[label_index].append(label_sbert_embeddings[0])
            label_verb_noun_sbert_embeddings_mapping[label_index].append(label_sbert_embeddings[1])
            label_verb_tool_sbert_embeddings_mapping[label_index].append(label_sbert_embeddings[2])
            label_verb_sbert_embeddings_mapping[label_index].append(label_sbert_embeddings[3])

    frame_id_blip2_answer_sbert_embeddings_mapping = dict()
    frame_id_blip2_verb_noun_tool_sbert_embeddings_mapping = dict()
    frame_id_blip2_verb_noun_sbert_embeddings_mapping = dict()
    frame_id_blip2_verb_tool_sbert_embeddings_mapping = dict()
    frame_id_blip2_verb_sbert_embeddings_mapping = dict()

    for frame_id, blip2_question_answer_verb_noun_tool_pairs_mapping in frame_id_blip2_question_answer_verb_noun_tool_pairs_mapping.items():
        frame_id_blip2_answer_sbert_embeddings_mapping[frame_id] = []
        frame_id_blip2_verb_noun_tool_sbert_embeddings_mapping[frame_id] = []
        frame_id_blip2_verb_noun_sbert_embeddings_mapping[frame_id] = []
        frame_id_blip2_verb_tool_sbert_embeddings_mapping[frame_id] = []
        frame_id_blip2_verb_sbert_embeddings_mapping[frame_id] = []
        for blip2_question, blip2_answer_verb_noun_tool_pairs in blip2_question_answer_verb_noun_tool_pairs_mapping.items():
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
                    blip2_sbert_embeddings = sbert.encode([
                        f"{blip2_verb} {blip2_noun} using {blip2_tool}",
                        f"{blip2_verb} {blip2_noun}",
                        f"{blip2_verb} using {blip2_tool}",
                        f"{blip2_verb}",
                        f"{blip2_answer}"
                    ])
                    frame_id_blip2_verb_noun_tool_sbert_embeddings_mapping[frame_id].append(blip2_sbert_embeddings[0])
                    frame_id_blip2_verb_noun_sbert_embeddings_mapping[frame_id].append(blip2_sbert_embeddings[1])
                    frame_id_blip2_verb_tool_sbert_embeddings_mapping[frame_id].append(blip2_sbert_embeddings[2])
                    frame_id_blip2_verb_sbert_embeddings_mapping[frame_id].append(blip2_sbert_embeddings[3])
                    frame_id_blip2_answer_sbert_embeddings_mapping[frame_id].append(blip2_sbert_embeddings[4])
                else:
                    blip2_sbert_embeddings = sbert.encode([
                        f"{blip2_verb} {blip2_noun} using {blip2_tool}",
                        f"{blip2_verb} {blip2_noun}",
                        f"{blip2_verb} using {blip2_tool}",
                        f"{blip2_verb}"
                    ])
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

        for label_index in range(len(distinct_ground_truth_labels)):
            if label_index not in frame_id_predicted_label_indices_and_scores[frame_id].keys():
                frame_id_predicted_label_indices_and_scores[frame_id][label_index] = 0.0

            label_verb_noun_tool_sbert_embeddings = label_verb_noun_tool_sbert_embeddings_mapping[label_index]
            label_verb_noun_sbert_embeddings = label_verb_noun_sbert_embeddings_mapping[label_index]
            label_verb_tool_sbert_embeddings = label_verb_tool_sbert_embeddings_mapping[label_index]
            label_verb_sbert_embeddings = label_verb_sbert_embeddings_mapping[label_index]

            frame_id_predicted_label_indices_and_scores[frame_id][label_index] = max([
                calculate_sbert_cosine_similarities(blip2_sbert_embeddings=blip2_answer_sbert_embeddings, label_sbert_embeddings=label_verb_noun_tool_sbert_embeddings),
                calculate_sbert_cosine_similarities(blip2_sbert_embeddings=blip2_answer_sbert_embeddings, label_sbert_embeddings=label_verb_noun_sbert_embeddings),
                calculate_sbert_cosine_similarities(blip2_sbert_embeddings=blip2_answer_sbert_embeddings, label_sbert_embeddings=label_verb_tool_sbert_embeddings),
                calculate_sbert_cosine_similarities(blip2_sbert_embeddings=blip2_answer_sbert_embeddings, label_sbert_embeddings=label_verb_sbert_embeddings),
                calculate_sbert_cosine_similarities(blip2_sbert_embeddings=blip2_verb_noun_tool_sbert_embeddings, label_sbert_embeddings=label_verb_noun_tool_sbert_embeddings),
                calculate_sbert_cosine_similarities(blip2_sbert_embeddings=blip2_verb_noun_sbert_embeddings, label_sbert_embeddings=label_verb_noun_sbert_embeddings),
                calculate_sbert_cosine_similarities(blip2_sbert_embeddings=blip2_verb_tool_sbert_embeddings, label_sbert_embeddings=label_verb_tool_sbert_embeddings),
                calculate_sbert_cosine_similarities(blip2_sbert_embeddings=blip2_verb_sbert_embeddings, label_sbert_embeddings=label_verb_sbert_embeddings)
            ])

        # Normalize scores per frame so that their sum is equal to 1.0.
        sum_scores = 0.0
        for label_index in range(len(distinct_ground_truth_labels)):
            sum_scores += frame_id_predicted_label_indices_and_scores[frame_id][label_index]
        for label_index in range(len(distinct_ground_truth_labels)):
            frame_id_predicted_label_indices_and_scores[frame_id][label_index] = frame_id_predicted_label_indices_and_scores[frame_id][label_index] / float(sum_scores)

    return clip_id, frame_id_predicted_label_indices_and_scores


def nontemporal_select_labels_with_maximum_score_higher_than_threshold(nontemporal_clip_id_frame_id_predicted_label_indices_and_scores: Dict[str, Dict[str, List[float]]], threshold: float):
    nontemporal_clip_id_frame_id_predicted_label_indices = dict()
    for clip_id, frame_id_predicted_label_indices_and_scores in tqdm(nontemporal_clip_id_frame_id_predicted_label_indices_and_scores.items()):
        nontemporal_clip_id_frame_id_predicted_label_indices[clip_id] = dict()
        for frame_id, predicted_label_indices_and_scores in frame_id_predicted_label_indices_and_scores.items():
            labels_with_maximum_score_higher_than_a_threshold = []
            for predicted_label_index, score in predicted_label_indices_and_scores.items():
                if score >= threshold:
                    labels_with_maximum_score_higher_than_a_threshold.append(predicted_label_index)
            nontemporal_clip_id_frame_id_predicted_label_indices[clip_id][frame_id] = labels_with_maximum_score_higher_than_a_threshold
    return nontemporal_clip_id_frame_id_predicted_label_indices


# Match Ratios

label_verb_noun_tool_match_ratio_mapping = dict()
label_verb_noun_match_ratio_mapping = dict()
label_verb_tool_match_ratio_mapping = dict()
label_verb_match_ratio_mapping = dict()
label_verb_noun_tool_count_mapping = dict()

background_match_count = 0
background_count = 0

for clip_id, frame_id_ground_truth_label_indices_mapping in clip_id_frame_id_ground_truth_label_indices_mapping.items():
    for frame_id, label_indices in frame_id_ground_truth_label_indices_mapping.items():
        blip2_question_answer_verb_noun_tool_mapping = clip_id_frame_id_blip2_verb_noun_tool_pair_mapping[clip_id][int((frame_id // 6) * 6)]
        for label_index in label_indices:
            for dictionary in [label_verb_noun_tool_match_ratio_mapping, label_verb_noun_match_ratio_mapping, label_verb_tool_match_ratio_mapping, label_verb_match_ratio_mapping, label_verb_noun_tool_count_mapping]:
                if label_index not in dictionary.keys() and label_index != 0:
                    dictionary[label_index] = 0

            label = distinct_ground_truth_labels[label_index]

            if label == "background":
                for blip2_question, blip2_answer_verb_noun_tools in blip2_question_answer_verb_noun_tool_mapping.items():
                    blip2_answer = blip2_answer_verb_noun_tools[0]
                    blip2_verb_noun_tools = blip2_answer_verb_noun_tools[1]
                    if len(blip2_verb_noun_tools) == 0:
                        background_match_count += 1
                    background_count += 1
            else:
                label_verb_noun_tools = label_verb_noun_tools_mapping[label]
                blip2_question_answer_verb_noun_tool_mapping = clip_id_frame_id_blip2_verb_noun_tool_pair_mapping[clip_id][(frame_id // 6) * 6]
                for blip2_question, blip2_answer_verb_noun_tools in blip2_question_answer_verb_noun_tool_mapping.items():
                    blip2_answer = blip2_answer_verb_noun_tools[0]
                    blip2_verb_noun_tools = blip2_answer_verb_noun_tools[1]

                    for blip2_verb_noun_tool in blip2_verb_noun_tools:
                        blip2_verb = blip2_verb_noun_tool[0]
                        blip2_noun = blip2_verb_noun_tool[1]
                        blip2_tool = blip2_verb_noun_tool[2]

                        for label_verb_noun_tool in label_verb_noun_tools:
                            label_verb = label_verb_noun_tool[0]
                            label_noun = label_verb_noun_tool[1]
                            label_tool = label_verb_noun_tool[2]
                            label_verb_noun_tool_count_mapping[label_index] += 1
                            if blip2_verb == label_verb and blip2_noun == label_noun and blip2_tool == label_tool:
                                label_verb_noun_tool_match_ratio_mapping[label_index] += 1
                            elif blip2_verb == label_verb and blip2_noun == label_noun:
                                label_verb_noun_match_ratio_mapping[label_index] += 1
                            elif blip2_verb == label_verb and blip2_tool == label_tool:
                                label_verb_tool_match_ratio_mapping[label_index] += 1
                            elif blip2_verb == label_verb:
                                label_verb_match_ratio_mapping[label_index] += 1

for dictionary in [label_verb_noun_tool_match_ratio_mapping, label_verb_noun_match_ratio_mapping, label_verb_tool_match_ratio_mapping, label_verb_match_ratio_mapping]:
    for label_index in dictionary.keys():
        dictionary[label_index] = dictionary[label_index] / float(label_verb_noun_tool_count_mapping[label_index])

background_match_ratio = background_match_count / float(background_count)

with open(os.path.join(os.environ["SCRATCH"], "ego4d_data/v2/analysis_data", "match_ratios.pickle"), "wb") as writer:
    pickle.dump(
        {
            "label_verb_noun_tool_match_ratio_mapping": label_verb_noun_tool_match_ratio_mapping,
            "label_verb_noun_match_ratio_mapping": label_verb_noun_match_ratio_mapping,
            "label_verb_tool_match_ratio_mapping": label_verb_tool_match_ratio_mapping,
            "label_verb_match_ratio_mapping": label_verb_match_ratio_mapping,
            "background_match_ratio": background_match_ratio
        },
        writer
    )

# ASL Ego4D Baseline

with open(os.path.join(os.environ["SCRATCH"], "ego4d_data/v2/analysis_data", "clip_id_frame_id_asl_predicted_label_indices_and_scores_mapping.pickle"), "rb") as reader:
    clip_id_frame_id_asl_predicted_label_indices_and_scores_mapping = pickle.load(reader)

for threshold in [0.05, 0.10, 0.25, 0.5, 0.75, 1.0]:
    nontemporal_selected_labels_with_maximum_score_higher_than_threshold = nontemporal_select_labels_with_maximum_score_higher_than_threshold(nontemporal_clip_id_frame_id_predicted_label_indices_and_scores=clip_id_frame_id_asl_predicted_label_indices_and_scores_mapping, threshold=threshold)
    weighted_f1, weighted_precision, weighted_recall = evaluate_predictions(clip_id_frame_id_predicted_label_indices_mapping=nontemporal_selected_labels_with_maximum_score_higher_than_threshold, clip_id_frame_id_ground_truth_label_indices_mapping=clip_id_frame_id_ground_truth_label_indices_mapping)
    print(f"ASL Ego4D Baseline | Threshold: {np.round(threshold, 2)} | Weighted F1 Score: {np.round(weighted_f1, 2)} | Weighted Precision: {np.round(weighted_precision, 2)} | Weighted Recall: {np.round(weighted_recall, 2)}")


# BLIP2 Dictionary Mapping

nontemporal_dictionary_matching_clip_id_frame_id_predicted_label_indices_and_scores = dict(pqdm(
    [{"clip_id": clip_id, "frame_id_blip2_question_answer_verb_noun_tool_pairs_mapping": frame_id_blip2_question_answer_verb_noun_tool_pairs_mapping, "label_verb_noun_tools_mapping": label_verb_noun_tools_mapping} for clip_id, frame_id_blip2_question_answer_verb_noun_tool_pairs_mapping in clip_id_frame_id_blip2_verb_noun_tool_pair_mapping.items()],
    function=nontemporal_dictionary_matching_for_given_clip,
    n_jobs=4,
    exception_behaviour="immediate",
    argument_type="kwargs",
))

with open(os.path.join(os.environ["SCRATCH"], "ego4d_data/v2/analysis_data", "nontemporal_dictionary_matching_clip_id_frame_id_predicted_label_indices_and_scores.pickle"), "wb") as writer:
    pickle.dump(nontemporal_dictionary_matching_clip_id_frame_id_predicted_label_indices_and_scores, writer)

for threshold in [0.05, 0.10, 0.25, 0.5, 0.75, 1.0]:
    nontemporal_selected_labels_with_maximum_score_higher_than_threshold = nontemporal_select_labels_with_maximum_score_higher_than_threshold(nontemporal_clip_id_frame_id_predicted_label_indices_and_scores=nontemporal_dictionary_matching_clip_id_frame_id_predicted_label_indices_and_scores, threshold=threshold)
    weighted_f1, weighted_precision, weighted_recall = evaluate_predictions(clip_id_frame_id_predicted_label_indices_mapping=nontemporal_selected_labels_with_maximum_score_higher_than_threshold, clip_id_frame_id_ground_truth_label_indices_mapping=clip_id_frame_id_ground_truth_label_indices_mapping)
    print(f"BLIP2 Dictionary Matching | Threshold: {np.round(threshold, 2)} | Weighted F1 Score: {np.round(weighted_f1, 2)} | Weighted Precision: {np.round(weighted_precision, 2)} | Weighted Recall: {np.round(weighted_recall, 2)}")


# BLIP2 SBERT Embedding

nontemporal_sbert_embedding_clip_id_frame_id_predicted_label_indices_and_scores = dict()
for clip_id, frame_id_blip2_verb_noun_tool_pair_mapping in tqdm(list(clip_id_frame_id_blip2_verb_noun_tool_pair_mapping.items())):
    nontemporal_sbert_embedding_clip_id_frame_id_predicted_label_indices_and_scores[clip_id] = nontemporal_sbert_embedding_for_given_clip(clip_id=clip_id, frame_id_blip2_question_answer_verb_noun_tool_pairs_mapping=frame_id_blip2_verb_noun_tool_pair_mapping, label_verb_noun_tools_mapping=label_verb_noun_tools_mapping)[1]

with open(os.path.join(os.environ["SCRATCH"], "ego4d_data/v2/analysis_data", "nontemporal_sbert_embedding_clip_id_frame_id_predicted_label_indices_and_scores.pickle"), "wb") as writer:
    pickle.dump(nontemporal_sbert_embedding_clip_id_frame_id_predicted_label_indices_and_scores, writer)

for threshold in [0.05, 0.10, 0.25, 0.5, 0.75, 1.0]:
    nontemporal_selected_labels_with_maximum_score_higher_than_threshold = nontemporal_select_labels_with_maximum_score_higher_than_threshold(nontemporal_clip_id_frame_id_predicted_label_indices_and_scores=nontemporal_sbert_embedding_clip_id_frame_id_predicted_label_indices_and_scores, threshold=threshold)
    weighted_f1, weighted_precision, weighted_recall = evaluate_predictions(clip_id_frame_id_predicted_label_indices_mapping=nontemporal_selected_labels_with_maximum_score_higher_than_threshold, clip_id_frame_id_ground_truth_label_indices_mapping=clip_id_frame_id_ground_truth_label_indices_mapping)
    print(f"BLIP2 SBERT Embedding | Threshold: {np.round(threshold, 2)} | Weighted F1 Score: {np.round(weighted_f1, 2)} | Weighted Precision: {np.round(weighted_precision, 2)} | Weighted Recall: {np.round(weighted_recall, 2)}")
