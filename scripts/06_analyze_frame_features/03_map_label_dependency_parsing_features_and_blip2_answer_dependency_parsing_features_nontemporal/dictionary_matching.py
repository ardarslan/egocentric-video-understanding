import os
import json
import pickle

import argparse
from pqdm.processes import pqdm

import constants

from typing import Dict, List, Tuple


def nontemporal_dictionary_matching_for_given_clip(
    clip_id: str,
    frame_id_blip2_question_answer_verb_noun_tool_pairs_mapping: Dict[str, List[Tuple[str, str, str]]],
    label_verb_noun_tool_mapping: Dict[str, List[Tuple[str, str, str]]],
):
    frame_id_predicted_label_indices_and_scores_nontemporal_dictionary_matching = dict()

    distinct_ground_truth_labels = list(label_verb_noun_tool_mapping.keys())

    for (
        frame_id,
        blip2_question_answer_verb_noun_tool_pairs_mapping,
    ) in frame_id_blip2_question_answer_verb_noun_tool_pairs_mapping.items():
        frame_id_predicted_label_indices_and_scores_nontemporal_dictionary_matching[frame_id] = dict()
        for label_index in range(len(distinct_ground_truth_labels) + 1):
            frame_id_predicted_label_indices_and_scores_nontemporal_dictionary_matching[frame_id][label_index] = []
            if label_index == len(distinct_ground_truth_labels):
                for (
                    _,
                    blip2_answer_verb_noun_tool_pairs,
                ) in blip2_question_answer_verb_noun_tool_pairs_mapping.items():
                    blip2_verb_noun_tool_pairs = blip2_answer_verb_noun_tool_pairs[1]

                    if len(blip2_verb_noun_tool_pairs) == 0:
                        match_type = constants.BACKGROUND_MATCH
                        frame_id_predicted_label_indices_and_scores_nontemporal_dictionary_matching[frame_id][label_index].append((match_type, 1.00))
                    else:
                        match_type = constants.NO_MATCH
                        frame_id_predicted_label_indices_and_scores_nontemporal_dictionary_matching[frame_id][label_index].append((match_type, 0.00))
            else:
                frame_id_predicted_label_indices_and_scores_nontemporal_dictionary_matching[frame_id][label_index] = []

                label = distinct_ground_truth_labels[label_index]
                label_verb_noun_tools = label_verb_noun_tool_mapping[label]

                for label_verb_noun_tool in label_verb_noun_tools:
                    label_verb = label_verb_noun_tool[0].replace(" ", "")
                    label_noun = label_verb_noun_tool[1].replace(" ", "")
                    label_tool = label_verb_noun_tool[2].replace(" ", "")
                    for (
                        _,
                        blip2_answer_verb_noun_tool_pairs,
                    ) in blip2_question_answer_verb_noun_tool_pairs_mapping.items():
                        blip2_verb_noun_tool_pairs = blip2_answer_verb_noun_tool_pairs[1]
                        for blip2_verb_noun_tool_pair in blip2_verb_noun_tool_pairs:
                            blip2_verb = blip2_verb_noun_tool_pair[0].replace(" ", "")
                            blip2_noun = blip2_verb_noun_tool_pair[1].replace(" ", "")
                            blip2_tool = blip2_verb_noun_tool_pair[2].replace(" ", "")
                            if label_verb == blip2_verb and label_noun == blip2_noun and label_tool == blip2_tool:
                                match_type = constants.BLIP2_VERB_NOUN_TOOL_LABEL_VERB_NOUN_TOOL
                                frame_id_predicted_label_indices_and_scores_nontemporal_dictionary_matching[frame_id][label_index].append((match_type, 1.00))
                            elif label_verb == blip2_verb and label_noun == blip2_noun:
                                match_type = constants.BLIP2_VERB_NOUN_LABEL_VERB_NOUN
                                frame_id_predicted_label_indices_and_scores_nontemporal_dictionary_matching[frame_id][label_index].append((match_type, 0.75))
                            elif label_verb == blip2_verb and label_tool == blip2_tool:
                                match_type = constants.BLIP2_VERB_TOOL_LABEL_VERB_TOOL
                                frame_id_predicted_label_indices_and_scores_nontemporal_dictionary_matching[frame_id][label_index].append((match_type, 0.50))
                            elif label_verb == blip2_verb:
                                match_type = constants.BLIP2_VERB_LABEL_VERB
                                frame_id_predicted_label_indices_and_scores_nontemporal_dictionary_matching[frame_id][label_index].append((match_type, 0.25))

                if len(frame_id_predicted_label_indices_and_scores_nontemporal_dictionary_matching[frame_id][label_index]) == 0:
                    match_type = constants.NO_MATCH
                    frame_id_predicted_label_indices_and_scores_nontemporal_dictionary_matching[frame_id][label_index].append((match_type, 0.00))

        # Normalize scores per frame so that their sum is equal to 1.0.
        # sum_scores = 0.0
        # for label_index in range(len(distinct_ground_truth_labels)):
        #     sum_scores += frame_id_predicted_label_indices_and_scores_nontemporal_dictionary_matching[
        #         frame_id
        #     ][
        #         label_index
        #     ]
        # for label_index in range(len(distinct_ground_truth_labels)):
        #     frame_id_predicted_label_indices_and_scores_nontemporal_dictionary_matching[
        #         frame_id
        #     ][
        #         label_index
        #     ] = frame_id_predicted_label_indices_and_scores_nontemporal_dictionary_matching[
        #         frame_id
        #     ][
        #         label_index
        #     ] / float(
        #         sum_scores
        #     )

    return (
        clip_id,
        frame_id_predicted_label_indices_and_scores_nontemporal_dictionary_matching,
    )


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
            "clip_id_frame_id_predicted_label_indices_and_scores_nontemporal_dictionary_matching",
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

    current_clip_id_frame_id_predicted_label_indices_and_scores_nontemporal_dictionary_matching = dict()
    clip_counter = 0
    file_name_counter = 0
    for clip_id, frame_id_verb_noun_tool_pairs_mapping in clip_id_frame_id_verb_noun_tool_pairs_mapping.items():
        current_clip_id_frame_id_predicted_label_indices_and_scores_nontemporal_dictionary_matching[clip_id] = nontemporal_dictionary_matching_for_given_clip(
            clip_id=clip_id, frame_id_blip2_question_answer_verb_noun_tool_pairs_mapping=frame_id_verb_noun_tool_pairs_mapping, label_verb_noun_tool_mapping=label_verb_noun_tool_mapping
        )
        clip_counter += 1
        if clip_counter % 100 == 0:
            with open(args.output_path_name_wo_ext + "_" + str(file_name_counter).zfill(3) + ".pickle", "wb") as writer:
                pickle.dump(current_clip_id_frame_id_predicted_label_indices_and_scores_nontemporal_dictionary_matching, writer)
            current_clip_id_frame_id_predicted_label_indices_and_scores_nontemporal_dictionary_matching = dict()
            file_name_counter += 1
