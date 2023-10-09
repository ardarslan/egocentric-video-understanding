import os
import pickle
import argparse
from pqdm.processes import pqdm
from typing import Dict, List, Tuple

from pathlib import Path

import constants


def nontemporal_prediction_postprocessing_per_clip(
    clip_id: str,
    frame_id_predicted_label_indices_and_scores: Dict[int, Dict[int, List[Tuple[int, float]]]],
    query_score_type: int,
):
    frame_id_selected_label_index_score_mapping = dict()
    for (
        frame_id,
        predicted_label_indices_and_scores_nontemporal_dictionary_matching,
    ) in frame_id_predicted_label_indices_and_scores.items():
        max_score = 0.0
        selected_label_index = -1
        for (
            predicted_label_index,
            score_tuples,
        ) in predicted_label_indices_and_scores_nontemporal_dictionary_matching.items():
            for score_tuple in score_tuples:
                if query_score_type == "max_of_max_of_all" or constants.query_score_type_constant_mapping[query_score_type] == score_tuple[0]:
                    if score_tuple[1] > max_score:
                        max_score = score_tuple[1]
                        selected_label_index = predicted_label_index
        frame_id_selected_label_index_score_mapping[frame_id] = (
            selected_label_index,
            max_score,
        )
    return clip_id, frame_id_selected_label_index_score_mapping


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Argument parser")
    parser.add_argument(
        "--clip_id_frame_id_predicted_label_indices_and_scores_file_path_wo_ext",
        type=str,
        default=os.path.join(
            os.environ["SCRATCH"],
            "ego4d_data/v2/analysis_data",
            "clip_id_frame_id_predicted_label_indices_and_scores_nontemporal_dictionary_matching",
        ),
    )
    parser.add_argument(
        "--query_score_type",
        type=str,
        choices=[
            "max_of_max_of_all",
            "max_of_max_of_blip2_answer_label_verb_noun_tool",
            "max_of_max_of_blip2_answer_label_verb_noun",
            "max_of_max_of_blip2_answer_label_verb",
            "max_of_max_of_blip2_verb_noun_tool_label_verb_noun_tool",
            "max_of_max_of_blip2_verb_noun_label_verb_noun",
            "max_of_max_of_blip2_verb_tool_label_verb_tool",
            "max_of_max_of_blip2_verb_label_verb",
        ],
        default="max_of_max_of_all",
    )
    parser.add_argument(
        "--clip_id_frame_id_selected_label_index_score_mapping_file_path",
        type=str,
        default=os.path.join(
            os.environ["SCRATCH"],
            "ego4d_data/v2/analysis_data",
            "clip_id_frame_id_selected_label_index_score_mapping.pickle",
        ),
    )
    args = parser.parse_args()

    parent_path = Path(args.clip_id_frame_id_predicted_label_indices_and_scores_file_path_wo_ext).parent
    file_paths = [
        os.path.join(parent_path, file_name) for file_name in os.listdir(parent_path) if file_name.startswith(args.clip_id_frame_id_predicted_label_indices_and_scores_file_path_wo_ext.split("/")[-1])
    ]

    for file_path in file_paths:
        with open(file_path, "rb") as reader:
            current_clip_id_frame_id_predicted_label_indices_and_scores = pickle.load(reader)

        pqdm(
            [
                {
                    "clip_id": clip_id,
                    "frame_id_predicted_label_indices_and_scores": frame_id_predicted_label_indices_and_scores,
                    "query_score_type": args.query_score_type,
                }
                for clip_id, frame_id_predicted_label_indices_and_scores in current_clip_id_frame_id_predicted_label_indices_and_scores.items()
            ],
            function=nontemporal_prediction_postprocessing_per_clip,
            n_jobs=8,
            argument_type="kwargs",
            exception_behaviour="immediate",
        )

        with open(
            args.clip_id_frame_id_selected_label_index_score_mapping_file_path_wo_ext + file_path.split("_")[-1],
            "wb",
        ) as writer:
            pickle.dump(
                clip_id_frame_id_selected_label_index_score_mapping,
                writer,
            )
