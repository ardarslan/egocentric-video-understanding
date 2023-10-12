import os
import pickle
import argparse
from pqdm.processes import pqdm
from typing import Dict, List, Tuple

import constants


def max_per_label_postprocessing_per_clip(
    clip_id: str,
    frame_id_predicted_label_indices_and_scores: Dict[int, Dict[int, List[Tuple[int, float]]]],
    query_score_type: int,
):
    frame_id_predicted_label_index_max_score_mapping = dict()
    for (
        frame_id,
        predicted_label_indices_and_scores,
    ) in frame_id_predicted_label_indices_and_scores.items():
        frame_id_predicted_label_index_max_score_mapping[frame_id] = dict()
        for predicted_label_index, score_tuples in predicted_label_indices_and_scores.items():
            constant_with_maximum_score = constants.query_score_type_constant_mapping[query_score_type]

            max_score = 0.0
            for score_tuple in score_tuples:
                current_constant = score_tuple[0]
                if query_score_type == "max_of_all" or constants.query_score_type_constant_mapping[query_score_type] == current_constant:
                    current_score = score_tuple[1]
                    if current_score > max_score:
                        max_score = current_score
                        constant_with_maximum_score = current_constant
            frame_id_predicted_label_index_max_score_mapping[frame_id][predicted_label_index] = (constant_with_maximum_score, max_score)
    return clip_id, frame_id_predicted_label_index_max_score_mapping


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Argument parser")
    parser.add_argument(
        "--input_folder_path",
        type=str,
        default=os.path.join(
            os.environ["SCRATCH"],
            "ego4d_data/v2/analysis_data",
            "dictionary_matching_results",
        ),
    )
    parser.add_argument(
        "--query_score_type",
        type=str,
        choices=[
            "max_of_all",
            "max_of_blip2_answer_label_verb_noun_tool",
            "max_of_blip2_answer_label_verb_noun",
            "max_of_blip2_answer_label_verb",
            "max_of_blip2_verb_noun_tool_label_verb_noun_tool",
            "max_of_blip2_verb_noun_label_verb_noun",
            "max_of_blip2_verb_tool_label_verb_tool",
            "max_of_blip2_verb_label_verb",
        ],
        default="max_of_all",
    )
    parser.add_argument(
        "--output_folder_path",
        type=str,
        default=os.path.join(
            os.environ["SCRATCH"],
            "ego4d_data/v2/analysis_data",
            "dictionary_matching_max_per_label_postprocessing_results",
        ),
    )
    args = parser.parse_args()

    os.makedirs(args.output_folder_path, exist_ok=True)

    for input_file_name in os.listdir(args.input_folder_path):
        input_file_path = os.path.join(args.input_folder_path, input_file_name)
        with open(input_file_path, "rb") as reader:
            current_clip_id_frame_id_predicted_label_indices_and_scores = pickle.load(reader)

        current_clip_id_frame_id_selected_label_index_score_mapping = dict(
            pqdm(
                [
                    {
                        "clip_id": clip_id,
                        "frame_id_predicted_label_indices_and_scores": frame_id_predicted_label_indices_and_scores,
                        "query_score_type": args.query_score_type,
                    }
                    for clip_id, frame_id_predicted_label_indices_and_scores in current_clip_id_frame_id_predicted_label_indices_and_scores.items()
                ],
                function=max_per_label_postprocessing_per_clip,
                n_jobs=24,
                argument_type="kwargs",
                exception_behaviour="immediate",
            )
        )

        with open(
            os.path.join(args.output_folder_path, input_file_path.split("/")[-1]),
            "wb",
        ) as writer:
            pickle.dump(
                current_clip_id_frame_id_selected_label_index_score_mapping,
                writer,
            )
