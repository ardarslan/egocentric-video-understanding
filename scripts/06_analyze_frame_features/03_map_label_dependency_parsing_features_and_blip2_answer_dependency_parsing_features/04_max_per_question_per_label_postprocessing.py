import os
import gc
import pickle
import argparse
from tqdm import tqdm
from typing import Dict, List, Tuple

import constants


def max_per_question_per_label_postprocessing_per_clip(
    clip_id: str,
    frame_id_blip2_question_index_label_index_scores_mapping: Dict[
        int, Dict[int, List[Tuple[int, float]]]
    ],
    query_score_type: int,
):
    frame_id_blip2_question_index_label_index_max_score_mapping = dict()
    for (
        frame_id,
        blip2_question_index_label_index_scores_mapping,
    ) in frame_id_blip2_question_index_label_index_scores_mapping.items():
        frame_id_blip2_question_index_label_index_max_score_mapping[frame_id] = dict()
        for (
            blip2_question_index,
            label_index_scores_mapping,
        ) in blip2_question_index_label_index_scores_mapping.items():
            frame_id_blip2_question_index_label_index_max_score_mapping[frame_id][
                blip2_question_index
            ] = dict()
            sum_max_scores = 0.0
            for label_index, score_tuples in label_index_scores_mapping.items():
                constant_with_maximum_score = (
                    constants.query_score_type_constant_mapping[query_score_type]
                )

                max_score = 0.0
                for score_tuple in score_tuples:
                    current_constant = score_tuple[0]
                    if (
                        query_score_type == "max_of_all"
                        or constants.query_score_type_constant_mapping[query_score_type]
                        == current_constant
                    ):
                        current_score = score_tuple[1]
                        if current_score > max_score:
                            max_score = current_score
                            constant_with_maximum_score = current_constant
                frame_id_blip2_question_index_label_index_max_score_mapping[frame_id][
                    blip2_question_index
                ][label_index] = (constant_with_maximum_score, max_score)
                sum_max_scores += max_score

            for (
                label_index
            ) in frame_id_blip2_question_index_label_index_max_score_mapping[frame_id][
                blip2_question_index
            ].keys():
                frame_id_blip2_question_index_label_index_max_score_mapping[frame_id][
                    blip2_question_index
                ][label_index] = (
                    frame_id_blip2_question_index_label_index_max_score_mapping[
                        frame_id
                    ][blip2_question_index][label_index][0],
                    frame_id_blip2_question_index_label_index_max_score_mapping[
                        frame_id
                    ][blip2_question_index][label_index][1]
                    / sum_max_scores,
                )

    return clip_id, frame_id_blip2_question_index_label_index_max_score_mapping


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Argument parser")
    parser.add_argument(
        "--predictions_folder_name",
        type=str,
        choices=[
            "asl_predictions",
            "blip2_dictionary_matching_predictions",
            "blip2_sbert_matching_all-distilroberta-v1_predictions",
            "blip2_sbert_matching_paraphrase-MiniLM-L6-v2_predictions",
        ],
        required=True,
    )
    parser.add_argument(
        "--query_score_type",
        type=str,
        choices=[
            "max_of_all",
            "max_of_blip2_answer_label_verb_noun_tool",
            "max_of_blip2_answer_label_verb_noun",
            "max_of_blip2_answer_label_verb",
            "max_of_blip2_answer_label_noun",
            "max_of_blip2_verb_noun_tool_label_verb_noun_tool",
            "max_of_blip2_verb_noun_label_verb_noun",
            "max_of_blip2_verb_tool_label_verb_tool",
            "max_of_blip2_verb_label_verb",
            "max_of_blip2_noun_label_noun",
        ],
        default="max_of_all",
    )
    args = parser.parse_args()

    input_folder_path = os.path.join(
        os.environ["SCRATCH"],
        "ego4d_data/v2/analysis_data",
        args.predictions_folder_name,
    )
    output_folder_path = os.path.join(
        os.environ["SCRATCH"],
        "ego4d_data/v2/analysis_data",
        f"{args.predictions_folder_name.replace('_predictions', '')}_max_per_label_predictions",
    )

    os.makedirs(output_folder_path, exist_ok=True)

    for input_file_name in tqdm(os.listdir(input_folder_path)):
        input_file_path = os.path.join(input_folder_path, input_file_name)
        if os.path.exists(
            os.path.join(output_folder_path, input_file_path.split("/")[-1])
        ):
            continue

        try:
            with open(input_file_path, "rb") as reader:
                current_clip_id_frame_id_blip2_question_index_label_index_scores_mapping = pickle.load(
                    reader
                )
        except Exception as e:
            print(input_file_path)
            raise Exception(e)

        current_clip_id_frame_id_blip2_question_index_selected_label_index_score_mapping = (
            dict()
        )

        for (
            clip_id,
            frame_id_blip2_question_index_label_index_scores_mapping,
        ) in (
            current_clip_id_frame_id_blip2_question_index_label_index_scores_mapping.items()
        ):
            current_clip_id_frame_id_blip2_question_index_selected_label_index_score_mapping[
                clip_id
            ] = max_per_question_per_label_postprocessing_per_clip(
                clip_id=clip_id,
                frame_id_blip2_question_index_label_index_scores_mapping=frame_id_blip2_question_index_label_index_scores_mapping,
                query_score_type=args.query_score_type,
            )[
                1
            ]

        with open(
            os.path.join(output_folder_path, input_file_path.split("/")[-1]),
            "wb",
        ) as writer:
            pickle.dump(
                current_clip_id_frame_id_blip2_question_index_selected_label_index_score_mapping,
                writer,
            )
        del current_clip_id_frame_id_blip2_question_index_selected_label_index_score_mapping
        gc.collect()
