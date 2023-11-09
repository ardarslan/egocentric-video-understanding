import os
import gc
import pickle
import argparse
from tqdm import tqdm
from typing import Dict, List

import sys

sys.path.append("../../04_extract_frame_features/")


def max_per_question_per_label_postprocessing_per_clip(
    clip_id: str,
    frame_id_blip2_question_index_label_index_scores_mapping: Dict[
        int, Dict[int, List[float]]
    ],
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
            for label_index, scores in label_index_scores_mapping.items():
                max_score = 0.0
                for current_score in scores:
                    if current_score > max_score:
                        max_score = current_score
                frame_id_blip2_question_index_label_index_max_score_mapping[frame_id][
                    blip2_question_index
                ][label_index] = max_score
                sum_max_scores += max_score

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
