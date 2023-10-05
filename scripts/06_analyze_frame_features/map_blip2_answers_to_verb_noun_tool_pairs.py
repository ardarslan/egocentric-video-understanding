import os
import pickle
import argparse
from pqdm.processes import pqdm

from utils import (
    get_clip_ids,
    get_clip_id_frame_id_label_indices_mapping,
    get_clip_id_frame_id_blip2_answers_mapping,
    get_clip_id_frame_id_asl_predicted_label_indices_and_scores_mapping,
    get_verb_noun_tool_pairs_per_clip,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--annotations_json_file_path",
        default=os.path.join(
            os.environ["CODE"],
            "scripts/07_reproduce_baseline_results/data/ego4d/ego4d_clip_annotations_v3.json",
        ),
        type=str,
    )
    parser.add_argument(
        "--input_data_file_path",
        default=os.path.join(
            os.environ["SCRATCH"],
            "ego4d_data/v2",
            "analysis_data",
            "analysis_data.pickle",
        ),
        type=str,
    )
    parser.add_argument(
        "--output_data_file_path",
        default=os.path.join(
            os.environ["SCRATCH"],
            "ego4d_data/v2",
            "analysis_data",
            "clip_id_frame_id_blip2_verb_noun_tool_pair_mapping.pickle",
        ),
        type=str,
    )
    args = parser.parse_args()

    if not os.path.exists(args.input_data_file_path):
        train_clip_ids, val_clip_ids, test_clip_ids = get_clip_ids(
            annotations_json_file_path=args.annotations_json_file_path
        )
        clip_ids = (
            set(train_clip_ids).union(set(val_clip_ids)).union(set(test_clip_ids))
        )
        clip_id_frame_id_label_indices_mapping = (
            get_clip_id_frame_id_label_indices_mapping(
                clip_ids=clip_ids,
                annotations_json_file_path=args.annotations_json_file_path,
            )
        )
        clip_id_frame_id_blip2_answers_mapping = (
            get_clip_id_frame_id_blip2_answers_mapping(clip_ids=clip_ids)
        )
        clip_id_frame_id_asl_predicted_label_indices_and_scores_mapping = get_clip_id_frame_id_asl_predicted_label_indices_and_scores_mapping(
            label_verb_noun_tool_mapping_path=os.path.join(
                os.environ["CODE"],
                "scripts/06_analyze_frame_features/label_verb_noun_tool_mapping.json",
            ),
            asl_predictions_path=os.path.join(
                os.environ["CODE"],
                "scripts/07_reproduce_baseline_results/submission_final.json",
            ),
        )

        os.makedirs(
            os.path.join(os.environ["SCRATCH"], "ego4d_data/v2", "analysis_data"),
            exist_ok=True,
        )

        analysis_data = {
            "clip_ids": clip_ids,
            "clip_id_frame_id_label_indices_mapping": clip_id_frame_id_label_indices_mapping,
            "clip_id_frame_id_blip2_answers_mapping": clip_id_frame_id_blip2_answers_mapping,
            "clip_id_frame_id_asl_predicted_label_indices_and_scores_mapping": clip_id_frame_id_asl_predicted_label_indices_and_scores_mapping,
        }

        with open(args.input_data_file_path, "wb") as writer:
            pickle.dump(analysis_data, writer)
    else:
        with open(args.input_data_file_path, "rb") as reader:
            analysis_data = pickle.load(reader)

        clip_ids = analysis_data["clip_ids"]
        clip_id_frame_id_label_indices_mapping = analysis_data[
            "clip_id_frame_id_label_indices_mapping"
        ]
        clip_id_frame_id_blip2_answers_mapping = analysis_data[
            "clip_id_frame_id_blip2_answers_mapping"
        ]

    result = pqdm(
        [
            {
                "clip_id": clip_id,
                "frame_id_blip2_answers_mapping": frame_id_blip2_answers_mapping,
            }
            for clip_id, frame_id_blip2_answers_mapping in list(
                clip_id_frame_id_blip2_answers_mapping.items()
            )
        ],
        function=get_verb_noun_tool_pairs_per_clip,
        n_jobs=8,
        argument_type="kwargs",
        exception_behaviour="immediate",
    )

    result = dict(result)

    with open(args.output_data_file_path, "wb") as writer:
        pickle.dump(result, writer)
