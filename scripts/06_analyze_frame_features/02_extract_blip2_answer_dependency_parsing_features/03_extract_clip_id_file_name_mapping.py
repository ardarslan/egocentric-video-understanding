import os
import pickle
import pandas as pd

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Argument parser")
    parser.add_argument(
        "--blip2_dictionary_matching_predicted_action_instances_folder_path",
        type=str,
        default=f"{os.environ['SCRATCH']}/ego4d_data/v2/analysis_data/blip2_dictionary_matching_max_per_label_predictions",
    )
    parser.add_argument(
        "--output_file_path",
        type=str,
        default=os.path.join(
            os.environ["SCRATCH"],
            "ego4d_data/v2/analysis_data/max_per_label_predictions_clip_id_file_name_mapping.tsv",
        ),
    )
    args = parser.parse_args()

    blip2_dictionary_matching_predicted_action_instances_file_names = os.listdir(
        args.blip2_dictionary_matching_predicted_action_instances_folder_path
    )

    blip2_dictionary_matching_clip_id_file_name_mapping = dict()
    for file_name in blip2_dictionary_matching_predicted_action_instances_file_names:
        with open(
            os.path.join(
                args.blip2_dictionary_matching_predicted_action_instances_folder_path,
                file_name,
            ),
            "rb",
        ) as reader:
            current_mapping = pickle.load(reader)
            for clip_id in current_mapping.keys():
                blip2_dictionary_matching_clip_id_file_name_mapping[clip_id] = file_name

    df = pd.DataFrame.from_dict(
        {
            "clip_id": blip2_dictionary_matching_clip_id_file_name_mapping.keys(),
            "file_name": blip2_dictionary_matching_clip_id_file_name_mapping.values(),
        }
    )
    df.to_csv(args.output_file_path, sep="\t", index=False)
