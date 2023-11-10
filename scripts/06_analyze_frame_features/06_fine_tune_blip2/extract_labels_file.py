import os
import pickle
import argparse
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file_path",
        type=str,
        default=os.path.join(
            os.environ["SCRATCH"],
            "ego4d_data/v2/analysis_data/ground_truth_labels/ground_truth_labels.pickle",
        ),
    )
    parser.add_argument(
        "--output_file_path",
        type=str,
        default=os.path.join(
            os.environ["SCRATCH"],
            "ego4d_data/v2/analysis_data/ground_truth_labels/ground_truth_labels.txt",
        ),
    )
    args = parser.parse_args()

    with open(args.input_file_path, "rb") as reader:
        clip_id_frame_id_label_indices_mapping = pickle.load(reader)

    with open(args.output_file_path, "w") as writer:
        for (
            clip_id,
            frame_id_label_indices_mapping,
        ) in tqdm(clip_id_frame_id_label_indices_mapping.items()):
            for frame_id, label_indices in frame_id_label_indices_mapping.items():
                for label_index in label_indices:
                    writer.write(
                        f"{os.path.join(os.environ['SCRATCH'], 'ego4d_data/v2/clips', clip_id + '.mp4')} {label_index} {frame_id} {frame_id + 1}\n"
                    )
