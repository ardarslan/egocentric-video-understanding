import os
import json
import pickle
import argparse
import tempfile
from tqdm import tqdm
from pathlib import Path

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
        "--annotations_file_path",
        type=str,
        default=os.path.join(
            os.environ["CODE"],
            "scripts/08_reproduce_mq_experiments/data/ego4d/ego4d_clip_annotations_v3.json",
        ),
    )
    parser.add_argument("--split", type=str, choices=["train", "val"], required=True)
    args = parser.parse_args()

    clip_ids = []
    with open(args.annotations_file_path, "rb") as reader:
        annotations = json.load(reader)
        for clip_id in annotations.keys():
            if annotations[clip_id]["subset"] == args.split:
                clip_ids.append(clip_id)

    with open(args.input_file_path, "rb") as reader:
        clip_id_frame_id_label_indices_mapping = pickle.load(reader)

    file_path = os.path.join(
        str(Path(args.input_file_path).parent), f"{args.split}_ground_truth_labels.txt"
    )
    dirname, basename = os.path.split(file_path)
    tf = tempfile.NamedTemporaryFile(prefix=basename, dir=dirname, delete=False)

    for (
        clip_id,
        frame_id_label_indices_mapping,
    ) in tqdm(clip_id_frame_id_label_indices_mapping.items()):
        if clip_id in clip_ids:
            for frame_id, label_indices in frame_id_label_indices_mapping.items():
                for label_index in label_indices:
                    tf.write(
                        str.encode(
                            f"{os.path.join(os.environ['SCRATCH'], 'ego4d_data/v2/clips', clip_id + '.mp4')} {label_index} {frame_id} {frame_id + 1}\n"
                        )
                    )

    tf.flush()
