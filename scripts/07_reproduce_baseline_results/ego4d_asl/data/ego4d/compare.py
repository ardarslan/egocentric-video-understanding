# encoding: utf-8
import json


def compare_anno(anno_path_old, anno_path_new):
    anno_old = json.load(open(anno_path_old, "r"))
    anno_new = json.load(open(anno_path_new, "r"))
    if len(anno_old.keys()) != len(anno_new.keys()):
        print("Lengths of the annotation files do not match!")

    for k, old_item in anno_old.items():
        new_item = anno_new[k]
        if len(old_item["annotations"]) != len(new_item["annotations"]):
            print("Lengths of the current clip's annotations do not match.")


if __name__ == "__main__":
    compare_anno(
        f"{os.environ['CODE']}/scripts/07_reproduce_baseline_results/ego4d_asl/data/ego4d/ego4d_clip_annotations_v3.json",
        f"{os.environ['CODE']}/scripts/07_reproduce_baseline_results/ego4d_asl/data/ego4d/ego4d_clip_annotations.json",
    )
