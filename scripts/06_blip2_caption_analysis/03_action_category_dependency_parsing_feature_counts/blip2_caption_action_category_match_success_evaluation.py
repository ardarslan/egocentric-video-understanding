import os
import pickle
import json
from tqdm import tqdm


os.environ["CODE"] = "/local/home/aarslan/mq"
os.environ["SCRATCH"] = "/data/aarslan"

with open(
    os.path.join(
        os.environ["SCRATCH"],
        "ego4d_data/v2/analysis_data/ground_truth_labels/ground_truth_labels.pickle",
    ),
    "rb",
) as reader:
    ground_truth_labels = pickle.load(reader)

with open(
    os.path.join(
        os.environ["CODE"],
        "scripts/06_blip2_caption_analysis/02_map_label_dependency_parsing_features_and_blip2_answer_dependency_parsing_features",
        "label_verb_noun_tool_mapping.json",
    ),
    "r",
) as reader:
    label_verb_noun_tools_mapping = json.load(reader)

with open(
    os.path.join(
        os.environ["CODE"],
        "scripts/08_reproduce_mq_experiments/data/ego4d",
        "ego4d_clip_annotations_v3.json",
    ),
    "r",
) as reader:
    annotations = json.load(reader)

label_verb_noun_tools_mapping_keys = sorted(
    list(label_verb_noun_tools_mapping.keys())
) + ["background"]

label_index_label_mapping = dict()
for i in range(len(label_verb_noun_tools_mapping_keys)):
    label_index_label_mapping[i] = label_verb_noun_tools_mapping_keys[i]


def get_per_frame_match_counts(clip_id, frame_id, question_index):
    current_label_indices = ground_truth_labels[clip_id][frame_id]
    current_labels = [
        label_index_label_mapping[label_index] for label_index in current_label_indices
    ]
    with open(
        os.path.join(
            os.environ["SCRATCH"],
            f"ego4d_data/v2/analysis_data/dependency_parsing_results/{clip_id}.pickle",
        ),
        "rb",
    ) as reader:
        dependency_parsing_results = pickle.load(reader)[clip_id]
    current_blip2_verb_noun_tools = dependency_parsing_results[int(frame_id // 6 * 6)][
        question_index
    ][1]
    results = {
        "background_match_count": 0,
        "background_count": 0,
        "verb_noun_tool_match_count": 0,
        "verb_noun_match_count": 0,
        "verb_tool_match_count": 0,
        "verb_match_count": 0,
        "noun_match_count": 0,
        "nonbackground_count": 0,
    }
    if current_labels[0] == "background":
        results["background_count"] = 1
        if len(current_blip2_verb_noun_tools) == 0:
            results["background_match_count"] = 1
    else:
        results["nonbackground_count"] = len(current_labels)

        for current_label in current_labels:
            current_label_verb_noun_tool_list = label_verb_noun_tools_mapping[
                current_label
            ]
            current_verb_noun_tool_match = 0
            current_verb_noun_match = 0
            current_verb_tool_match = 0
            current_verb_match = 0
            current_noun_match = 0

            for current_label_verb_noun_tool in current_label_verb_noun_tool_list:
                current_label_verb = current_label_verb_noun_tool[0]
                current_label_noun = current_label_verb_noun_tool[1]
                current_label_tool = current_label_verb_noun_tool[2]

                for current_blip2_verb_noun_tool in current_blip2_verb_noun_tools:
                    current_blip2_verb = current_blip2_verb_noun_tool[0]
                    current_blip2_noun = current_blip2_verb_noun_tool[1]
                    current_blip2_tool = current_blip2_verb_noun_tool[2]
                    if (
                        current_blip2_verb == current_label_verb
                        and current_blip2_noun == current_label_noun
                        and current_blip2_tool == current_label_tool
                    ):
                        current_verb_noun_tool_match = 1
                    if (
                        current_blip2_verb == current_label_verb
                        and current_blip2_noun == current_label_noun
                    ):
                        current_verb_noun_match = 1
                    if (
                        current_blip2_verb == current_label_verb
                        and current_blip2_tool == current_label_tool
                    ):
                        current_verb_tool_match = 1
                    if current_blip2_verb == current_label_verb:
                        current_verb_match = 1
                    if current_blip2_noun == current_label_noun:
                        current_noun_match = 1

            results["verb_noun_tool_match_count"] += current_verb_noun_tool_match
            results["verb_noun_match_count"] += current_verb_noun_match
            results["verb_tool_match_count"] += current_verb_tool_match
            results["verb_match_count"] += current_verb_match
            results["noun_match_count"] += current_noun_match

    return results


question_index_ratios_mapping = dict()
for question_index in [0, 1, 2]:
    question_index_ratios_mapping[question_index] = dict()
    final_match_counts = {
        "background_match_count": 0,
        "background_count": 0,
        "verb_noun_tool_match_count": 0,
        "verb_noun_match_count": 0,
        "verb_tool_match_count": 0,
        "verb_match_count": 0,
        "noun_match_count": 0,
        "nonbackground_count": 0,
    }
    for clip_id in tqdm(ground_truth_labels.keys()):
        if annotations[clip_id]["subset"] != "val":
            continue
        for frame_id in ground_truth_labels[clip_id].keys():
            current_match_counts = get_per_frame_match_counts(
                clip_id=clip_id, frame_id=frame_id, question_index=question_index
            )
            for key, value in current_match_counts.items():
                final_match_counts[key] += value

    question_index_ratios_mapping[question_index][
        "background_match_ratio"
    ] = final_match_counts["background_match_count"] / float(
        final_match_counts["background_count"]
    )
    question_index_ratios_mapping[question_index][
        "verb_noun_tool_match_ratio"
    ] = final_match_counts["verb_noun_tool_match_count"] / float(
        final_match_counts["nonbackground_count"]
    )
    question_index_ratios_mapping[question_index][
        "verb_noun_match_ratio"
    ] = final_match_counts["verb_noun_match_count"] / float(
        final_match_counts["nonbackground_count"]
    )
    question_index_ratios_mapping[question_index][
        "verb_tool_match_ratio"
    ] = final_match_counts["verb_tool_match_count"] / float(
        final_match_counts["nonbackground_count"]
    )
    question_index_ratios_mapping[question_index][
        "verb_match_ratio"
    ] = final_match_counts["verb_match_count"] / float(
        final_match_counts["nonbackground_count"]
    )
    question_index_ratios_mapping[question_index][
        "noun_match_ratio"
    ] = final_match_counts["noun_match_count"] / float(
        final_match_counts["nonbackground_count"]
    )

print(
    'Statistics for "What does the image describe?"', question_index_ratios_mapping[0]
)
print(
    'Statistics for "What is the person in this picture doing?"',
    question_index_ratios_mapping[1],
)
print(
    'Statistics for "What is happening in this picture?"',
    question_index_ratios_mapping[2],
)
