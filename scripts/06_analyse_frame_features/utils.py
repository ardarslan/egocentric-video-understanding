import os
import cv2
import json
import pickle
import argparse
import numpy as np
import pandas as pd
import itertools
from pqdm.processes import pqdm
from pathlib import Path

from frame_embedder.frame_embedder import FrameEmbedder
from frame_embedder.word2vec_frame_embedder import Word2VecFrameEmbedder
from frame_embedder.glove_frame_embedder import GloveFrameEmbedder
from frame_embedder.one_hot_frame_embedder import OneHotFrameEmbedder
from frame_embedder.universal_sentence_encoder_frame_embedder import (
    UniversalSentenceEncoderFrameEmbedder,
)
from frame_embedder.sentence_transformer_frame_embedder import (
    SentenceTransformerFrameEmbedder,
)

from nltk.parse.corenlp import CoreNLPDependencyParser

from typing import Dict, List


taggers = {}


def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def fill_missing_cells(df_group):
    df_group_sorted = df_group.sort_values(by="frame_index").reset_index(drop=False)
    delete_first_row = False
    for index, row in df_group_sorted.iterrows():
        if pd.isnull(row["answer"]):
            if index == 0:
                delete_first_row = True
            else:
                df_group_sorted.at[index, "answer"] = df_group_sorted.at[
                    index - 1, "answer"
                ]
    if delete_first_row:
        df_group_sorted = df_group_sorted.iloc[1:, :]
    return df_group_sorted


def get_fill_value(blip2_vqa_answers_df: pd.DataFrame, frame_index: int, question: str):
    blip2_vqa_row = blip2_vqa_answers_df[
        (blip2_vqa_answers_df["frame_index"] == frame_index - 6)
        & (blip2_vqa_answers_df["question"] == question)
    ]
    if len(blip2_vqa_row) == 0:
        if frame_index == 0:
            return "no_answer"
        else:
            return get_fill_value(
                blip2_vqa_answers_df=blip2_vqa_answers_df,
                frame_index=frame_index - 6,
                question=question,
            )
    elif len(blip2_vqa_row) == 1:
        if pd.isnull(blip2_vqa_row["answer"].values[0]):
            return get_fill_value(
                blip2_vqa_answers_df=blip2_vqa_answers_df,
                frame_index=frame_index - 6,
                question=question,
            )
        else:
            return blip2_vqa_row["answer"].values[0]
    else:
        raise Exception("Duplicate rows!")


def get_clip_info(clip_id: str):
    cap = cv2.VideoCapture(
        os.path.join(os.environ["SCRATCH"], "ego4d_data/v2/clips", clip_id + ".mp4")
    )
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    return {"num_frames": num_frames, "fps": fps}


def get_clip_id_frame_id_blip2_answers_mapping(clip_ids: str):
    clip_id_frame_id_blip2_answers_mapping = dict()
    for clip_id in clip_ids:
        clip_info = get_clip_info(clip_id=clip_id)
        num_frames = clip_info["num_frames"]
        blip2_vqa_answers_df = pd.read_csv(
            os.path.join(
                os.environ["SCRATCH"],
                "ego4d_data/v2/frame_features/",
                clip_id,
                "blip2_vqa_features.tsv",
            ),
            sep="\t",
        )

        for index, row in blip2_vqa_answers_df[
            pd.isnull(blip2_vqa_answers_df["answer"])
        ].iterrows():
            frame_index = row["frame_index"]
            question = row["question"]
            fill_value = get_fill_value(
                blip2_vqa_answers_df=blip2_vqa_answers_df,
                frame_index=frame_index,
                question=question,
            )
            blip2_vqa_answers_df.iat[index, 2] = fill_value

        frame_id_blip2_answers_mapping = {}
        for frame_id in range(0, num_frames, 6):
            current_blip2_answers = blip2_vqa_answers_df[
                blip2_vqa_answers_df["frame_index"] == frame_id
            ]
            frame_id_blip2_answers_mapping[frame_id] = dict()
            for question in [
                "What is happening in this picture?",
                "What does the image describe?",
                "What does the image describe?",
            ]:
                answer = current_blip2_answers[
                    current_blip2_answers["question"] == question
                ]["answer"].values[0]
                frame_id_blip2_answers_mapping[frame_id][question] = answer
        clip_id_frame_id_blip2_answers_mapping[clip_id] = frame_id_blip2_answers_mapping
    return clip_id_frame_id_blip2_answers_mapping


def get_clip_id_frame_id_labels_mapping(
    clip_ids: List[str], annotations_json_file_path: str
):
    clip_id_frame_id_labels_mapping = dict()
    for clip_id in clip_ids:
        with open(annotations_json_file_path, "r") as reader:
            annotations = json.load(reader)
        clip_info = get_clip_info(clip_id)
        num_frames = clip_info["num_frames"]
        fps = clip_info["fps"]
        frame_id_labels_mapping = {}
        current_annotations = annotations[clip_id]["annotations"]
        for frame_id in range(num_frames):
            current_labels = set()
            if len(current_annotations) == 0:
                current_labels.add("no_annotation")
            else:
                for current_annotation in current_annotations:
                    if (
                        frame_id / fps >= current_annotation["segment"][0]
                        and frame_id / fps <= current_annotation["segment"][1]
                    ):
                        current_labels.add(current_annotation["label"])
                if len(current_labels) == 0:
                    current_labels.add("background")
            frame_id_labels_mapping[frame_id] = current_labels
        clip_id_frame_id_labels_mapping[clip_id] = frame_id_labels_mapping
    return clip_id_frame_id_labels_mapping


def get_analysis_data_file_name_wo_ext_analysis_data_mapping(
    args: argparse.Namespace,
    analysis_data_file_name_wo_ext_analysis_data_mapping_file_path: str,
):
    train_clip_ids, val_clip_ids, test_clip_ids = get_clip_ids(
        annotations_json_file_path=args.annotations_json_file_path
    )

    train_clip_id_frame_id_labels_mapping = get_clip_id_frame_id_labels_mapping(
        clip_ids=train_clip_ids,
        annotations_json_file_path=args.annotations_json_file_path,
    )
    val_clip_id_frame_id_labels_mapping = get_clip_id_frame_id_labels_mapping(
        clip_ids=val_clip_ids,
        annotations_json_file_path=args.annotations_json_file_path,
    )

    train_clip_id_frame_id_blip2_answers_mapping = (
        get_clip_id_frame_id_blip2_answers_mapping(clip_ids=train_clip_ids)
    )
    val_clip_id_frame_id_blip2_answers_mapping = (
        get_clip_id_frame_id_blip2_answers_mapping(clip_ids=val_clip_ids)
    )
    test_clip_id_frame_id_blip2_answers_mapping = (
        get_clip_id_frame_id_blip2_answers_mapping(clip_ids=test_clip_ids)
    )

    train_clip_id_frame_id_blip2_words_mapping = get_clip_id_frame_id_blip2_words_mapping(
        clip_id_frame_id_blip2_answers_mapping=train_clip_id_frame_id_blip2_answers_mapping
    )
    val_clip_id_frame_id_blip2_words_mapping = get_clip_id_frame_id_blip2_words_mapping(
        clip_id_frame_id_blip2_answers_mapping=val_clip_id_frame_id_blip2_answers_mapping
    )
    test_clip_id_frame_id_blip2_words_mapping = get_clip_id_frame_id_blip2_words_mapping(
        clip_id_frame_id_blip2_answers_mapping=test_clip_id_frame_id_blip2_answers_mapping
    )

    train_labels, train_blip2_answer_word_label_mapping = get_train_labels(
        train_clip_id_frame_id_blip2_words_mapping=train_clip_id_frame_id_blip2_words_mapping,
        train_clip_id_frame_id_labels_mapping=train_clip_id_frame_id_labels_mapping,
    )

    train_blip2_answer_word_weight_mapping = get_train_blip2_answer_word_weight_mapping(
        train_blip2_answer_word_weight_type=args.train_blip2_answer_word_weight_type,
        train_blip2_answer_word_label_mapping=train_blip2_answer_word_label_mapping,
    )

    frame_embedder = get_frame_embedder(
        frame_embedder_name=args.frame_embedder,
        train_blip2_answer_word_label_mapping=train_blip2_answer_word_weight_mapping,
        unify_words=False,
    )

    train_clip_id_frame_id_embedding_mapping = get_clip_id_frame_id_embedding_mapping(
        frame_embedder=frame_embedder,
        clip_id_frame_id_blip2_answers_mapping=train_clip_id_frame_id_blip2_answers_mapping,
        clip_id_frame_id_blip2_words_mapping=train_clip_id_frame_id_blip2_words_mapping,
    )
    val_clip_id_frame_id_embedding_mapping = get_clip_id_frame_id_embedding_mapping(
        frame_embedder=frame_embedder,
        clip_id_frame_id_blip2_answers_mapping=val_clip_id_frame_id_blip2_answers_mapping,
        clip_id_frame_id_blip2_words_mapping=val_clip_id_frame_id_blip2_words_mapping,
    )
    test_clip_id_frame_id_embedding_mapping = get_clip_id_frame_id_embedding_mapping(
        frame_embedder=frame_embedder,
        clip_id_frame_id_blip2_answers_mapping=test_clip_id_frame_id_blip2_answers_mapping,
        clip_id_frame_id_blip2_words_mapping=test_clip_id_frame_id_blip2_words_mapping,
    )

    train_X, train_y, train_clip_ids, train_frame_ids = get_data(
        clip_id_frame_id_embedding_mapping=train_clip_id_frame_id_embedding_mapping,
        clip_id_frame_id_labels_mapping=train_clip_id_frame_id_labels_mapping,
        train_labels=train_labels,
    )
    val_X, val_y, val_clip_ids, val_frame_ids = get_data(
        clip_id_frame_id_embedding_mapping=val_clip_id_frame_id_embedding_mapping,
        clip_id_frame_id_labels_mapping=val_clip_id_frame_id_labels_mapping,
        train_labels=train_labels,
    )
    test_X, _, test_clip_ids, test_frame_ids = get_data(
        clip_id_frame_id_embedding_mapping=test_clip_id_frame_id_embedding_mapping,
        clip_id_frame_id_labels_mapping=None,
        train_labels=train_labels,
    )

    analysis_data_file_name_wo_ext_analysis_data_mapping = {
        "train_labels": train_labels,
        "train_blip2_answer_word_label_mapping": train_blip2_answer_word_label_mapping,
        "train_X": train_X,
        "train_y": train_y,
        "train_clip_ids": train_clip_ids,
        "train_frame_ids": train_frame_ids,
        "val_X": val_X,
        "val_y": val_y,
        "val_clip_ids": val_clip_ids,
        "val_frame_ids": val_frame_ids,
        "test_X": test_X,
        "test_clip_ids": test_clip_ids,
        "test_frame_ids": test_frame_ids,
    }

    os.makedirs(
        Path(analysis_data_file_name_wo_ext_analysis_data_mapping_file_path).parent,
        exist_ok=True,
    )

    with open(
        analysis_data_file_name_wo_ext_analysis_data_mapping_file_path, "wb"
    ) as writer:
        pickle.dump(analysis_data_file_name_wo_ext_analysis_data_mapping, writer)

    return analysis_data_file_name_wo_ext_analysis_data_mapping


def get_data(
    clip_id_frame_id_embedding_mapping: Dict[str, Dict[str, np.array]],
    clip_id_frame_id_labels_mapping: Dict[str, Dict[str, List[str]]],
    train_labels: List[str],
):
    X = []
    if clip_id_frame_id_labels_mapping is not None:
        y = []

    clip_ids = []
    frame_ids = []

    for (
        clip_id,
        frame_id_embedding_mapping,
    ) in clip_id_frame_id_embedding_mapping.items():
        for frame_id, embedding in frame_id_embedding_mapping.items():
            if embedding is None:
                continue
            if clip_id_frame_id_labels_mapping is not None:
                labels = clip_id_frame_id_labels_mapping[clip_id][frame_id]
                found_not_train_label = False
                for label in labels:
                    if label not in train_labels:
                        found_not_train_label = True
                        break
                if found_not_train_label is False:
                    for label in labels:
                        X.append(embedding)
                        y.append(train_labels.index(label))
            else:
                X.append(embedding)
            clip_ids.append(clip_id)
            frame_ids.append(frame_id)

    X = np.vstack(X)
    if clip_id_frame_id_labels_mapping is not None:
        y = np.array(y)
    else:
        y = None

    return X, y, clip_ids, frame_ids


def get_frame_embedder(
    frame_embedder_name: str,
    train_blip2_answer_word_label_mapping: Dict[str, float],
    unify_words: bool,
):
    if frame_embedder_name == "word2vec":
        frame_embedder_class = Word2VecFrameEmbedder
    elif frame_embedder_name == "glove":
        frame_embedder_class = GloveFrameEmbedder
    elif frame_embedder_name == "one_hot":
        frame_embedder_class = OneHotFrameEmbedder
    elif frame_embedder_name == "universal_sentence_encoder":
        frame_embedder_class = UniversalSentenceEncoderFrameEmbedder
    elif frame_embedder_name == "sentence_transformer":
        frame_embedder_class = SentenceTransformerFrameEmbedder
    return frame_embedder_class(
        train_blip2_answer_word_label_mapping=train_blip2_answer_word_label_mapping,
        unify_words=unify_words,
    )


def get_clip_ids(annotations_json_file_path: str):
    with open(annotations_json_file_path, "r") as reader:
        annotations = json.load(reader)
    clip_ids = set(
        os.listdir(os.path.join(os.environ["SCRATCH"], "ego4d_data/v2/frame_features"))
    ).intersection(annotations.keys())
    train_clip_ids = []
    val_clip_ids = []
    test_clip_ids = []
    for clip_id in clip_ids:
        if annotations[clip_id]["subset"] == "train":
            train_clip_ids.append(clip_id)
        elif annotations[clip_id]["subset"] == "val":
            val_clip_ids.append(clip_id)
        else:
            test_clip_ids.append(clip_id)
    return train_clip_ids, val_clip_ids, test_clip_ids


def get_dependency_parser(server_url: str):
    if "dependency_parser" in taggers.keys():
        dependency_parser = taggers["dependency_parser"]
        return dependency_parser
    else:
        dependency_parser = CoreNLPDependencyParser(url=server_url)
        taggers["dependency_parser"] = dependency_parser
        return dependency_parser


def get_clip_id_frame_id_blip2_words_mapping(
    clip_id_frame_id_blip2_answers_mapping: Dict[str, Dict[str, Dict[str, str]]]
):
    return dict(
        pqdm(
            [
                {
                    "clip_id": clip_id,
                    "frame_id_blip2_answers_mapping": frame_id_blip2_answers_mapping,
                }
                for clip_id, frame_id_blip2_answers_mapping in clip_id_frame_id_blip2_answers_mapping.items()
            ],
            function=FrameEmbedder.process_per_clip_blip2_answers,
            n_jobs=8,
            argument_type="kwargs",
        )
    )


def get_train_labels(
    train_clip_id_frame_id_blip2_words_mapping: Dict[
        str, Dict[str, Dict[str, List[str]]]
    ],
    train_clip_id_frame_id_labels_mapping: Dict[str, Dict[str, List[str]]],
):
    train_labels = set()
    train_blip2_answer_word_label_mapping = {}
    for clip_id in train_clip_id_frame_id_blip2_words_mapping.keys():
        for frame_id in train_clip_id_frame_id_blip2_words_mapping[clip_id]:
            current_labels = train_clip_id_frame_id_labels_mapping[clip_id][frame_id]
            for word in train_clip_id_frame_id_blip2_words_mapping[clip_id][frame_id]:
                for label in current_labels:
                    train_labels.add(label)
                    if word not in train_blip2_answer_word_label_mapping.keys():
                        train_blip2_answer_word_label_mapping[word] = set([label])
                    else:
                        train_blip2_answer_word_label_mapping[word].add(label)
    train_labels = list(train_labels)
    return train_labels, train_blip2_answer_word_label_mapping


def get_train_blip2_answer_word_weight_mapping(
    train_blip2_answer_word_weight_type: str,
    train_blip2_answer_word_label_mapping: Dict[str, Dict[str, str]],
):
    if train_blip2_answer_word_weight_type == "idf":
        train_blip2_answer_word_weight_mapping = {}
        min_idf = np.inf
        max_idf = -np.inf
        for train_blip2_answer_word in train_blip2_answer_word_label_mapping.keys():
            current_idf = 1 / float(
                len(train_blip2_answer_word_label_mapping[train_blip2_answer_word])
            )
            train_blip2_answer_word_weight_mapping[
                train_blip2_answer_word
            ] = current_idf
            if current_idf > max_idf:
                max_idf = current_idf
            if current_idf < min_idf:
                min_idf = current_idf

        for (
            train_blip2_answer_word,
            idf,
        ) in train_blip2_answer_word_weight_mapping.items():
            train_blip2_answer_word_weight_mapping[train_blip2_answer_word] = (
                idf - min_idf
            ) / (max_idf - min_idf)

    elif train_blip2_answer_word_weight_type == "uniform":
        train_blip2_answer_word_weight_mapping = {}
        for train_blip2_answer_word in train_blip2_answer_word_label_mapping.keys():
            train_blip2_answer_word_weight_mapping[train_blip2_answer_word] = 1.0

    else:
        raise Exception(
            f"{train_blip2_answer_word_weight_type} is not a valid train_blip2_answer_word_weight_type."
        )

    return train_blip2_answer_word_weight_mapping


def get_clip_id_frame_id_embedding_mapping(
    frame_embedder: FrameEmbedder,
    clip_id_frame_id_blip2_answers_mapping: Dict[str, Dict[str, List[str]]],
    clip_id_frame_id_blip2_words_mapping: Dict[str, Dict[str, List[str]]],
):
    result = pqdm(
        [
            {
                "clip_id": clip_id,
                "frame_id_blip2_answers_mapping": clip_id_frame_id_blip2_answers_mapping[
                    clip_id
                ],
                "frame_id_blip2_words_mapping": clip_id_frame_id_blip2_words_mapping[
                    clip_id
                ],
            }
            for clip_id in clip_id_frame_id_blip2_answers_mapping.keys()
        ],
        function=frame_embedder.get_embedding_per_clip,
        n_jobs=8,
        argument_type="kwargs",
    )
    return dict(result)


def save_evaluation_metrics(
    args: argparse.Namespace, evaluation_metrics: Dict[str, List[float]]
):
    evaluation_metrics_df = pd.DataFrame.from_dict(evaluation_metrics)
    evaluation_metrics_file_path = os.path.join(
        os.environ["SCRATCH"],
        "ego4d_data/v2",
        "analysis_data",
        args.frame_embedder,
        args.train_blip2_answer_word_weight_type,
        "evaluation_metrics.tsv",
    )
    evaluation_metrics_df.to_csv(evaluation_metrics_file_path, sep="\t")


def save_validation_predictions(
    args: argparse.Namespace,
    val_clip_ids: List[str],
    val_frame_ids: List[int],
    val_ys: List[int],
    val_yhats: List[int],
):
    validation_predictions_df = pd.DataFrame.from_dict(
        {
            "val_clip_id": val_clip_ids,
            "val_frame_id": val_frame_ids,
            "val_y": val_ys,
            "val_yhats": val_yhats,
        }
    )
    validation_predictions_file_path = os.path.join(
        os.environ["SCRATCH"],
        "ego4d_data/v2",
        "analysis_data",
        args.frame_embedder,
        args.train_blip2_answer_word_weight_type,
        "validation_predictions.tsv",
    )
    validation_predictions_df.to_csv(validation_predictions_file_path, sep="\t")


def get_verb_noun_tool_pairs(
    blip2_answer: str,
    dependency_parser: CoreNLPDependencyParser,
):
    blip2_answer_clean = ""

    for char in blip2_answer:
        if char.isalpha() or char in [" ", "'"]:
            blip2_answer_clean += char
    blip2_answer_clean_splitted = blip2_answer_clean.split()

    verb_noun_tool_pairs = []
    dependency_parser_results = dependency_parser.parse(blip2_answer_clean_splitted)

    try:
        dependency_parser_result = dependency_parser_results.__next__()
    except Exception as e:
        e = ""
        e = e + ""
        return []

    using_addresses = []
    for current_word_address in dependency_parser_result.nodes.keys():
        current_word_node = dependency_parser_result.nodes[current_word_address]
        if current_word_node["word"] == "using":
            using_addresses.append(current_word_address)

    if dependency_parser_result is not None:
        for current_word_address in dependency_parser_result.nodes.keys():
            current_word_node = dependency_parser_result.nodes[current_word_address]
            current_word_pos_tag = current_word_node["tag"]

            # We are only interested in sentences which include a verb. We extract nouns and tools that are dependent on the verb.
            if not current_word_pos_tag.startswith("VB"):
                continue

            verb_word = current_word_node["word"]

            # If current word is "using" and the word before the current word is not "is", then continue.
            if current_word_address > 0 and verb_word == "using":
                previous_word_address = current_word_address - 1
                previous_word_node = dependency_parser_result.nodes[
                    previous_word_address
                ]
                verb_previous_word = previous_word_node["word"]
                if verb_word == "using" and verb_previous_word != "is":
                    continue

            verb_lemma = current_word_node["lemma"]

            # If the lemma of the verb is one of the following verb lemmas, then continue.
            if verb_lemma in ["be", "describe", "show", "have", "tell"]:
                continue

            if "compound:prt" in current_word_node["deps"].keys():
                verb_compound_prt_address = current_word_node["deps"]["compound:prt"][0]
                verb_compound_prt_node = dependency_parser_result.nodes[
                    verb_compound_prt_address
                ]
                verb_compound_prt_lemma = verb_compound_prt_node["lemma"]
                verb_lemma = verb_lemma + " " + verb_compound_prt_lemma

            # Extract tools that are related to the verb with a "with".
            tool_addresses = current_word_node["deps"].get("obl", [])
            filtered_tool_addresses = set()
            for tool_address in tool_addresses:
                tool_node = dependency_parser_result.nodes[tool_address]
                tool_case_addresses = tool_node["deps"].get("case", [])
                for tool_case_address in tool_case_addresses:
                    tool_case_node = dependency_parser_result.nodes[tool_case_address]
                    if tool_case_node["lemma"] == "with":
                        filtered_tool_addresses.add(tool_address)

            # Extract tools that are related to the verb with a "using".
            for using_address in using_addresses:
                using_node = dependency_parser_result.nodes[using_address]
                for using_obj_address in using_node["deps"].get("obj", []):
                    filtered_tool_addresses.add(using_obj_address)

                for using_acl_address in using_node["deps"].get("acl", []):
                    using_acl_node = dependency_parser_result[using_acl_address]
                    for using_acl_obj_address in using_acl_node["deps"].get("obj", []):
                        filtered_tool_addresses.add(using_acl_obj_address)

            # Extract noun addresses.
            noun_addresses = current_word_node["deps"].get(
                "obj", []
            ) + current_word_node["deps"].get("nsubj:pass", [])

            # Extract noun lemmas.
            noun_lemmas = []
            for noun_address in noun_addresses:
                noun_node = dependency_parser_result.nodes[noun_address]
                noun_lemma = noun_node["lemma"]
                if "compound" in noun_node["deps"]:
                    noun_compound_address = noun_node["deps"]["compound"][0]
                    noun_compound_node = dependency_parser_result.nodes[
                        noun_compound_address
                    ]
                    noun_compound_lemma = noun_compound_node["lemma"]
                    noun_lemma = noun_compound_lemma + " " + noun_lemma
                    noun_lemmas.append(noun_lemma)

            # Extract tool lemmas.
            tool_lemmas = []

            for tool_address in filtered_tool_addresses:
                tool_node = dependency_parser_result.nodes[tool_address]
                tool_lemma = tool_node["lemma"]
                if "compound" in tool_node["deps"]:
                    tool_compound_address = tool_node["deps"]["compound"][0]
                    tool_compound_node = dependency_parser_result.nodes[
                        tool_compound_address
                    ]
                    tool_compound_lemma = tool_compound_node["lemma"]
                    tool_lemma = tool_compound_lemma + " " + tool_lemma
                    tool_lemmas.append(tool_lemma)

            if len(noun_lemmas) == 0:
                noun_lemmas = ["NaN"]
            if len(tool_lemmas) == 0:
                tool_lemmas = ["NaN"]

            # For the current verb, and for each associated noun lemma and tool lemma combination, add them to verb_noun_tool_pairs.
            for noun_lemma, tool_lemma in itertools.product(noun_lemmas, tool_lemmas):
                verb_noun_tool_pairs.append((verb_lemma, noun_lemma, tool_lemma))

    return verb_noun_tool_pairs


def get_verb_noun_tool_pairs_per_clip(
    clip_id: str,
    frame_id_blip2_answers_mapping: Dict[int, Dict[str, str]],
    server_url: str,
):
    dependency_parser = get_dependency_parser(server_url=server_url)

    frame_id_verb_noun_tool_pairs_mapping = dict()
    for frame_id, blip2_answers_mapping in frame_id_blip2_answers_mapping.items():
        frame_id_verb_noun_tool_pairs_mapping[frame_id] = dict()
        for blip2_question, blip2_answer in blip2_answers_mapping.items():
            frame_id_verb_noun_tool_pairs_mapping[frame_id][blip2_question] = (
                blip2_answer,
                get_verb_noun_tool_pairs(blip2_answer, dependency_parser),
            )
    return clip_id, frame_id_verb_noun_tool_pairs_mapping
