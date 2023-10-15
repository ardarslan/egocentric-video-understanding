import os
import itertools

import cv2
import pickle
import pandas as pd
import argparse
import traceback
from pqdm.processes import pqdm
from nltk.parse.corenlp import CoreNLPServer, CoreNLPDependencyParser
import sys

sys.path.append("../03_map_label_dependency_parsing_features_and_blip2_answer_dependency_parsing_features")
import constants

try:
    server = CoreNLPServer(
        port=5960,
        path_to_jar=os.path.join(
            os.environ["SCRATCH"],
            "mq_libs/stanford-corenlp",
            "stanford-corenlp-4.5.5.jar",
        ),
        path_to_models_jar=os.path.join(
            os.environ["SCRATCH"],
            "mq_libs/stanford-corenlp",
            "stanford-corenlp-4.5.5-models.jar",
        ),
    )
    server.start()
except Exception as e:
    e = ""
    e += ""
    pass


dependency_parser = CoreNLPDependencyParser(url="http://localhost:5960")


def get_clip_info(clip_id: str):
    cap = cv2.VideoCapture(os.path.join(os.environ["SCRATCH"], "ego4d_data/v2/clips", clip_id + ".mp4"))
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    return {"num_frames": num_frames, "fps": fps}


def get_verb_noun_tool_pairs_given_a_blip2_answer(
    blip2_answer: str,
    dependency_parser: CoreNLPDependencyParser,
):
    if blip2_answer == "" or pd.isnull(blip2_answer):
        return []

    blip2_answer_clean = ""

    for char in blip2_answer:
        if char.isalpha() or char in [" ", "'"]:
            blip2_answer_clean += char
    blip2_answer_clean_splitted = blip2_answer_clean.split()

    verb_noun_tool_pairs = []

    try:
        dependency_parser_results = dependency_parser.parse(blip2_answer_clean_splitted)
        dependency_parser_result = dependency_parser_results.__next__()
    except Exception as e:
        traceback.print_exc()
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
            if current_word_pos_tag.startswith("VB"):
                verb_word = current_word_node["word"]

                # If current word is "using" and the word before the current word is not "is", then continue.
                if current_word_address > 0 and verb_word == "using":
                    previous_word_address = current_word_address - 1
                    previous_word_node = dependency_parser_result.nodes[previous_word_address]
                    verb_previous_word = previous_word_node["word"]
                    if verb_previous_word != "is":
                        continue

                verb_lemma = current_word_node["lemma"]

                # If the lemma of the verb is one of the following verb lemmas, then continue.
                if verb_lemma in ["be", "describe", "show", "have", "tell"]:
                    continue

                if "compound:prt" in current_word_node["deps"].keys():
                    verb_compound_prt_address = current_word_node["deps"]["compound:prt"][0]
                    verb_compound_prt_node = dependency_parser_result.nodes[verb_compound_prt_address]
                    verb_compound_prt_lemma = verb_compound_prt_node["lemma"]
                    verb_lemma = verb_lemma + " " + verb_compound_prt_lemma
                verb_lemma = verb_lemma.replace("-", " ")

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

                if verb_lemma != "use":
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
                noun_addresses = current_word_node["deps"].get("obj", []) + current_word_node["deps"].get("nsubj:pass", [])

                # Extract noun lemmas.
                noun_lemmas = []
                for noun_address in noun_addresses:
                    noun_node = dependency_parser_result.nodes[noun_address]
                    noun_lemma = noun_node["lemma"]
                    if "compound" in noun_node["deps"]:
                        noun_compound_address = noun_node["deps"]["compound"][0]
                        noun_compound_node = dependency_parser_result.nodes[noun_compound_address]
                        noun_compound_lemma = noun_compound_node["lemma"]
                        noun_lemma = noun_compound_lemma + " " + noun_lemma
                    noun_lemma = noun_lemma.replace("-", " ")
                    noun_lemmas.append(noun_lemma)

                # Extract tool lemmas.
                tool_lemmas = []

                for tool_address in filtered_tool_addresses:
                    tool_node = dependency_parser_result.nodes[tool_address]
                    tool_lemma = tool_node["lemma"]
                    if "compound" in tool_node["deps"]:
                        tool_compound_address = tool_node["deps"]["compound"][0]
                        tool_compound_node = dependency_parser_result.nodes[tool_compound_address]
                        tool_compound_lemma = tool_compound_node["lemma"]
                        tool_lemma = tool_compound_lemma + " " + tool_lemma
                    tool_lemma = tool_lemma.replace("-", " ")
                    tool_lemmas.append(tool_lemma)

                if len(noun_lemmas) == 0:
                    noun_lemmas = ["NaN"]
                if len(tool_lemmas) == 0:
                    tool_lemmas = ["NaN"]

                # For the current verb, and for each associated noun lemma and tool lemma combination, add them to verb_noun_tool_pairs.
                for noun_lemma, tool_lemma in itertools.product(noun_lemmas, tool_lemmas):
                    verb_noun_tool_pairs.append((verb_lemma, noun_lemma, tool_lemma))

            elif current_word_pos_tag.startswith("NN"):
                noun_lemma = current_word_node["lemma"]
                if "compound" in current_word_node["deps"]:
                    noun_compound_address = current_word_node["deps"]["compound"][0]
                    noun_compound_node = dependency_parser_result.nodes[noun_compound_address]
                    noun_compound_lemma = noun_compound_node["lemma"]
                    noun_lemma = noun_compound_lemma + " " + noun_lemma

                verb_noun_tool_pairs.append(("NaN", noun_lemma, "NaN"))

    return verb_noun_tool_pairs


def get_verb_noun_tool_pairs_per_clip(
    clip_id: str,
):
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

    frame_id_verb_noun_tool_pairs_mapping = dict()
    for frame_id in range(0, num_frames, 6):
        frame_id_verb_noun_tool_pairs_mapping[frame_id] = dict()
        current_blip2_answers = blip2_vqa_answers_df[blip2_vqa_answers_df["frame_index"] == frame_id]
        for blip2_question, blip2_constant in constants.blip2_question_constant_mapping.items():
            blip2_question_index = constants.blip2_question_constant_mapping[blip2_question]
            blip2_answer = current_blip2_answers[current_blip2_answers["question"] == blip2_question]["answer"].values[0]
            frame_id_verb_noun_tool_pairs_mapping[frame_id][blip2_question_index] = (
                blip2_answer,
                get_verb_noun_tool_pairs_given_a_blip2_answer(blip2_answer, dependency_parser),
            )
    return clip_id, frame_id_verb_noun_tool_pairs_mapping


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder_path", type=str, default=os.path.join(os.environ["SCRATCH"], "ego4d_data/v2/frame_features"))
    parser.add_argument("--output_folder_path", type=str, default=os.path.join(os.environ["SCRATCH"], "ego4d_data/v2/analysis_data/dependency_parsing_results"))
    args = parser.parse_args()

    clip_ids = os.listdir(args.input_folder_path)
    dependency_parsing_results = dict(
        pqdm(
            [
                {
                    "clip_id": clip_id,
                }
                for clip_id in clip_ids
            ],
            function=get_verb_noun_tool_pairs_per_clip,
            n_jobs=24,
            argument_type="kwargs",
            exception_behaviour="immediate",
        )
    )
    os.makedirs(
        args.output_folder_path,
        exist_ok=True,
    )
    with open(
        os.path.join(
            args.output_folder_path,
            "dependency_parsing_results.pickle",
        ),
        "wb",
    ) as writer:
        pickle.dump(dependency_parsing_results, writer)
