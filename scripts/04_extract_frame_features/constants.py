BLIP2_ANSWER_LABEL_VERB_NOUN_TOOL = 0
BLIP2_ANSWER_LABEL_VERB_NOUN = 1
BLIP2_ANSWER_LABEL_VERB_TOOL = 2
BLIP2_ANSWER_LABEL_VERB = 3
BLIP2_ANSWER_LABEL_NOUN = 4
BLIP2_VERB_NOUN_TOOL_LABEL_VERB_NOUN_TOOL = 5
BLIP2_VERB_NOUN_LABEL_VERB_NOUN = 6
BLIP2_VERB_TOOL_LABEL_VERB_TOOL = 7
BLIP2_VERB_LABEL_VERB = 8
BLIP2_NOUN_LABEL_NOUN = 9
BACKGROUND_MATCH = 10
BLIP2_ALL_LABEL_ALL = 11
ASL_MATCH = 12


query_score_type_constant_mapping = {
    "max_of_all": BLIP2_ALL_LABEL_ALL,
    "max_of_blip2_answer_label_verb_noun_tool": BLIP2_ANSWER_LABEL_VERB_NOUN_TOOL,
    "max_of_blip2_answer_label_verb_noun": BLIP2_ANSWER_LABEL_VERB_NOUN,
    "max_of_blip2_answer_label_verb_tool": BLIP2_ANSWER_LABEL_VERB_TOOL,
    "max_of_blip2_answer_label_noun": BLIP2_ANSWER_LABEL_NOUN,
    "max_of_blip2_answer_label_verb": BLIP2_ANSWER_LABEL_VERB,
    "max_of_blip2_verb_noun_tool_label_verb_noun_tool": BLIP2_VERB_NOUN_TOOL_LABEL_VERB_NOUN_TOOL,
    "max_of_blip2_verb_noun_label_verb_noun": BLIP2_VERB_NOUN_LABEL_VERB_NOUN,
    "max_of_blip2_verb_tool_label_verb_tool": BLIP2_VERB_TOOL_LABEL_VERB_TOOL,
    "max_of_blip2_verb_label_verb": BLIP2_VERB_LABEL_VERB,
    "max_of_blip2_noun_label_noun": BLIP2_NOUN_LABEL_NOUN,
}

blip2_dependency_parsing_feature_label_dependency_parsing_feature_mapping = {
    "blip2_answer_label_verb_noun_tool": BLIP2_ANSWER_LABEL_VERB_NOUN_TOOL,
    "blip2_answer_label_verb_noun": BLIP2_ANSWER_LABEL_VERB_NOUN,
    "blip2_answer_label_verb_tool": BLIP2_ANSWER_LABEL_VERB_TOOL,
    "blip2_answer_label_verb": BLIP2_ANSWER_LABEL_VERB,
    "blip2_answer_label_noun": BLIP2_ANSWER_LABEL_NOUN,
    "blip2_verb_noun_tool_label_verb_noun_tool": BLIP2_VERB_NOUN_TOOL_LABEL_VERB_NOUN_TOOL,
    "blip2_verb_noun_label_verb_noun": BLIP2_VERB_NOUN_LABEL_VERB_NOUN,
    "blip2_verb_label_verb": BLIP2_VERB_LABEL_VERB,
    "blip2_noun_label_noun": BLIP2_NOUN_LABEL_NOUN,
    "blip2_verb_tool_label_verb_tool": BLIP2_VERB_TOOL_LABEL_VERB_TOOL,
}

question_constant_mapping = {
    "What does the image describe?": 0,
    "What is the person in this picture doing?": 1,
    "What is happening in this picture?": 2,
    # "What are the objects that the person is interacting with in this picture?": 3,
    # "What action the person in this picture is doing?": 4,
    # "This picture was taken with a camera mounted on the head of a person. What is this person doing?": 5,
    # "What are the objects that the person is holding in his hands?": 6,
    "asl": 6,
}
