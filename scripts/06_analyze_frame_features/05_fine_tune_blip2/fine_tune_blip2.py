import os
import cv2
import torch
import pickle
import torch.nn as nn
import numpy as np
from sklearn.utils import shuffle
from torch.nn import CrossEntropyLoss
from transformers import Blip2Processor
from transformers.models.blip_2.configuration_blip_2 import Blip2Config
from transformers.models.auto import AutoModelForCausalLM, AutoModelForSeq2SeqLM
from transformers.models.blip_2.modeling_blip_2 import (
    Blip2PreTrainedModel,
    Blip2VisionModel,
    Blip2QFormerModel,
    Blip2ForConditionalGenerationModelOutput,
)
from PIL import Image

from typing import Optional, Union, Tuple


class Blip2ForConditionalGeneration(Blip2PreTrainedModel):
    config_class = Blip2Config
    main_input_name = "pixel_values"

    def __init__(self, config: Blip2Config):
        super().__init__(config)

        self.vision_model = Blip2VisionModel(config.vision_config)

        self.query_tokens = nn.Parameter(
            torch.zeros(1, config.num_query_tokens, config.qformer_config.hidden_size)
        )
        self.qformer = Blip2QFormerModel(config.qformer_config)

        self.language_projection = nn.Linear(
            config.qformer_config.hidden_size, config.text_config.hidden_size
        )
        if config.use_decoder_only_language_model:
            language_model = AutoModelForCausalLM.from_config(config.text_config)
        else:
            language_model = AutoModelForSeq2SeqLM.from_config(config.text_config)

        # Update _tied_weights_keys using the base model used.
        if language_model._tied_weights_keys is not None:
            self._tied_weights_keys = [
                f"language_model.{k}" for k in language_model._tied_weights_keys
            ]

        self.language_model = language_model

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    def get_output_embeddings(self) -> nn.Module:
        return self.language_model.get_output_embeddings()

    def get_encoder(self):
        return self.language_model.get_encoder()

    def get_decoder(self):
        return self.language_model.get_decoder()

    def _tie_weights(self):
        if not self.config.use_decoder_only_language_model:
            self.language_model.encoder.embed_tokens = self.language_model.shared
            self.language_model.decoder.embed_tokens = self.language_model.shared

    def _preprocess_accelerate(self):
        r"""
        Some pre-processing hacks to make the model `accelerate` compatible. Check
        https://github.com/huggingface/transformers/pull/21707 for more details.
        """

        if hasattr(self.language_model, "_hf_hook"):
            self.language_model._hf_hook.io_same_device = (
                True  # For `generate` compatibility
            )

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: torch.FloatTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Blip2ForConditionalGenerationModelOutput]:
        r"""
        Returns:

        Examples:

        Image captioning (without providing a text prompt):

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import Blip2Processor, Blip2ForConditionalGeneration
        >>> import torch

        >>> device = "cuda" if torch.cuda.is_available() else "cpu"

        >>> processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        >>> model = Blip2ForConditionalGeneration.from_pretrained(
        ...     "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16
        ... )
        >>> model.to(device)  # doctest: +IGNORE_RESULT

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, return_tensors="pt").to(device, torch.float16)

        >>> generated_ids = model.generate(**inputs)
        >>> generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        >>> print(generated_text)
        two cats laying on a couch
        ```

        Visual question answering (prompt = question):

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import Blip2Processor, Blip2ForConditionalGeneration
        >>> import torch

        >>> device = "cuda" if torch.cuda.is_available() else "cpu"

        >>> processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        >>> model = Blip2ForConditionalGeneration.from_pretrained(
        ...     "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16
        ... )
        >>> model.to(device)  # doctest: +IGNORE_RESULT

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> prompt = "Question: how many cats are there? Answer:"
        >>> inputs = processor(images=image, text=prompt, return_tensors="pt").to(device, torch.float16)

        >>> generated_ids = model.generate(**inputs)
        >>> generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        >>> print(generated_text)
        two
        ```"""
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # step 1: forward the images through the vision encoder,
        # to get image embeddings of shape (batch_size, seq_len, hidden_size)
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        image_embeds = vision_outputs[0]

        # step 2: forward the query tokens through the QFormer, using the image embeddings for cross-attention
        image_attention_mask = torch.ones(
            image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device
        )

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_outputs = self.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        query_output = query_outputs[0]

        # step 3: use the language model, conditioned on the query outputs and the prompt
        language_model_inputs = self.language_projection(query_output)
        language_model_attention_mask = torch.ones(
            language_model_inputs.size()[:-1],
            dtype=torch.long,
            device=language_model_inputs.device,
        )
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat(
            [language_model_inputs, inputs_embeds.to(language_model_inputs.device)],
            dim=1,
        )

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        expected_device = language_model_attention_mask.device
        attention_mask = torch.cat(
            [language_model_attention_mask, attention_mask.to(expected_device)], dim=1
        )

        if self.config.use_decoder_only_language_model:
            outputs = self.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            logits = outputs.logits if return_dict else outputs[0]
            loss = None
            # we compute the loss here since we need to take into account the sequence length of the query embeds
            if labels is not None:
                labels = labels.to(logits.device)
                logits = logits[:, -labels.size(1) :, :]
                # Shift so that tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous().to(logits.device)

                # Flatten the tokens
                loss_fct = CrossEntropyLoss(reduction="mean")

                loss = loss_fct(
                    shift_logits.view(-1, self.config.text_config.vocab_size),
                    shift_labels.view(-1),
                )
        else:
            outputs = self.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                labels=labels,
            )
            loss = outputs.loss if return_dict else outputs[0]
            logits = outputs.logits if return_dict else outputs[1]

        if not return_dict:
            output = (logits, vision_outputs, query_outputs, outputs)
            return ((loss,) + output) if loss is not None else output

        return Blip2ForConditionalGenerationModelOutput(
            loss=loss,
            logits=logits,
            vision_outputs=vision_outputs,
            qformer_outputs=query_outputs,
            language_model_outputs=outputs,
        )

    @torch.no_grad()
    def generate(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        **generate_kwargs,
    ) -> torch.LongTensor:
        """
        Overrides `generate` function to be able to use the model as a conditional generator.

        Args:
            pixel_values (`torch.FloatTensor` of shape (batch_size, num_channels, height, width)):
                Input images to be processed.
            input_ids (`torch.LongTensor` of shape (batch_size, sequence_length), *optional*):
                The sequence used as a prompt for the generation.
            attention_mask (`torch.LongTensor` of shape (batch_size, sequence_length), *optional*):
                Mask to avoid performing attention on padding token indices

        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        if hasattr(self, "hf_device_map"):
            # preprocess for `accelerate`
            self._preprocess_accelerate()

        batch_size = pixel_values.shape[0]
        image_embeds = self.vision_model(
            pixel_values, return_dict=True
        ).last_hidden_state
        image_attention_mask = torch.ones(
            image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device
        )

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_outputs = self.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            return_dict=True,
        )
        query_output = query_outputs.last_hidden_state

        language_model_inputs = self.language_projection(query_output)
        language_attention_mask = torch.ones(
            language_model_inputs.size()[:-1],
            dtype=torch.long,
            device=language_model_inputs.device,
        )
        if input_ids is None:
            input_ids = (
                torch.LongTensor([[self.config.text_config.bos_token_id]])
                .repeat(batch_size, 1)
                .to(image_embeds.device)
            )
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        attention_mask = torch.cat(
            [
                language_attention_mask,
                attention_mask.to(language_attention_mask.device),
            ],
            dim=1,
        )

        # concatenate query embeddings with prompt embeddings
        inputs_embeds = self.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat(
            [language_model_inputs, inputs_embeds.to(language_model_inputs.device)],
            dim=1,
        )

        outputs = self.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **generate_kwargs,
        )

        return outputs


class BLIP2VQAFrameFeatureExtractor(object):
    def __init__(self):
        self.processor = Blip2Processor.from_pretrained(
            os.path.join(os.environ["SCRATCH"], "mq_libs/blip2")
        )
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            os.path.join(os.environ["SCRATCH"], "mq_libs/blip2"),
            torch_dtype=torch.float16,
        )
        self.model.to("cuda:3")


blip2_vqa_frame_feature_extractor = BLIP2VQAFrameFeatureExtractor()

with open(
    os.path.join(
        os.environ["SCRATCH"],
        "ego4d_data/v2/analysis_data/ground_truth_labels/ground_truth_labels.pickle",
    ),
    "rb",
) as reader:
    ground_truth_label_indices = pickle.load(reader)
    clip_ids = shuffle(list(ground_truth_label_indices.keys()))


def get_random_frame_and_corresponding_label():
    random_clip_id_index = np.random.randint(low=0, high=len(clip_ids))
    random_clip_id = clip_ids[random_clip_id_index]
    random_cap = cv2.VideoCapture(
        os.path.join(
            os.environ["SCRATCH"], "ego4d_data/v2/clips", random_clip_id + ".mp4"
        )
    )
    number_of_frames = random_cap.get(cv2.CAP_PROP_FRAME_COUNT)
    random_frame_id = np.random.randint(low=0, high=number_of_frames)
    random_cap.set(cv2.CAP_PROP_POS_FRAMES, random_frame_id - 1)
    success, random_frame = random_cap.read()
    corresponding_label = ground_truth_label_indices[random_clip_id][random_frame_id]
    return random_frame, corresponding_label


with torch.no_grad():
    random_frame, corresponding_label = get_random_frame_and_corresponding_label()
    preprocessed_frames_batch_dict = blip2_vqa_frame_feature_extractor.processor(
        images=[Image.fromarray(random_frame[:, :, ::-1])],
        text=["Question: " + "What is the person in this image doing?" + " Answer:"],
        return_tensors="pt",
    ).to("cuda:3", torch.float16)

    generated_ids = blip2_vqa_frame_feature_extractor.model.generate(
        **preprocessed_frames_batch_dict
    )
