import os
import json
from ast import literal_eval
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset
from torch.nn import functional as F

from .datasets import register_dataset
from .data_utils import truncate_feats
from ..utils import remove_duplicate_annotations


@register_dataset("ego4d")
class Ego4dDataset(Dataset):
    def __init__(
        self,
        is_training,  # if in training mode
        split,  # split, a tuple/list allowing concat of subsets
        video_feat_folder,  # folder for features
        frame_feat_names,
        json_file,  # json file for annotations
        feat_stride,  # temporal stride of the feats
        num_frames,  # number of frames for each feat
        default_fps,  # default fps
        downsample_rate,  # downsample rate for feats
        max_seq_len,  # maximum sequence length during training
        trunc_thresh,  # threshold for truncate an action segment
        crop_ratio,  # a tuple (e.g., (0.9, 1.0)) for random cropping
        input_dim,  # input feat dim
        num_classes,  # number of action categories
        file_prefix,  # feature file prefix if any
        file_ext,  # feature file extension if any
        force_upsampling,  # force to upsample to max_seq_len
    ):
        # file path
        assert os.path.exists(json_file)
        assert isinstance(split, tuple) or isinstance(split, list)
        assert crop_ratio == None or len(crop_ratio) == 2
        self.video_feat_folder = video_feat_folder
        self.frame_feat_names = frame_feat_names
        # self.use_hdf5 = '.hdf5' in video_feat_folder
        if file_prefix is not None:
            self.file_prefix = file_prefix
        else:
            self.file_prefix = ""
        self.file_ext = ".pt"
        self.json_file = json_file

        # anet uses fixed length features, make sure there is no downsampling
        self.force_upsampling = force_upsampling

        # split / training mode
        self.split = split
        self.is_training = is_training

        # features meta info
        self.feat_stride = feat_stride
        self.num_frames = num_frames
        self.input_feat_dim = input_dim
        self.default_fps = default_fps
        self.downsample_rate = downsample_rate
        self.max_seq_len = max_seq_len
        self.temporal_scale = max_seq_len
        self.trunc_thresh = trunc_thresh
        self.num_classes = num_classes
        self.label_dict = None
        self.crop_ratio = crop_ratio

        # load database and select the subset
        dict_db, label_dict = self._load_json_db(self.json_file)

        # proposal vs action categories
        assert (num_classes == 1) or (len(label_dict) == num_classes)
        self.data_list = dict_db
        self.label_dict = label_dict

        # dataset specific attributes
        self.db_attributes = {
            "dataset_name": "ego4d moment query 1.3",
            "tiou_thresholds": np.linspace(0.1, 0.5, 5),
            "empty_label_ids": [],
        }
        # self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    def get_attributes(self):
        return self.db_attributes

    def _load_json_db(self, json_file):
        # load database and select the subset
        with open(json_file, "r") as fid:
            json_data = json.load(fid)
        json_db = json_data

        # if label_dict is not available
        if self.label_dict is None:
            label_dict = {}
            for key, value in json_db.items():
                for act in value["annotations"]:
                    label_dict[act["label"]] = act["label_id"]
        # fill in the db (immutable afterwards)
        dict_db = tuple()
        for key, value in json_db.items():
            # skip the video if not in the split
            if value["subset"].lower() not in self.split:
                continue

            # get fps if available
            if self.default_fps is not None:
                fps = self.default_fps
            elif "fps" in value:
                fps = value["fps"]
            else:
                assert False, "Unknown video FPS."
            duration = value["duration"]
            segmentation_labels = torch.zeros(
                (int(duration), self.num_classes), dtype=torch.float
            )

            # get annotations if available
            if ("annotations" in value) and (len(value["annotations"]) > 0):
                valid_acts = remove_duplicate_annotations(value["annotations"])
                num_acts = len(valid_acts)
                segments = np.zeros([num_acts, 2], dtype=np.float32)
                labels = np.zeros(
                    [
                        num_acts,
                    ],
                    dtype=np.int64,
                )
                for idx, act in enumerate(valid_acts):
                    segments[idx][0] = act["segment"][0]
                    segments[idx][1] = act["segment"][1]
                    if self.num_classes == 1:
                        labels[idx] = 0
                    else:
                        labels[idx] = label_dict[act["label"]]

                    for frame in range(int(duration)):
                        if frame > act["segment"][0] and frame < act["segment"][1]:
                            segmentation_labels[frame, int(act["label_id"])] = 1
            else:
                segments = None
                labels = None
            dict_db += (
                {
                    "id": key,
                    "fps": fps,
                    "duration": duration,
                    "segments": segments,
                    "labels": labels,
                    "parent_video_id": value["video_id"],
                    "parent_start_sec": value["parent_start_sec"],
                    "parent_end_sec": value["parent_end_sec"],
                    #  'prompt': value['prompt'],
                    #  'negative_prompt': value['negative_prompt'],
                    "segmentation_labels": segmentation_labels,
                },
            )
        return dict_db, label_dict

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        # directly return a (truncated) data point (so it is very fast!)
        # auto batching will be disabled in the subsequent dataloader
        # instead the model will need to decide how to batch / preporcess the data
        clip_info = self.data_list[idx]
        video_name = clip_info["parent_video_id"]
        clip_name = clip_info["id"]
        segmentation_labels = clip_info["segmentation_labels"]

        # self.input_feat_dim = 3840          # if add egovlp

        # video_data = torch.zeros(self.input_feat_dim, self.temporal_scale)
        # win_data = v_data[:, clip_start: clip_end+1]
        # num_frms = min(win_data.shape[-1], self.temporal_scale)
        # video_data[:, :num_frms] = win_data[:, :num_frms]
        # feats = video_data[:, :num_frms]
        # feats = feats.permute(1,0)      # [t,c]

        # egovlp
        if isinstance(self.video_feat_folder, str):
            filename = os.path.join(
                self.video_feat_folder, self.file_prefix + clip_name + self.file_ext
            )
            feats = torch.load(filename)
            # case 1: variable length features for training
            if self.feat_stride > 0 and (not self.force_upsampling):
                # var length features
                feat_stride, num_frames = self.feat_stride, self.num_frames
                # only apply down sampling here
                if self.downsample_rate > 1:
                    feats = feats[:: self.downsample_rate, :]
                    feat_stride = self.feat_stride * self.downsample_rate
            # case 2: variable length features for input, yet resized for training
            elif (
                self.feat_stride > 0 and self.force_upsampling
            ):  # activitynet 会upsample到fixed length
                feat_stride = (
                    float((feats.shape[0] - 1) * self.feat_stride + self.num_frames)
                    / self.max_seq_len
                )
                # center the features
                num_frames = feat_stride
            # case 3: fixed length features for input
            else:
                # deal with fixed length feature, recompute feat_stride, num_frames
                seq_len = feats.shape[0]
                assert seq_len <= self.max_seq_len
                if self.force_upsampling:
                    # reset to max_seq_len
                    seq_len = self.max_seq_len
                feat_stride = clip_info["duration"] * clip_info["fps"] / seq_len
                # center the features
                num_frames = feat_stride

            # T x C -> C x T
            feats = feats.permute(1, 0)

            # resize the features if needed
            if (feats.shape[-1] != self.max_seq_len) and self.force_upsampling:
                resize_feats = F.interpolate(
                    feats.unsqueeze(0),
                    size=self.max_seq_len,
                    mode="linear",
                    align_corners=False,
                )
                segmentation_labels = (
                    F.interpolate(
                        segmentation_labels.unsqueeze(0).unsqueeze(0),
                        size=(self.max_seq_len, self.num_classes),
                        mode="nearest",
                    )
                    .squeeze(0)
                    .squeeze(0)
                )
                feats = resize_feats.squeeze(0)  # [d,192]       upsample到一个fixed length
        else:
            all_features = []
            for f_t in self.video_feat_folder:
                filename = os.path.join(
                    f_t, self.file_prefix + clip_name + self.file_ext
                )
                feats = torch.load(filename)
                # case 1: variable length features for training
                if self.feat_stride > 0 and (not self.force_upsampling):
                    # var length features
                    feat_stride, num_frames = self.feat_stride, self.num_frames
                    # only apply down sampling here
                    if self.downsample_rate > 1:
                        feats = feats[:: self.downsample_rate, :]
                        feat_stride = self.feat_stride * self.downsample_rate
                # case 2: variable length features for input, yet resized for training
                elif (
                    self.feat_stride > 0 and self.force_upsampling
                ):  # activitynet 会upsample到fixed length
                    feat_stride = (
                        float((feats.shape[0] - 1) * self.feat_stride + self.num_frames)
                        / self.max_seq_len
                    )
                    # center the features
                    num_frames = feat_stride
                # case 3: fixed length features for input
                else:
                    # deal with fixed length feature, recompute feat_stride, num_frames
                    seq_len = feats.shape[0]
                    assert seq_len <= self.max_seq_len
                    if self.force_upsampling:
                        # reset to max_seq_len
                        seq_len = self.max_seq_len
                    feat_stride = clip_info["duration"] * clip_info["fps"] / seq_len
                    # center the features
                    num_frames = feat_stride

                # T x C -> C x T
                feats = feats.permute(1, 0)

                # resize the features if needed
                if (feats.shape[-1] != self.max_seq_len) and self.force_upsampling:
                    resize_feats = F.interpolate(
                        feats.unsqueeze(0),
                        size=self.max_seq_len,
                        mode="linear",
                        align_corners=False,
                    )
                    segmentation_labels = (
                        F.interpolate(
                            segmentation_labels.unsqueeze(0).unsqueeze(0),
                            size=(self.max_seq_len, self.num_classes),
                            mode="nearest",
                        )
                        .squeeze(0)
                        .squeeze(0)
                    )
                    feats = resize_feats.squeeze(
                        0
                    )  # [d,192]       upsample到一个fixed length

                all_features.append(feats)
            feats = torch.cat(all_features, dim=0)

        # convert time stamp (in second) into temporal feature grids
        # ok to have small negative values here
        if clip_info["segments"] is not None:
            segments = torch.from_numpy(
                (clip_info["segments"] * clip_info["fps"] - 0.5 * num_frames)
                / feat_stride
            )  # 到frame数
            labels = torch.from_numpy(clip_info["labels"])
            # for activity net, we have a few videos with a bunch of missing frames
            # here is a quick fix for training
            if self.is_training:
                vid_len = feats.shape[1] + 0.5 * num_frames / feat_stride
                valid_seg_list, valid_label_list = [], []
                for seg, label in zip(segments, labels):
                    if seg[0] >= vid_len:
                        # skip an action outside of the feature map
                        continue
                    # skip an action that is mostly outside of the feature map
                    ratio = (min(seg[1].item(), vid_len) - seg[0].item()) / (
                        seg[1].item() - seg[0].item()
                    )
                    if ratio >= self.trunc_thresh:
                        valid_seg_list.append(seg.clamp(max=vid_len, min=0))
                        # some weird bug here if not converting to size 1 tensor
                        valid_label_list.append(label.view(1))
                segments = torch.stack(valid_seg_list, dim=0)
                labels = torch.cat(valid_label_list)
        else:
            segments, labels = None, None

        frame_indices = [
            int((30 * i * clip_info["duration"] / 1024.0) // 6 * 6)
            for i in range(1, 1025)
        ]

        frame_feats = [[]] * len(self.frame_feat_names)

        blip2_vqa_feature_file_names = sorted(
            [
                file_name
                for file_name in os.listdir(
                    os.path.join(
                        os.environ["SCRATCH"],
                        "ego4d_data/v2/postprocessed_frame_features",
                        clip_name,
                        "caption_sbert_embedding",
                    )
                )
                if file_name.startswith("blip2_vqa")
            ]
        )

        for current_blip2_vqa_feature_file_name in blip2_vqa_feature_file_names:
            for index, frame_feat_name in enumerate(self.frame_feat_names):
                if frame_feat_name == "encoder_output":
                    current_df = pd.read_csv(
                        os.path.join(
                            os.environ["SCRATCH"],
                            "ego4d_data/v2/postprocessed_frame_features",
                            clip_name,
                            "encoder_output",
                            current_blip2_vqa_feature_file_name,
                        ),
                        sep="\t",
                    )
                    current_df = current_df[
                        current_df["frame_index"].isin(frame_indices)
                    ]
                    for encoder_output in current_df["encoder_output"].values:
                        frame_feats[index].append(
                            np.array([literal_eval(encoder_output)])  # (1, 94208)
                        )

                elif frame_feat_name == "caption_sbert_embedding":
                    current_df = pd.read_csv(
                        os.path.join(
                            os.environ["SCRATCH"],
                            "ego4d_data/v2/postprocessed_frame_features",
                            clip_name,
                            "caption_sbert_embedding",
                            current_blip2_vqa_feature_file_name,
                        ),
                        sep="\t",
                    )
                    current_df = current_df[
                        current_df["frame_index"].isin(frame_indices)
                    ]
                    for encoder_output in current_df["caption_sbert_embedding"].values:
                        frame_feats[index].append(
                            np.array([literal_eval(encoder_output)])  # (1, 768)
                        )

        print("len(frame_feats[0]):", len(frame_feats[0]))
        print("len(frame_feats[1]):", len(frame_feats[1]))
        print("[a.shape for a in frame_feats[0]]:", [a.shape for a in frame_feats[0]])
        print("[b.shape for b in frame_feats[1]]:", [b.shape for b in frame_feats[1]])

        for i in range(len(self.frame_feat_names)):
            frame_feats[i] = np.vstack(
                frame_feats[i]
            ).transpose()  # Now each frame_feats[i] has shape (C, T)

        if len(self.frame_feat_names) > 0:
            frame_feats = np.vstack(frame_feats)
            frame_feats = torch.tensor(frame_feats)
            print("Before feats.shape:", feats.shape)
            print("frame_feats.shape:", frame_feats.shape)
            feats = torch.cat([feats, frame_feats], dim=0)
            print("After feats.shape:", feats.shape)
            raise Exception("asd")

        # return a data dict
        data_dict = {
            "video_id": clip_info["id"],
            "feats": feats,  # C x T
            "segments": segments,  # N x 2
            "labels": labels,  # N
            "fps": clip_info["fps"],
            "duration": clip_info["duration"],
            "feat_stride": feat_stride,
            "feat_num_frames": num_frames,
            "segmentation_labels": segmentation_labels,
        }

        # no truncation is needed
        # truncate the features during training             truncate一下，并且保证有action在里面（iou大于threshold）
        if self.is_training and (segments is not None):
            data_dict = truncate_feats(
                data_dict, self.max_seq_len, self.trunc_thresh, self.crop_ratio
            )

        return data_dict
