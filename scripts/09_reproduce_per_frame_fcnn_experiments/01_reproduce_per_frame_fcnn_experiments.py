import os
import numpy as np
import pandas as pd

os.environ["CODE"] = "/home/aarslan/mq"
os.environ["SLURM_CONF"] = "/home/sladmcvl/slurm/slurm.conf"
os.environ["SCRATCH"] = "/srv/beegfs-benderdata/scratch/aarslan_data/data"
os.environ["CUDA_HOME"] = "/usr/lib/nvidia-cuda-toolkit"

import sys

sys.path.append("../08_reproduce_mq_experiments/")

import torch
import argparse
from tqdm import tqdm

from libs.core import load_config
from libs.utils import fix_random_seed
from libs.datasets import make_dataset, make_data_loader

from model import MLP
from sklearn.metrics import f1_score, precision_score, recall_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_nonlinear_layers", type=int, default=0)
    parser.add_argument(
        "--config",
        type=str,
        default="/home/aarslan/mq/scripts/08_reproduce_mq_experiments/configs/proposed_features_v6.yaml",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg["loader"]["num_workers"] = 0
    cfg["dataset_name"] = "ego4d_per_frame"

    for i in range(len(cfg["dataset"]["video_feat_folder"])):
        cfg["dataset"]["video_feat_folder"][i] = os.path.join(
            os.environ["SCRATCH"], cfg["dataset"]["video_feat_folder"][i]
        )

    rng_generator = fix_random_seed(cfg["init_rand_seed"], include_cuda=True)

    train_dataset = make_dataset(
        cfg["dataset_name"], True, cfg["train_split"], **cfg["dataset"]
    )
    train_data_loader = make_data_loader(
        train_dataset, True, rng_generator, **cfg["loader"]
    )

    val_dataset = make_dataset(
        cfg["dataset_name"], False, cfg["val_split"], **cfg["dataset"]
    )
    val_data_loader = make_data_loader(
        val_dataset, False, None, 1, cfg["loader"]["num_workers"]
    )

    model = MLP(
        input_dimension=sum(cfg["dataset"]["input_dim"]),
        output_dimension=111,
        hidden_dimension=int((sum(cfg["dataset"]["input_dim"]) + 111) / 2),
        num_nonlinear_layers=args.num_nonlinear_layers,
    )

    model.to(args.device)

    optimizer = torch.optim.AdamW(params=model.parameters())

    criterion = torch.nn.BCEWithLogitsLoss()

    for epoch in range(1, args.num_epochs + 1):
        total_train_loss = 0.0
        total_train_sample_count = 0.0

        for batch in tqdm(train_data_loader):
            optimizer.zero_grad()
            yhat = model(batch["feats"])
            y = model(batch["segmentation_labels"])
            loss = criterion(yhat, y)
            loss.backward()
            optimizer.step()
            total_train_loss += (
                float(loss.detach().cpu().numpy()) * batch["feats"].shape[0]
            )
            total_train_sample_count += float(batch["feats"].shape[0])

        total_val_loss = 0.0
        total_val_sample_count = 0.0
        with torch.no_grad():
            for batch in tqdm(val_data_loader):
                yhat = model(batch["feats"])
                y = model(batch["segmentation_labels"])
                loss = criterion(yhat, y)
                total_val_loss += float(loss.cpu().numpy()) * batch["feats"].shape[0]
                total_val_sample_count += float(batch["feats"].shape[0])

        current_train_loss = total_train_loss / total_train_sample_count
        current_val_loss = total_val_loss / total_val_sample_count

        print(
            f"Epoch: {epoch}, Train Loss: {np.round(current_train_loss)}, Val Loss: {np.round(current_val_loss)}"
        )

    thresholds = [0.2, 0.4, 0.6, 0.8, 1.0, "max"]
    threshold_yhats_mapping = dict()
    val_y_wbs = []
    val_y_nbs = []
    val_y_bs = []

    for threshold in thresholds:
        threshold_yhats_mapping[threshold] = {
            "val_yhat_wb_thresholdeds": [],
            "val_yhat_nb_thresholdeds": [],
            "val_yhat_b_thresholdeds": [],
        }

    for batch in val_data_loader:
        current_val_yhat_wbs = model(batch["feats"])
        current_val_y_wbs = batch["segmentation_labels"]
        for current_val_yhat_wb, current_val_y_wb in zip(
            current_val_yhat_wbs, current_val_y_wbs
        ):
            for threshold in thresholds:
                if threshold == "max":
                    current_val_yhat_wb_thresholded = np.zeros(len(current_val_yhat_wb))
                    current_val_yhat_wb_thresholded[np.argmax(current_val_yhat_wb)] = 1
                    current_val_yhat_nb_thresholded = current_val_yhat_wb_thresholded[
                        :-1
                    ]
                    current_val_yhat_b_thresholded = current_val_yhat_wb_thresholded[-1]
                else:
                    current_val_yhat_wb_thresholded = current_val_yhat_wb >= threshold
                    current_val_yhat_nb_thresholded = current_val_yhat_wb_thresholded[
                        :-1
                    ]
                    current_val_yhat_b_thresholded = current_val_yhat_wb_thresholded[-1]
                threshold_yhats_mapping[threshold]["val_yhat_wb_thresholdeds"].append(
                    current_val_yhat_wb_thresholded
                )
                threshold_yhats_mapping[threshold]["val_yhat_nb_thresholdeds"].append(
                    current_val_yhat_nb_thresholded
                )
                threshold_yhats_mapping[threshold]["val_yhat_b_thresholdeds"].append(
                    current_val_yhat_b_thresholded
                )
                val_y_wbs.append(current_val_y_wb)
                val_y_nbs.append(current_val_y_wb[:-1])
                val_y_bs.append(current_val_y_wb[-1])

    evaluation_metrics = []
    for threshold in thresholds:
        wwb_f1_score = f1_score(
            y_true=val_y_wbs,
            y_pred=threshold_yhats_mapping[threshold]["val_yhat_wb_thresholdeds"],
            average="weighted",
            zero_division=0,
        )
        wwb_precision_score = precision_score(
            y_true=val_y_wbs,
            y_pred=threshold_yhats_mapping[threshold]["val_yhat_wb_thresholdeds"],
            average="weighted",
            zero_division=0,
        )
        wwb_recall_score = recall_score(
            y_true=val_y_wbs,
            y_pred=threshold_yhats_mapping[threshold]["val_yhat_wb_thresholdeds"],
            average="weighted",
            zero_division=0,
        )

        wnb_f1_score = f1_score(
            y_true=val_y_nbs,
            y_pred=threshold_yhats_mapping[threshold]["val_yhat_nb_thresholdeds"],
            average="weighted",
            zero_division=0,
        )
        wnb_precision_score = precision_score(
            y_true=val_y_nbs,
            y_pred=threshold_yhats_mapping[threshold]["val_yhat_nb_thresholdeds"],
            average="weighted",
            zero_division=0,
        )
        wnb_recall_score = recall_score(
            y_true=val_y_nbs,
            y_pred=threshold_yhats_mapping[threshold]["val_yhat_nb_thresholdeds"],
            average="weighted",
            zero_division=0,
        )

        b_f1_score = f1_score(
            y_true=val_y_bs,
            y_pred=threshold_yhats_mapping[threshold]["val_yhat_b_thresholdeds"],
            zero_division=0,
        )
        b_precision_score = precision_score(
            y_true=val_y_bs,
            y_pred=threshold_yhats_mapping[threshold]["val_yhat_b_thresholdeds"],
            zero_division=0,
        )
        b_recall_score = recall_score(
            y_true=val_y_bs,
            y_pred=threshold_yhats_mapping[threshold]["val_yhat_b_thresholdeds"],
            zero_division=0,
        )

        evaluation_metrics.append(
            (
                threshold,
                wwb_f1_score,
                wwb_precision_score,
                wwb_recall_score,
                wnb_f1_score,
                wnb_precision_score,
                wnb_recall_score,
                b_f1_score,
                b_precision_score,
                b_recall_score,
            )
        )
    evaluation_metrics_df = pd.DataFrame(
        data=evaluation_metrics,
        columns=[
            "threshold",
            "wwb_f1_score",
            "wwb_precision_score",
            "wwb_recall_score",
            "wnb_f1_score",
            "wnb_precision_score",
            "wnb_recall_score",
            "b_f1_score",
            "b_precision_score",
            "b_recall_score",
        ],
    )
    evaluation_metrics_df.to_csv("evaluation_metrics.tsv", sep="\t", index=False)
