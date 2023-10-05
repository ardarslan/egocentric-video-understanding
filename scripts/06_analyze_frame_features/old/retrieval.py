import os
import pickle
import torch
import argparse
import numpy as np
from copy import deepcopy
from torch.utils.data import DataLoader
from sklearn.metrics import balanced_accuracy_score, f1_score

from dataset import Dataset
from model import RetrievalModel

from utils import (
    get_analysis_data_file_name_wo_ext_analysis_data_mapping,
    save_evaluation_metrics,
    save_validation_predictions,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Argument parser")
    parser.add_argument(
        "--frame_label_embedder",
        default="word2vec",
        choices=[
            "word2vec",
            "glove",
            "sentence_transformer",
            "universal_sentence_encoder",
            "one_hot",
        ],
        type=str,
    )
    parser.add_argument(
        "--train_blip2_answer_word_weight_type",
        choices=["idf", "uniform"],
        default="idf",
        type=str,
    )
    parser.add_argument(
        "--annotations_json_file_path",
        default=f"{os.environ['CODE']}/scripts/07_reproduce_baseline_results/data/ego4d/ego4d_clip_annotations_v3.json",
        type=str,
    )
    parser.add_argument("--batch_size", default=16, type=int)
    args = parser.parse_args()

    analysis_data_file_name_wo_ext_analysis_data_mapping_file_path = os.path.join(
        os.environ["SCRATCH"],
        "ego4d_data/v2",
        "analysis_data",
        args.frame_embedder,
        args.train_blip2_answer_word_weight_type,
        "analysis_data_file_name_wo_ext_analysis_data_mapping.pickle",
    )
    if os.path.exists(analysis_data_file_name_wo_ext_analysis_data_mapping_file_path):
        with open(
            analysis_data_file_name_wo_ext_analysis_data_mapping_file_path,
            "rb",
        ) as reader:
            analysis_data_file_name_wo_ext_analysis_data_mapping = pickle.load(reader)
    else:
        analysis_data_file_name_wo_ext_analysis_data_mapping = get_analysis_data_file_name_wo_ext_analysis_data_mapping(
            args=args,
            analysis_data_file_name_wo_ext_analysis_data_mapping_file_path=analysis_data_file_name_wo_ext_analysis_data_mapping_file_path,
        )

    train_labels = analysis_data_file_name_wo_ext_analysis_data_mapping["train_labels"]
    train_X = analysis_data_file_name_wo_ext_analysis_data_mapping["train_X"]
    train_y = analysis_data_file_name_wo_ext_analysis_data_mapping["train_y"]
    train_clip_ids = analysis_data_file_name_wo_ext_analysis_data_mapping[
        "train_clip_ids"
    ]
    train_frame_ids = analysis_data_file_name_wo_ext_analysis_data_mapping[
        "train_frame_ids"
    ]
    val_X = analysis_data_file_name_wo_ext_analysis_data_mapping["val_X"]
    val_y = analysis_data_file_name_wo_ext_analysis_data_mapping["val_y"]
    val_clip_ids = analysis_data_file_name_wo_ext_analysis_data_mapping["val_clip_ids"]
    val_frame_ids = analysis_data_file_name_wo_ext_analysis_data_mapping[
        "val_frame_ids"
    ]
    test_X = analysis_data_file_name_wo_ext_analysis_data_mapping["test_X"]
    test_clip_ids = analysis_data_file_name_wo_ext_analysis_data_mapping[
        "test_clip_ids"
    ]
    test_frame_ids = analysis_data_file_name_wo_ext_analysis_data_mapping[
        "test_frame_ids"
    ]

    model = RetrievalModel(train_labels=train_labels)

    train_dataset = Dataset(
        X=train_X, y=train_y, clip_ids=train_clip_ids, frame_ids=train_frame_ids
    )
    train_data_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )

    val_dataset = Dataset(
        X=val_X, y=val_y, clip_ids=val_clip_ids, frame_ids=val_frame_ids
    )
    val_data_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    test_dataset = Dataset(
        X=test_X, y=None, clip_ids=test_clip_ids, frame_ids=test_frame_ids
    )
    test_data_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False
    )

    for batch in train_data_loader:
        yhat = model(batch["X"].to(args.device))
        y_indices = batch["y"].to(args.device)
        train_ys.append(batch["y"])
        train_yhats.append(yhat.detach().cpu())
        loss = torch.nn.BCEWithLogitsLoss()(yhat, y)
        loss.backward()
        optimizer.step()
        train_ys = torch.vstack(train_ys)
        train_yhats = torch.vstack(train_yhats)
        train_ys_argmaxed = np.argmax(train_ys.numpy(), axis=1)
        train_yhats_argmaxed = np.argmax(train_yhats.numpy(), axis=1)
        train_loss = torch.nn.BCEWithLogitsLoss()(train_yhats, train_ys).item()
        train_bmac = balanced_accuracy_score(train_ys_argmaxed, train_yhats_argmaxed)
        train_f1 = f1_score(train_ys_argmaxed, train_yhats_argmaxed, average="weighted")

        evaluation_metrics["train_loss_values"].append(train_loss)
        evaluation_metrics["train_bmac_values"].append(train_bmac)
        evaluation_metrics["train_f1_values"].append(train_f1)

        val_ys = []
        val_yhats = []
        model.eval()
        for batch in val_data_loader:
            with torch.no_grad():
                yhat = model(batch["X"].to(args.device))
                y = batch["y"].to(args.device)
                val_ys.append(batch["y"])
                val_yhats.append(yhat.detach().cpu())
        val_ys = torch.vstack(val_ys)
        val_yhats = torch.vstack(val_yhats)
        val_loss = torch.nn.BCEWithLogitsLoss()(val_yhats, val_ys).item()
        val_ys_argmaxed = np.argmax(val_ys.numpy(), axis=1)
        val_yhats_argmaxed = np.argmax(val_yhats.numpy(), axis=1)
        val_bmac = balanced_accuracy_score(val_ys_argmaxed, val_yhats_argmaxed)
        val_f1 = f1_score(val_ys_argmaxed, val_yhats_argmaxed, average="weighted")

        evaluation_metrics["val_loss_values"].append(val_loss)
        evaluation_metrics["val_bmac_values"].append(val_bmac)
        evaluation_metrics["val_f1_values"].append(val_f1)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = deepcopy(model)

        print(
            f"Epoch: {str(epoch).zfill(2)} | Train BMAC: {np.round(train_bmac, 2)} | Train F1: {np.round(train_f1, 2)} | Train Loss: {np.round(train_loss, 2)} | Val BMAC: {np.round(val_bmac, 2)} | Val F1: {np.round(val_f1, 2)} | Val Loss: {np.round(val_loss, 2)} \n"
        )
        scheduler.step(val_loss)

        save_evaluation_metrics(args=args, evaluation_metrics=evaluation_metrics)

    val_ys = []
    val_yhats = []
    val_clip_ids = []
    val_frame_ids = []
    best_model.eval()
    for batch in val_data_loader:
        with torch.no_grad():
            yhat = best_model(batch["X"].to(args.device))
            y = batch["y"].to(args.device)
            val_ys.append(batch["y"])
            val_yhats.append(yhat.detach().cpu())
            val_clip_ids.extend(batch["clip_id"])
            val_frame_ids.extend(batch["frame_id"])

    val_ys = torch.vstack(val_ys)
    val_yhats = torch.vstack(val_yhats)
    val_loss = torch.nn.BCEWithLogitsLoss()(val_yhats, val_ys).item()
    val_ys_argmaxed = np.argmax(val_ys.numpy(), axis=1)
    val_yhats_argmaxed = np.argmax(val_yhats.numpy(), axis=1)

    save_validation_predictions(
        args=args,
        val_clip_ids=val_clip_ids,
        val_frame_ids=val_frame_ids,
        val_ys=val_ys_argmaxed,
        val_yhats=val_yhats_argmaxed,
    )
