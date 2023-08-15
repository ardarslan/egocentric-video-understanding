# %%

import argparse
import cv2
import multiprocessing as mp
from multiprocessing.pool import Pool
import numpy as np
import os
from os.path import dirname, isdir, isfile, join
import pickle
from PIL import Image
import sys
import time
import ssl
import sys
import torch
from tqdm import tqdm
import zipfile

sys.path.append(dirname(dirname(__file__)))

from data_handling.specific.ek100 import *
from data_handling.video_reader import VideoReader
from utils.args import arg_dict_to_list
from utils.globals import *
from utils.io import read_pkl

os.chdir(VISOR_PATH)

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.data import MetadataCatalog
from detectron2.structures import Instances, Boxes

from detectron2.projects import point_rend
from hos.data.datasets.epick import register_epick_instances
from hos.data.hos_datasetmapper import HOSMapper
from hos.visualization.v import Visualizer as HOS_Visualizer


def main(arg_dict=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str)
    parser.add_argument("--egohos_max_hand_intersection_ioa", type=float, default=0.25)
    args, _ = parser.parse_known_args(arg_dict_to_list(arg_dict))

    for root, dirs, files in os.walk(args.input_dir):
        for fn in files:
            fn_lower = fn.lower()
            if not fn_lower.endswith(".pkl") and not fn_lower.endswith(".zip"):
                continue

            path = join(root, fn)
            data = read_pkl(path)

            if len(data["instances"]) == 0:
                continue

            # parse video and frame id from filename
            spl = fn.replace(".", "_").split("_")
            video_id = "_".join(spl[:2])
            frame_id = "_".join(spl[:3])
            frame_idx = int(spl[2])

            egohos_path = CHANNEL_FRAME_PATH_FUNCTS["hos_hands"](
                video_id, frame_idx, frame_id, "egohos"
            )
            if isfile(egohos_path):
                # numpy mask
                egohos_data = read_pkl(egohos_path) > 0
                im_shape = data["instances"].pred_masks[0].shape
                if (
                    egohos_data.shape[0] != im_shape[0]
                    or egohos_data.shape[1] != im_shape[1]
                ):
                    egohos_data_img = Image.fromarray(egohos_data)
                    egohos_data_img = egohos_data_img.resize(
                        (egohos_data.shape[1], egohos_data.shape[0]), Image.NEAREST
                    )
                    egohos_data = np.array(egohos_data_img) > 0

                egohos_data = torch.from_numpy(egohos_data)

                pred_class_list = []
                pred_handside_list = []
                pred_mask_list = []
                pred_box_list = []
                pred_score_list = []

                for cls, handside, mask, box, score in zip(
                    data["instances"].pred_classes,
                    data["instances"].pred_handsides,
                    data["instances"].pred_masks,
                    data["instances"].pred_boxes,
                    data["instances"].scores,
                ):
                    add = False
                    if cls != 1:  # filter for objects only
                        add = True

                    if not add:
                        anded = torch.logical_and(egohos_data, mask)
                        ioa = anded.sum() / max(1, mask.sum())
                        if ioa <= args.egohos_max_hand_intersection_ioa:
                            add = True
                            print(
                                f"IoA {ioa} <= {args.egohos_max_hand_intersection_ioa}; mask kept"
                            )
                        else:
                            print(
                                f"IoA {ioa} > {args.egohos_max_hand_intersection_ioa}; mask ignored"
                            )

                    if add:
                        pred_class_list.append(cls)
                        pred_handside_list.append(handside)
                        pred_mask_list.append(mask)
                        pred_box_list.append(box)
                        pred_score_list.append(score)

                new_instances = Instances(
                    image_size=(im_shape[0], im_shape[1])
                )  # (height, width)
                if len(pred_class_list) > 0:
                    new_instances.pred_classes = torch.stack(pred_class_list)
                    new_instances.pred_handsides = torch.stack(pred_handside_list)
                    new_instances.pred_masks = torch.stack(pred_mask_list).cuda()
                    new_instances.pred_boxes = Boxes(torch.stack(pred_box_list))
                    new_instances.scores = torch.stack(pred_score_list)

                data["instances"] = new_instances
                if not fn_lower.endswith(".zip"):
                    path += ".zip"
                with zipfile.ZipFile(
                    path, "w", zipfile.ZIP_DEFLATED, False
                ) as zip_file:
                    #  {k: v for k, v in outputs["instances"]._fields.items() if "mask" not in k}
                    zip_file.writestr(os.path.basename(path), pickle.dumps(data))
                print(f"Processed {path}")


if __name__ == "__main__":
    main()
