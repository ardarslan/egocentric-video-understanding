import os
import torch
import argparse
import mmcv
import numpy as np
from mmseg.apis import init_segmentor
from mmcv.parallel import collate, scatter
from mmseg.datasets.pipelines import Compose

from PIL import Image
from skimage.io import imsave
from pathlib import Path


class LoadImage:
    """A simple pipeline to load image."""

    def __call__(self, results):
        """Call function to load images into results.

        Args:
            results (dict): A result dict contains the file name
                of the image to be read.

        Returns:
            dict: ``results`` will be returned containing loaded image.
        """

        if isinstance(results["img"], str):
            results["filename"] = results["img"]
            results["ori_filename"] = results["img"]
        else:
            results["filename"] = None
            results["ori_filename"] = None
        img = mmcv.imread(results["img"])
        results["img"] = img
        results["img_shape"] = img.shape
        results["ori_shape"] = img.shape
        return results


def inference_segmentor(model, img, previous_results):
    """Inference image(s) with the segmentor.

    Args:
        model (nn.Module): The loaded segmentor.
        imgs (str/ndarray or list[str/ndarray]): Either image files or loaded
            images.

    Returns:
        (list[Tensor]): The segmentation result.
    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    test_pipeline = [LoadImage()] + cfg.data.test.pipeline[1:]
    test_pipeline = Compose(test_pipeline)
    # prepare data
    data = dict(img=img)
    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        data["img_metas"] = [i.data[0] for i in data["img_metas"]]

    if "additional_channel" in cfg.keys():
        data["img_metas"][0][0]["additional_channel"] = cfg["additional_channel"]
    if "twohands_dir" in cfg.keys():
        data["img_metas"][0][0]["twohands_dir"] = cfg["twohands_dir"]
    if "cb_dir" in cfg.keys():
        data["img_metas"][0][0]["cb_dir"] = cfg["cb_dir"]

    # forward the model
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, previous_results=previous_results, **data)
    return result


parser = argparse.ArgumentParser()
parser.add_argument(
    "--ego_hos_seg_twohands_config_file_path",
    default="/home/aarslan/mq/scripts/extract_frame_features/ego_hos/configs/seg_twohands_ccda.py",
    type=str,
)
parser.add_argument(
    "--ego_hos_seg_twohands_model_file_path",
    default="/srv/beegfs02/scratch/aarslan_data/data/mq_libs/ego_hos/seg_twohands_ccda/best_mIoU_iter_56000.pth",
    type=str,
)
parser.add_argument(
    "--ego_hos_twohands_to_cb_config_file_path",
    default="/home/aarslan/mq/scripts/extract_frame_features/ego_hos/configs/twohands_to_cb_ccda.py",
    type=str,
)
parser.add_argument(
    "--ego_hos_twohands_to_cb_model_file_path",
    default="/srv/beegfs02/scratch/aarslan_data/data/mq_libs/ego_hos/twohands_to_cb_ccda/best_mIoU_iter_76000.pth",
    type=str,
)
parser.add_argument(
    "--ego_hos_twohands_cb_to_obj2_config_file_path",
    default="/home/aarslan/mq/scripts/extract_frame_features/ego_hos/configs/twohands_cb_to_obj2_ccda.py",
    type=str,
)
parser.add_argument(
    "--ego_hos_twohands_cb_to_obj2_model_file_path",
    default="/srv/beegfs02/scratch/aarslan_data/data/mq_libs/ego_hos/twohands_cb_to_obj2_ccda/best_mIoU_iter_32000.pth",
    type=str,
)
args = parser.parse_args()

seg_twohands_model = init_segmentor(
    args.ego_hos_seg_twohands_config_file_path,
    args.ego_hos_seg_twohands_model_file_path,
    device="cuda",
)

twohands_to_cb_model = init_segmentor(
    args.ego_hos_twohands_to_cb_config_file_path,
    args.ego_hos_twohands_to_cb_model_file_path,
    device="cuda",
)

twohands_cb_to_obj2_model = init_segmentor(
    args.ego_hos_twohands_cb_to_obj2_config_file_path,
    args.ego_hos_twohands_cb_to_obj2_model_file_path,
    device="cuda",
)

alpha = 0.5

img_path = "/home/aarslan/EgoHOS/testimages/inputs/first_person_cooking.png"

seg_twohands_result = inference_segmentor(seg_twohands_model, img_path, previous_results={})[0].astype(np.uint8)

twohands_to_cb_result = inference_segmentor(
    twohands_to_cb_model,
    img_path,
    previous_results={"seg_twohands_result": Image.fromarray(seg_twohands_result)},
)[
    0
].astype(np.uint8)

twohands_cb_to_obj2_result = inference_segmentor(
    twohands_cb_to_obj2_model,
    img_path,
    previous_results={
        "seg_twohands_result": Image.fromarray(seg_twohands_result),
        "twohands_to_cb_result": Image.fromarray(twohands_to_cb_result),
    },
)[0]

seg_twohands_result[twohands_cb_to_obj2_result == 1] = 3
seg_twohands_result[twohands_cb_to_obj2_result == 2] = 4
seg_twohands_result[twohands_cb_to_obj2_result == 3] = 5
seg_twohands_result[twohands_cb_to_obj2_result == 4] = 6
seg_twohands_result[twohands_cb_to_obj2_result == 5] = 7
seg_twohands_result[twohands_cb_to_obj2_result == 6] = 8

img = np.array(Image.open(img_path))[:, :, :3]

alpha = 0.5
seg_color = np.zeros(img.shape)
# seg_color[seg_twohands_result == 0] = (0, 0, 0)  # background
# seg_color[seg_twohands_result == 1] = (255, 0, 0)  # left_hand
# seg_color[seg_twohands_result == 2] = (0, 0, 255)  # right_hand
seg_color[seg_twohands_result == 3] = (255, 0, 255)  # left_object1
seg_color[seg_twohands_result == 4] = (0, 255, 255)  # right_object1
seg_color[seg_twohands_result == 5] = (0, 255, 0)  # two_object1
seg_color[seg_twohands_result == 6] = (255, 204, 255)  # left_object2
seg_color[seg_twohands_result == 7] = (204, 255, 255)  # right_object2
seg_color[seg_twohands_result == 8] = (204, 255, 204)  # two_object2
twohands_obj2_vis = img * (1 - alpha) + seg_color * alpha

os.makedirs(Path(img_path.replace("inputs", "visualization")).parent, exist_ok=True)
imsave(
    img_path.replace("inputs", "visualization"),
    twohands_obj2_vis.astype(np.uint8),
)
