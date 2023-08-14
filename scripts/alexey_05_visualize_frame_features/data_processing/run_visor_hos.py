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


POINTREND_CFG = "./configs/hos/hos_pointrend_rcnn_R_50_FPN_1x_trainset.yaml"
EPICK_MODEL = f"./checkpoints/model_final_hos.pth"


# register epick visor dataset
version = "datasets/epick_visor_coco_hos"
register_epick_instances("epick_visor_2022_val_hos", {}, f"{version}/annotations/val.json", f"{version}/val")
MetadataCatalog.get("epick_visor_2022_val_hos").thing_classes = ["hand", "object"]
epick_visor_metadata = MetadataCatalog.get("epick_visor_2022_val_hos")

    
def hos_postprocessing(predictions):
    """
    Use predicted offsets to associate hand and its in-contact obj.
    """
    preds = predictions["instances"].to("cpu")
    if len(preds) == 0: return predictions
    # separate hand, obj preds
    hand_preds = preds[preds.pred_classes == 0]
    obj_preds = preds[preds.pred_classes == 1]
    
    if len(obj_preds) == 0: return {"instances":hand_preds}
    
    # find incontact obj
    incontact_obj = []
    updated_hand_preds = []
    for i in range(len(hand_preds)):
        box = hand_preds[i].pred_boxes.tensor.cpu().detach().numpy()[0]
        side = hand_preds[i].pred_handsides.cpu().detach().numpy()[0]
        contact = hand_preds[i].pred_contacts.cpu().detach().numpy()[0]
        offset = hand_preds[i].pred_offsets.cpu().detach().numpy()[0]
        # if incontact
        if int(np.argmax(contact)):
            obj = get_incontact_obj(hand_preds[i], offset, obj_preds)
            if isinstance(obj, Instances):
                incontact_obj.append(obj)
                new = Instances(hand_preds[i].image_size)
                for field in hand_preds[i]._fields:
                    if field == "pred_offsets":
                        new.set(field, torch.Tensor([get_offset(box, obj.pred_boxes.tensor.cpu().detach().numpy()[0])])  )
                    else:
                        new.set(field, hand_preds[i].get(field))
                updated_hand_preds.append(new)
               
        else:
            updated_hand_preds.append(hand_preds[i])
            
    if len(incontact_obj) > 0:
        incontact_obj.extend(updated_hand_preds)
        ho = Instances.cat(incontact_obj)
    else:
        if len(updated_hand_preds) > 0:
            ho = Instances.cat(updated_hand_preds)
        else:
            ho = Instances( preds[0].image_size)
        
    return {"instances": ho}


def get_offset(h_bbox_xyxy, o_bbox_xyxy):
    h_center = [int((h_bbox_xyxy[0] + h_bbox_xyxy[2]) / 2), int((h_bbox_xyxy[1] + h_bbox_xyxy[3]) / 2)]
    o_center = [int((o_bbox_xyxy[0] + o_bbox_xyxy[2]) / 2), int((o_bbox_xyxy[1] + o_bbox_xyxy[3]) / 2)]
    # offset: [vx, vy, magnitute]
    scalar = 1000 
    vec = np.array([o_center[0]-h_center[0], o_center[1]-h_center[1]]) / scalar
    norm = np.linalg.norm(vec)
    unit_vec = vec / norm
    offset = [unit_vec[0], unit_vec[1], norm]
    return offset    
    

def get_incontact_obj(h_box, offset, obj_preds):
    h_center = get_center(h_box)
    scalar = 1000
    offset_vec = [ offset[0]*offset[2]*scalar, offset[1]*offset[2]*scalar ] 
    pred_o_center = [h_center[0]+offset_vec[0], h_center[1]+offset_vec[1]]
    
    # choose from obj_preds
    dist_ls = []
    for i in range(len(obj_preds)):
        o_center = get_center(obj_preds[i])
        dist = np.linalg.norm(np.array(o_center) - np.array(pred_o_center))
        dist_ls.append(dist)
    
    if len(dist_ls) == 0: 
        return []
    else:
        o_ind = np.argmin(np.array(dist_ls))
        return obj_preds[int(o_ind)]


def get_center(box):
    box = box.pred_boxes.tensor.cpu().detach().numpy()[0]
    x0, y0, x1, y1 = box
    center = [int((x0+x1)/2), int((y0+y1)/2)]
    return center


def main(arg_dict=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--visualization_interval", type=int, default=100)
    parser.add_argument("--generator_videos", action="append", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--score_threshold", type=float, default=DEFAULT_HOS_THRESHOLD)
    parser.add_argument("--egohos_max_hand_intersection_ioa", type=float, default=1.0)
    args, _ = parser.parse_known_args(arg_dict_to_list(arg_dict))
    
    if args.output_dir is None:
        args.output_dir = join(HOS_DATA_DIR, "threshold=" + str(args.score_threshold))
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    cfg = get_cfg()
    point_rend.add_pointrend_config(cfg)
    cfg.merge_from_file(POINTREND_CFG)

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
    cfg.MODEL.POINT_HEAD.NUM_CLASSES = 2

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.score_threshold  # set threshold for this model
    cfg.MODEL.WEIGHTS = EPICK_MODEL
    predictor = DefaultPredictor(cfg)

    if args.generator_videos is None:
        args.generator_videos = get_video_list()
    else:
        args.generator_videos = [s.strip() for v in args.generator_videos for s in v.split(",")]

    global_frame_idx = 0
    for video_idx, video_id in tqdm(enumerate(args.generator_videos)):
        reader = VideoReader(get_video_path(video_id), get_extracted_frame_dir_path(video_id),
                             assumed_fps=EK_ASSUMED_FPS)
        
        #try:
        if True:
            range_obj = range(reader.get_virtual_frame_count())
            
            os.makedirs(os.path.join(out_dir, video_id, "pkls"), exist_ok=True)
            os.makedirs(os.path.join(out_dir, video_id, "jpgs"), exist_ok=True)
            
            for frame_idx in range_obj:
                frame_id = fmt_frame(video_id, frame_idx)
                
                try:
                    im = reader.get_frame(frame_idx)[:, :, ::-1]
                except ToggleableException as ex:
                    print(f"Error reading frame {frame_idx} from video {video_id}:", ex)
                    continue
                out_path = os.path.join(out_dir, video_id, "jpgs", frame_id + ".jpg")
                pkl_out_path = os.path.join(out_dir, video_id, "pkls", frame_id + ".pkl")
                if os.path.isfile(pkl_out_path + ".zip"):
                    continue

                outputs = predictor(im)
                v = HOS_Visualizer(im[:, :, ::-1], epick_visor_metadata, scale=1.2, instance_mode=ColorMode.IMAGE_BW)

                if args.egohos_max_hand_intersection_ioa != 1.0:
                    # check if we have EgoHOS data for this frame
                    egohos_path = CHANNEL_FRAME_PATH_FUNCTS["hos_hands"](video_id, frame_idx, frame_id, "egohos")
                    if isfile(egohos_path):
                        # numpy mask
                        egohos_data = read_pkl(egohos_path) > 0
                        if egohos_data.shape[0] != im.shape[0] or egohos_data.shape[1] != im.shape[1]:
                            egohos_data_img = Image.fromarray(egohos_data)
                            egohos_data_img = egohos_data_img.resize((egohos_data.shape[1], egohos_data.shape[0]), Image.NEAREST)
                            egohos_data = np.array(egohos_data_img) > 0
                        
                        egohos_data = torch.from_numpy(egohos_data)

                        pred_class_list = []
                        pred_handside_list = []
                        pred_mask_list = []
                        pred_box_list = []
                        pred_score_list = []

                        for cls, handside, mask, box, score in zip(outputs["instances"].pred_classes,
                                                                   outputs["instances"].pred_handsides,
                                                                   outputs["instances"].pred_masks,
                                                                   outputs["instances"].pred_boxes,
                                                                   outputs["instances"].scores):
                            add = False
                            if cls != 1:  # filter for objects only
                                add = True

                            if not add:
                                anded = torch.logical_and(egohos_data.cpu(), mask.cpu())
                                ioa = anded.sum().item() / max(1, mask.sum().item())
                                if ioa <= args.egohos_max_hand_intersection_ioa:
                                    add = True
                                else:
                                    print(f"IoA {ioa} > {args.egohos_max_hand_intersection_ioa}; mask ignored")

                            if add:
                                pred_class_list.append(cls)
                                pred_handside_list.append(handside)
                                pred_mask_list.append(mask)
                                pred_box_list.append(box)
                                pred_score_list.append(score)
                        
                        new_instances = Instances(image_size=(im.shape[0], im.shape[1]))  # (height, width)
                        if len(pred_class_list) > 0:
                            new_instances.pred_classes = torch.stack(pred_class_list)
                            new_instances.pred_handsides = torch.stack(pred_handside_list)
                            new_instances.pred_masks = torch.stack(pred_mask_list).cuda()
                            new_instances.pred_boxes = Boxes(torch.stack(pred_box_list))
                            new_instances.pred_scores = torch.stack(pred_score_list)
                            new_instances.scores = torch.stack(pred_score_list)

                        outputs["instances"] = new_instances


                image_size_data = {"image_width": im.shape[1], "image_height": im.shape[0]}
                with zipfile.ZipFile(pkl_out_path + ".zip", "w",
                                     zipfile.ZIP_DEFLATED, False) as zip_file:
                    #  {k: v for k, v in outputs["instances"]._fields.items() if "mask" not in k}
                    zip_file.writestr(os.path.basename(pkl_out_path), pickle.dumps({**outputs, **image_size_data})) 

                # outputs = hos_postprocessing(outputs)

                if args.visualization_interval > 0 and global_frame_idx % args.visualization_interval == 0:
                    point_rend_result = v.draw_instance_predictions(outputs["instances"].to("cpu")).get_image()
                    cv2.imwrite(out_path, point_rend_result[:, :, ::-1])

            global_frame_idx += 1
        #except Exception as ex:
        #    print(f"Error processing {video_id}:", ex)
        #    continue


if __name__ == "__main__":
    main()
