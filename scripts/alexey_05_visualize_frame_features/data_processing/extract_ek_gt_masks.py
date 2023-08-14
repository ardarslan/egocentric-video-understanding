import argparse
import numpy as np
from PIL import Image, ImageDraw
from os.path import basename, dirname, isfile, isdir, join
import pickle
import string
import sys
from tqdm import tqdm
import zipfile

sys.path.append(dirname(dirname(__file__)))

from data_handling.specific.ek100 import *
from utils.args import arg_dict_to_list
from utils.imaging import calculate_iou
from utils.io import read_json


def draw_mask(target_draw_object, annot_segments, color, orig_width, orig_height, target_width, target_height):
    x_ratio = target_width / orig_width
    y_ratio = target_height / orig_height
    for annot_segments_sub in annot_segments:
        if len(annot_segments_sub) >= 3:  # need at least 3 coordinates
            polygon_segments = list(map(lambda c: (c[0] * x_ratio, c[1] * y_ratio), annot_segments_sub))
            target_draw_object.polygon(polygon_segments, fill=color)


def main(arg_dict=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--annotation_root_path", action="append", type=str, default=join(DATA_ROOT_PATH, "EK_VISOR"))
    parser.add_argument("--full_video", action="store_true")
    parser.add_argument("--generator_videos", action="append", type=str, default=None)
    parser.add_argument("--scaled_width", action="append", type=int, default=854)
    parser.add_argument("--scaled_height", action="append", type=int, default=480)
    parser.add_argument("--iou_matching_threshold_initial", type=float, default=0.95)
    parser.add_argument("--iou_matching_threshold_sequential", type=float, default=0.6)
    
    args, _ = parser.parse_known_args(arg_dict_to_list(arg_dict))
    
    if args.generator_videos is None:
        args.generator_videos = get_video_list()
    else:
        args.generator_videos = [s.strip() for v in args.generator_videos for s in v.split(",")]

    sparse_annotation_root_path = join(args.annotation_root_path, "GroundTruth-SparseAnnotations", "annotations")
    dense_annotation_root_path = join(args.annotation_root_path, "Interpolations-DenseAnnotations")
    for video_id in args.generator_videos:
        sparse_annotation_path_train_noext = join(sparse_annotation_root_path, "train", f"{video_id}")
        sparse_annotation_path_val_noext = join(sparse_annotation_root_path, "val", f"{video_id}")
        sparse_annotation_path = None

        # dense annotations and sparse annotations use a disjoint ID space, so we need to detect
        # active objects in the dense annotations by the IOU

        original_sparse_frame_idxs_to_annot_idxs = {}
        original_contact_ids_to_annot_idxs = {}
        original_sparse_frame_idxs_to_contacted_objects = {}
        original_sparse_contact_ids = set()

        # needed for correction factor:
        reader_original = VideoReader(get_video_path(video_id, use_original=True), get_extracted_frame_dir_path(video_id, use_original=True))
        original_fps = reader_original.fps

        reader = VideoReader(get_video_path(video_id), get_extracted_frame_dir_path(video_id),
                             assumed_fps=60 if len(video_id.split('_')[-1]) == 2 else 50)
        
        for sparse_annotation_path_candidate in [sparse_annotation_path_train_noext,
                                                 sparse_annotation_path_val_noext]:
            for ext in [".json", ".zip"]:
                if isfile(sparse_annotation_path_candidate + ext):
                    sparse_annotation_path = sparse_annotation_path_candidate + ext
                    sparse_annotation_data = read_json(sparse_annotation_path)
                    
                    for annot_top_data_idx, annot_top_data in enumerate(sparse_annotation_data["video_annotations"]):
                        name = annot_top_data["image"].get("name", "")
                        if name.startswith(video_id):
                            name = name[len(video_id):]
                        original_frame_idx = next((int(spl) for spl in name.replace(".", "_").split("_") if spl.isnumeric()), None)
                        if original_frame_idx is None:
                            continue
            
                        for annot_subentry_idx, annot_subentry in enumerate(annot_top_data["annotations"]):
                            annot_id = annot_subentry.get("id", "")
                            full_idx_id = (annot_top_data_idx, annot_subentry_idx)
                            if original_frame_idx not in original_sparse_frame_idxs_to_annot_idxs:
                                original_sparse_frame_idxs_to_annot_idxs[original_frame_idx] = []
                            original_sparse_frame_idxs_to_annot_idxs[original_frame_idx].append(full_idx_id)

                            contact_key = annot_id  # unique across frames
                            if contact_key not in original_contact_ids_to_annot_idxs:
                                original_contact_ids_to_annot_idxs[contact_key] = []
                            original_contact_ids_to_annot_idxs[contact_key].append(full_idx_id)

                            if ("in_contact_object" in annot_subentry
                                and (in_contact_object := annot_subentry["in_contact_object"]) is not None
                                and in_contact_object not in ["", "hand-not-in-contact", "none-of-the-above", "inconclusive"]
                                and all((c in string.hexdigits for c in in_contact_object))):
                                if original_frame_idx not in original_sparse_frame_idxs_to_contacted_objects:
                                    original_sparse_frame_idxs_to_contacted_objects[original_frame_idx] = []
                                original_sparse_frame_idxs_to_contacted_objects[original_frame_idx].append(in_contact_object)
                                original_sparse_contact_ids.add(in_contact_object)

                                # do not break here; need to register other objects

                    for annot_top_data_idx, annot_top_data in enumerate(sparse_annotation_data["video_annotations"]):
                        name = annot_top_data["image"].get("name", "")
                        if name.startswith(video_id):
                            name = name[len(video_id):]
                        original_frame_idx = next((int(spl) for spl in name.replace(".", "_").split("_") if spl.isnumeric()), None)
                        if original_frame_idx is None:
                            continue
            
                        contact_mask_frame = np.zeros((reader.video_height, reader.video_width), dtype=bool)

                        for annot_subentry_idx, annot_subentry in enumerate(annot_top_data["annotations"]):
                            annot_id = annot_subentry.get("id", "")
                            annot_segments = annot_subentry.get("segments", "")
                            if len(annot_segments) > 0 and annot_id in original_sparse_contact_ids:
                                # test for intersection with contacted objects
                                dense_annot_img = Image.new("L", (reader.video_width, reader.video_height), (0,))
                                dense_annot_img_draw = ImageDraw.Draw(dense_annot_img)
                                # here, we *do* need to rescale
                                draw_mask(dense_annot_img_draw, annot_segments, (255,),
                                          reader.video_width, reader.video_height, reader.video_width, reader.video_height)
                                contact_mask_frame |= (np.array(dense_annot_img) > 0)
                                
                        if contact_mask_frame.max():
                            hos_output_path_nozip = join(HOS_DATA_DIR, "ek-gt-sparse", "object", video_id, "pkls", f"{fmt_frame(video_id, original_frame_idx)}.pkl")
                            os.makedirs(dirname(hos_output_path_nozip), exist_ok=True)
                            with zipfile.ZipFile(f"{hos_output_path_nozip}.zip", "w", zipfile.ZIP_DEFLATED, False) as zip_file:
                                zip_file.writestr(basename(hos_output_path_nozip), pickle.dumps(contact_mask_frame.astype(np.uint8)))
                                
                                
            if sparse_annotation_path is not None:
                break

        if sparse_annotation_path is None:
            print(f"No sparse annotations found for {video_id}; skipping")
            continue

        dense_annotation_path_train_noext = join(dense_annotation_root_path, "train", f"{video_id}_interpolations")
        dense_annotation_path_val_noext = join(dense_annotation_root_path, "val", f"{video_id}_interpolations")
        dense_annotation_path = None

        for dense_annotation_path_candidate in [dense_annotation_path_train_noext,
                                                dense_annotation_path_val_noext]:
            for ext in [".json", ".zip"]:
                if isfile(dense_annotation_path_candidate + ext):
                    dense_annotation_path = dense_annotation_path_candidate + ext
                    break
            if dense_annotation_path is not None:
                break
        
        if dense_annotation_path is None:
            print(f"No dense annotations found for {video_id}; skipping")
            continue

        dense_annotation_data = read_json(dense_annotation_path)
        if "video_annotations" not in dense_annotation_data:
            print(f"Invalid annotations for {video_id}; skipping")
            continue
        
        original_dense_frame_idxs_to_annot_idxs = {}
        for annot_top_data_idx, annot_top_data in enumerate(dense_annotation_data["video_annotations"]):
            name = annot_top_data["image"]["name"]
            name_lower = name.lower()
            if name_lower.endswith(".jpg") or name_lower.endswith(".png"):
                name = name.rsplit(".", 1)[0]
            if name.startswith(video_id):
                name = name[len(video_id):]
            name_lower = name.lower()
            
            original_frame_idx = next((int(spl) for spl in name.replace(".", "_").split("_") if spl.isnumeric()), None)
            if original_frame_idx is None:
                continue
            
            for annot_subentry_idx, annot_subentry in enumerate(annot_top_data["annotations"]):
                if original_frame_idx not in original_dense_frame_idxs_to_annot_idxs:
                    original_dense_frame_idxs_to_annot_idxs[original_frame_idx] = []
                
                original_dense_frame_idxs_to_annot_idxs[original_frame_idx].append( (annot_top_data_idx, annot_subentry_idx) )

        # dict of interpolation ID -> (dict of name -> (dict of ID -> mask))
        dense_contact_store = {}

        converted_fps = reader.fps
        annot_interpolation_id_first_frames = {}
        annot_interpolation_id_recent_frames = {}
        for frame_idx in range(reader.get_virtual_frame_count()):
            frame_id = fmt_frame(video_id, frame_idx)
            original_frame_idx = frame_idx  #  frame_data["original_frame_idx"]  # !!! changed from original_frame_idx, since that's what EK does

            img_np = reader.get_frame(frame_idx)
            img = Image.fromarray(img_np)
            img = img.convert("RGBA")

            if original_frame_idx in original_dense_frame_idxs_to_annot_idxs:
                # create masks for all objects that are in contact, to later check for IoU
                # id: mask
                frame_contacted_object_masks = {}
                contacted_objects = original_sparse_frame_idxs_to_contacted_objects.get(original_frame_idx, [])
                for contacted_object_id in contacted_objects:
                    contact_annot_idx = original_contact_ids_to_annot_idxs.get(contacted_object_id)
                    if contact_annot_idx is not None:
                        annot_subentry_data = sparse_annotation_data["video_annotations"][contact_annot_idx[0][0]]["annotations"][contact_annot_idx[0][1]]
                        annot_segments = annot_subentry_data.get("segments", "")
                        if len(annot_segments) > 0:
                            contact_img = Image.new("L", (img.width, img.height), (0,))
                            contact_img_draw = ImageDraw.Draw(contact_img)
                            # here, we *do not* need to rescale
                            draw_mask(contact_img_draw, annot_subentry_data["segments"], 255,
                                      img.width, img.height, img.width, img.height)
                            frame_contacted_object_masks[contacted_object_id] = np.array(contact_img)

                img_polygons = Image.new("RGBA", (img.width, img.height), color=(0, 0, 0, 0))
                draw_polygons = ImageDraw.Draw(img_polygons)
                annots_found = 0
                found_initial_contact = False
                contact_mask_frame = np.zeros((img.height, img.width), dtype=bool)

                for idx_pair in original_dense_frame_idxs_to_annot_idxs[original_frame_idx]:
                    annot = dense_annotation_data["video_annotations"][idx_pair[0]]["annotations"][idx_pair[1]]
                    annot_cls = annot.get("class_id", 0)
                    annot_segments = annot.get("segments", "")
                    annot_name = annot.get("name", "")
                    annot_id = annot.get("id", "")

                    annot_interpolation_id = dense_annotation_data["video_annotations"][idx_pair[0]]["image"]["interpolation_start_frame"]

                    if len(annot_segments) > 0:
                        # test for intersection with contacted objects

                        found_contact_id = None

                        if found_contact_id is None:
                            dense_annot_img = Image.new("L", (img.width, img.height), (0,))
                            dense_annot_img_draw = ImageDraw.Draw(dense_annot_img)
                            # here, we *do* need to rescale
                            draw_mask(dense_annot_img_draw, annot_segments, (255,),
                                      args.scaled_width, args.scaled_height, img.width, img.height)
                            dense_annot_mask = np.array(dense_annot_img)
                            
                            for contact_id, contact_mask in frame_contacted_object_masks.items():
                                if calculate_iou(contact_mask, dense_annot_mask) >= args.iou_matching_threshold_initial:
                                    found_contact_id = contact_id
                                    if annot_interpolation_id not in dense_contact_store:
                                        dense_contact_store[annot_interpolation_id] = {}
                                    if annot_name not in dense_contact_store[annot_interpolation_id]:
                                        dense_contact_store[annot_interpolation_id][annot_name] = {}
                                    found_initial_contact = True
                                    annot_interpolation_id_first_frames[annot_interpolation_id] = frame_idx
                                    dense_contact_store[annot_interpolation_id][annot_name][found_contact_id] = dense_annot_mask
                                    contact_mask_frame |= (dense_annot_mask > 0)
                                    break
                                
                            if found_contact_id is None:
                                if ((annot_stored_names_contacts := dense_contact_store.get(annot_interpolation_id)) is not None
                                    and (annot_name_stored_contacts := annot_stored_names_contacts.get(annot_name)) is not None):
                                    for contact_id, contact_mask in annot_name_stored_contacts.items():
                                        if (calculate_iou(contact_mask, dense_annot_mask) >= args.iou_matching_threshold_sequential
                                            or len(annot_name_stored_contacts) == 1):
                                            found_contact_id = contact_id
                                            dense_contact_store[annot_interpolation_id][annot_name][found_contact_id] = dense_annot_mask
                                            contact_mask_frame |= (dense_annot_mask > 0)
                                            break

                        if found_contact_id is None:
                            continue

                        rng = np.random.default_rng(seed=annot_cls)
                        color = (*[int(round(i)) for i in rng.uniform(low=0, high=255, size=3)], 128)
                        draw_mask(draw_polygons, annot_segments, color,
                                  args.scaled_width, args.scaled_height, img.width, img.height)
                        # test for intersection with contacted object
                        annots_found += 1
                
                # write HOS output

                if annot_interpolation_id not in annot_interpolation_id_first_frames:
                    continue

                original_frame_idx = frame_idx
                if not found_initial_contact:
                    # need to recalculate the frame ID
                    correction_factor = converted_fps / original_fps

                    if abs(correction_factor - 1.0) > 1e-5:
                        frame_idx = round(annot_interpolation_id_first_frames[annot_interpolation_id]
                                          + correction_factor * (frame_idx - annot_interpolation_id_first_frames[annot_interpolation_id]))
                        frame_id = fmt_frame(video_id, frame_idx)
                        img_np = reader.get_frame(frame_idx)
                        img = Image.fromarray(img_np).convert("RGBA")
                        
                # keep this here
                annot_recent_frame = annot_interpolation_id_recent_frames.get(annot_interpolation_id)
                annot_interpolation_id_recent_frames[annot_interpolation_id] = frame_idx

                hos_output_path_nozip = join(HOS_DATA_DIR, "ek-gt-dense", "object", video_id, "pkls", f"{frame_id}.pkl")
                os.makedirs(dirname(hos_output_path_nozip), exist_ok=True)
                with zipfile.ZipFile(f"{hos_output_path_nozip}.zip", "w", zipfile.ZIP_DEFLATED, False) as zip_file:
                    zip_file.writestr(basename(hos_output_path_nozip), pickle.dumps(contact_mask_frame.astype(np.uint8)))
                
                if annot_recent_frame is not None and annot_recent_frame + 1 != frame_idx:
                    for intermediate_idx in range(annot_recent_frame + 1, frame_idx):
                        hos_output_path_nozip = join(HOS_DATA_DIR, "ek-gt-dense", "object", video_id, "pkls", f"{fmt_frame(video_id, intermediate_idx)}.pkl")
                        with zipfile.ZipFile(f"{hos_output_path_nozip}.zip", "w", zipfile.ZIP_DEFLATED, False) as zip_file:
                            zip_file.writestr(basename(hos_output_path_nozip), pickle.dumps(contact_mask_frame.astype(np.uint8)))

                if annots_found > 0:
                    img_output_path = join(args.output_dir, video_id, f"{frame_id}.jpg")
                    img_composite = Image.alpha_composite(img, img_polygons).convert("RGB")
                    os.makedirs(dirname(img_output_path), exist_ok=True)
                    img_composite.save(img_output_path)
                    print(f"{img_output_path=} {annots_found=} {found_initial_contact=} {frame_idx=} {original_frame_idx=} {annot_interpolation_id_first_frames.get(annot_interpolation_id)=} {annot_interpolation_id=}")
                    pass
                        


if __name__ == "__main__":
    main()
