import cv2
from django.http import HttpResponse, JsonResponse
from django.template import loader
import imageio
import io
import json
import math
import numpy as np
from os.path import dirname, isdir, isfile, join
import pickle
from PIL import Image, ImageDraw, ImageFont
import sys
import zipfile

sys.path.append(dirname(dirname(dirname(__file__))))

from data_handling.specific.ek100 import *
from data_handling.reflection import *
import utils.globals
from utils.globals import *
from utils.imaging import *
from utils.io import read_pkl
from utils.unpicklers import CPUUnpickler


def index(request):
    template = loader.get_template("webserver/index.html")
    context = {}
    # add videos
    context["videos"] = [{"name": "P01_12"}, {"name": "P01_13"}, {"name": "P08_17"}, {"name": "P09_07"}]
    #context["videos"] = [{"name": video_name} for video_name in get_video_list()]
    
    for global_name in dir(utils.globals):
        if global_name.startswith("__"):
            continue
        context[global_name] = getattr(utils.globals, global_name)

    return HttpResponse(template.render(context, request))


# def create_comment(request, video_id, frame_idx):
#     comment_id = int(time.time() * 1000)
#     frame_id = fmt_frame(video_id, frame_idx)
#     comment_path = join(CHANNEL_FRAME_PATH_FUNCTS["comment"](video_id, frame_idx, frame_id), f"{comment_id}.pkl")
#     output_dict = {"frame_id": frame_id,
#                    "video_id": video_id,
#                    "frame_idx": frame_idx,
#                    "timestamp": int(time.time()),
#                    "comment_id": str(comment_id),
#                    "text": request.GET.get("text", "")}
#     os.makedirs(dirname(comment_path), exist_ok=True)
#     with open(comment_path, "wb") as f:
#         pickle.dump(output_dict, f)
#     return JsonResponse(output_dict)


# def update_comment(request, video_id, frame_idx, comment_id):
#     frame_id = fmt_frame(video_id, frame_idx)
#     comment_path = join(CHANNEL_FRAME_PATH_FUNCTS["comment"](video_id, frame_idx, frame_id), f"{comment_id}.pkl")
#     output_dict = {"success": True,
#                    "updated": isfile(comment_path),
#                    "frame_id": frame_id,
#                    "video_id": video_id,
#                    "frame_idx": frame_idx,
#                    "comment_id": str(comment_id),
#                    "timestamp": int(time.time()),
#                    "text": request.GET.get("text", "")}
#     os.makedirs(dirname(comment_path), exist_ok=True)
#     with open(comment_path, "wb") as f:
#         pickle.dump(output_dict, f)
#     return JsonResponse(output_dict)


# def remove_comment(request, video_id, frame_idx, comment_id):
#     frame_id = fmt_frame(video_id, frame_idx)
#     comment_path = join(CHANNEL_FRAME_PATH_FUNCTS["comment"](video_id, frame_idx, frame_id), f"{comment_id}.pkl")
#     if isfile(comment_path):
#         os.unlink(comment_path)
#         return JsonResponse({"success": True})
#     else:
#         return JsonResponse({"success": False})


# def get_video_virtual_availability_aux(request, video_id, csv_data):
#     def get_channel_cache_path(channel):
#         return join(EK_AVAILABILITY_CACHE_DIR, video_id, f"{channel}.pkl")

#     start_tm = time.time()

#     av = {c: [] for c in [*CHANNEL_FRAME_PATH_FUNCTS.keys(), "tracking_bbox"]
#           if ("hos" not in c
#               and "tracking_mask" not in c
#               and "inpainted" not in c
#               and "segmentation_mask" not in c)}  # availability data

#     for _, row in csv_data.iterrows():
#         range_obj = range(row["start_frame"], row["stop_frame"])

#         av["gt_activity"].append([row["start_frame"], row["stop_frame"]-1, row["verb"], row["noun"]])
    
#     print("csv_data.iterrows(): %.4f" % (time.time() - start_tm))

#     with open(FOCUS_DATA_PATH, "rb") as f:
#         focus_dict = pickle.load(f)

#     print("focus_dict: %.4f" % (time.time() - start_tm))

#     tracking_bbox_channels = [strp for s in request.GET["tracking_channels"].split(",") if len(strp := s.strip()) > 0 ] if "tracking_channels" in request.GET else []
#     available_hos_versions = get_available_hos_versions()
#     available_image_versions = get_available_image_versions()
#     available_tracking_mask_min_lengths = get_available_tracking_mask_min_lengths()
#     available_segmentation_mask_versions = get_available_segmentation_mask_versions()
#     available_hand_mesh_versions = get_available_hand_mesh_versions()
#     available_object_bbox_versions = get_available_object_bbox_versions()

#     all_channels = get_all_channels()
#     all_channels_filtered = set([s for s in all_channels if s not in EXCLUDED_CHANNELS])

#     handled_channels = set()
#     channel_fc_lm = {}
#     print("pre channel loop: %.4f" % (time.time() - start_tm))
#     for channel in all_channels:
#         if channel in EXCLUDED_CHANNELS:
#             continue

#         channel_cache_path = get_channel_cache_path(channel)
#         file_count, last_mod_timestamp = get_file_count_and_last_mod_timestamp(channel, video_id)
#         channel_fc_lm[channel] = (file_count, last_mod_timestamp)
#         if isfile(channel_cache_path):
#             channel_data = read_pkl(channel_cache_path)
#             if ("file_count" in channel_data
#                 and "last_mod_timestamp" in channel_data
#                 and channel_data["file_count"] == file_count
#                 and channel_data["last_mod_timestamp"] == last_mod_timestamp):
#                 av[channel] = channel_data["data"]
#                 handled_channels.add(channel)

#     print(f"{all_channels_filtered.difference(handled_channels)=}")
#     print("post channel loop: %.4f" % (time.time() - start_tm))

#     if len(all_channels_filtered.difference(handled_channels)) > 0:
#         reader = VideoReader(get_video_path(video_id), get_extracted_frame_dir_path(video_id), assumed_fps=EK_ASSUMED_FPS)

#         print("post reader init: %.4f" % (time.time() - start_tm))

#         if "image" not in handled_channels:
#             av["image"].append([0, reader.get_virtual_frame_count()])
        
#         unavailable_channels = set()

#         for frame_idx in range(reader.get_virtual_frame_count()):
#             frame_id = fmt_frame(video_id, frame_idx)
#             for version in available_image_versions:
#                 if version == "image":
#                     continue

#                 path_funct_channel = "inpainted" if version.startswith("inpaint") else "depth_postprocessing"
#                 if path_funct_channel == "inpainted":
#                     channel = f"inpainted_{version}"
#                 else:
#                     channel = version
#                 if channel in unavailable_channels:
#                     continue

#                 if channel not in av:
#                     av[channel] = []
                
#                 path = CHANNEL_FRAME_PATH_FUNCTS[path_funct_channel](video_id, frame_idx, frame_id, version)
                
#                 if isfile(path):
#                     if len(av[channel]) == 0 or av[channel][-1][1] != frame_idx:
#                         av[channel].append([frame_idx, frame_idx+1])
#                     else:
#                         av[channel][-1][1] = frame_idx+1
#                 elif not isdir(dirname(path)):
#                     unavailable_channels.add(channel)
#                     continue

#             for version in available_hand_mesh_versions:
#                 channel = f"hand_mesh_{version}"
#                 if channel not in av:
#                     av[channel] = []
#                 if channel not in handled_channels and channel not in unavailable_channels:
#                     hand_mesh_path = CHANNEL_FRAME_PATH_FUNCTS["hand_mesh"](video_id, frame_idx, frame_id, version)
#                     hand_mesh_vis_path = CHANNEL_FRAME_PATH_FUNCTS["hand_mesh_vis"](video_id, frame_idx, frame_id, version)
#                     if isfile(hand_mesh_path) or isfile(hand_mesh_vis_path):
#                         if len(av[channel]) == 0 or av[channel][-1][1] != frame_idx:
#                             av[channel].append([frame_idx, frame_idx+1])
#                         else:
#                             av[channel][-1][1] = frame_idx+1
#                     elif not isdir(dirname(hand_mesh_path)) and not isdir(dirname(hand_mesh_vis_path)):
#                         unavailable_channels.add(channel)
#                         continue
                    
#             if "hand_bbox" not in handled_channels and "hand_bbox" not in unavailable_channels:
#                 hand_bbox_path = CHANNEL_FRAME_PATH_FUNCTS["hand_bbox"](video_id, frame_idx, frame_id)
#                 if isfile(hand_bbox_path):
#                     if len(av["hand_bbox"]) == 0 or av["hand_bbox"][-1][-1] != frame_idx:
#                         av["hand_bbox"].append([frame_idx, frame_idx+1])
#                     else:
#                         av["hand_bbox"][-1][-1] = frame_idx+1
#                 elif not isdir(dirname(hand_bbox_path)):
#                     unavailable_channels.add("hand_bbox")

#             for version in available_hos_versions:
#                 for super_channel in ["hos", "hos_hands", "hos_object"]:
#                     channel = f"{super_channel}_{version}"
#                     if channel not in handled_channels and channel not in unavailable_channels:
#                         if channel not in av:
#                             av[channel] = []
                        
#                         hos_path = CHANNEL_FRAME_PATH_FUNCTS[super_channel](video_id, frame_idx, frame_id, version)
#                         if isfile(hos_path):
#                             if len(av[channel]) == 0 or av[channel][-1][-1] != frame_idx:
#                                 av[channel].append([frame_idx, frame_idx+1])
#                             else:
#                                 av[channel][-1][-1] = frame_idx+1
#                         elif not isdir(dirname(hos_path)):
#                             unavailable_channels.add(channel)

#             for segmentation_mask_version in available_segmentation_mask_versions:
#                 channel = f"segmentation_mask_{segmentation_mask_version}"
#                 if channel not in handled_channels and channel not in unavailable_channels:
#                     segmentation_mask_path = CHANNEL_FRAME_PATH_FUNCTS["segmentation_mask"](video_id, frame_idx, frame_id, segmentation_mask_version)
#                     if isfile(segmentation_mask_path):
#                         if channel not in av:
#                             av[channel] = []
                        
#                         if len(av[channel]) == 0 or av[channel][-1][-1] != frame_idx:
#                             av[channel].append([frame_idx, frame_idx+1])
#                         else:
#                             av[channel][-1][-1] = frame_idx+1
#                     elif not isdir(dirname(segmentation_mask_path)):
#                         unavailable_channels.add(channel)

#             for object_bbox_version in available_object_bbox_versions:
#                 channel = f"object_bbox_{object_bbox_version}"
#                 if channel not in handled_channels:
#                     if isfile(CHANNEL_FRAME_PATH_FUNCTS["object_bbox"](video_id, frame_idx, frame_id, object_bbox_version)):
#                         if channel not in av:
#                             av[channel] = []
                        
#                         if len(av[channel]) == 0 or av[channel][-1][-1] != frame_idx:
#                             av[channel].append([frame_idx, frame_idx+1])
#                         else:
#                             av[channel][-1][-1] = frame_idx+1

#             if "tracking_bbox" not in handled_channels and "tracking_bbox" not in unavailable_channels:
#                 tracking_bbox_path = CHANNEL_FRAME_PATH_FUNCTS["tracking_bbox"](video_id, frame_idx, frame_id)
#                 if isfile(tracking_bbox_path):
#                     if len(av["tracking_bbox"]) == 0 or av["tracking_bbox"][-1][1] != frame_idx:
#                         av["tracking_bbox"].append([frame_idx, frame_idx+1])
#                     else:
#                         av["tracking_bbox"][-1][1] = frame_idx+1
#                 elif not isdir(dirname(tracking_bbox_path)):
#                     unavailable_channels.add("tracking_bbox")

#             for image_version in available_image_versions:
#                 for hos_version in available_hos_versions:
#                     for min_length in available_tracking_mask_min_lengths: 
                        
#                         found_object_tracks, found_left_hand_tracks, found_right_hand_tracks  = [], [], []
#                         for track_type, track_list in zip(["object", "left_hand", "right_hand"],
#                                                           [found_object_tracks, found_left_hand_tracks, found_right_hand_tracks]):
#                             channel = f"tracking_mask_track={track_type.replace('_', '-')}_image-version={image_version.replace('_', '-')}_min-length={min_length}_{hos_version}"

#                             if channel not in handled_channels and channel not in unavailable_channels:
#                                 if channel not in av:
#                                     av[channel] = []
                                
#                                 tracking_mask_path = CHANNEL_FRAME_PATH_FUNCTS["tracking_mask"](video_id, frame_idx, frame_id, image_version, hos_version, min_length, track_type)
                        
#                                 if isdir(tracking_mask_path):
#                                     for dir_name in os.listdir(join(tracking_mask_path)):
#                                         dir_path = join(tracking_mask_path, dir_name)
#                                         file_path = join(dir_path, f"{frame_id}__{dir_name}.pkl.zip")
#                                         if isfile(file_path):
#                                             track_list.append(dir_name)
#                                 else:
#                                     unavailable_channels.add(channel)

#                                 # slowdown?      
#                                 # track_list.sort()

#                                 if len(track_list) > 0:
#                                     if (len(av[channel]) == 0 or av[channel][-1][1] != frame_idx
#                                         or av[channel][-1][2] != track_list):
#                                         av[channel].append([frame_idx, frame_idx+1, track_list])
#                                     else:
#                                         av[channel][-1][1] = frame_idx+1

#             if "focus" not in handled_channels:
#                 if frame_id in focus_dict:
#                     if len(av["focus"]) == 0 or av["focus"][-1][-1] != frame_idx:
#                         av["focus"].append([frame_idx, frame_idx+1])
#                     else:
#                         av["focus"][-1][-1] = frame_idx+1
    
#     for channel in all_channels:
#         if channel in handled_channels or channel == "focus" or channel not in av:
#             continue
        
#         file_count, last_mod_timestamp = channel_fc_lm[channel]
#         channel_cache_path = get_channel_cache_path(channel)
#         os.makedirs(dirname(channel_cache_path), exist_ok=True)
#         with open(channel_cache_path, "wb") as f:
#             pickle.dump({"file_count": file_count, "last_mod_timestamp": last_mod_timestamp, "data": av[channel]}, f)

#     return av


# def get_video_virtual_availability_data(request, video_id):
#     video_data = {}
#     val_csv = get_dataset_csv("val")
#     val_csv_video = val_csv[val_csv["video_id"] == video_id]

#     av = get_video_virtual_availability_aux(request, video_id, val_csv_video)
#     av["image"].append([0, LARGE])
#     video_data["virtual_availability"] = av
#     video_data["video_id"] = video_id
#     return JsonResponse(video_data)


# def get_detailed_video_virtual_availability_data(request, video_id):
#     video_data = {}
#     val_csv = get_dataset_csv("val")
#     val_csv_video = val_csv[val_csv["video_id"] == video_id]

#     start_frame = int(request.GET.get("start_frame", 0))
#     end_frame = int(request.GET.get("end_frame", LARGE))
#     av = get_availability_data(video_id, start_frame, end_frame, request)
#     video_data["virtual_availability"] = av
#     video_data["video_id"] = video_id
#     video_data["start_frame"] = start_frame
#     video_data["end_frame"] = end_frame
#     return JsonResponse(video_data)


# def get_video_data(request, video_id):
#     video_data = {}
#     reader = VideoReader(get_video_path(video_id), get_extracted_frame_dir_path(video_id), assumed_fps=EK_ASSUMED_FPS)
#     video_data["video_id"] = video_id
#     video_data["num_real_frames"] = len(reader)
#     video_data["real_fps"] = reader.fps
#     val_csv = get_dataset_csv("val")
#     val_csv_video = val_csv[val_csv["video_id"] == video_id]
#     video_data["num_activities"] = len(val_csv_video)
#     video_data["virtual_fps"] = EK_ASSUMED_FPS
#     video_data["num_virtual_frames"] = reader.get_virtual_frame_count()
#     video_data["available_channels"] = list(get_all_channels())
#     video_data["comments"] = {}

#     # check comments
#     comment_dir_path = CHANNEL_VIDEO_PATH_FUNCTS["comment"](video_id)
#     for root, dirs, files in os.walk(comment_dir_path):
#         for fn in files:
#             if not fn.endswith(".pkl"):
#                 continue

#             path = join(root, fn)
#             with open(path, "rb") as f:
#                 video_data["comments"][fn.replace(".pkl", "")] = pickle.load(f)
    
#     return JsonResponse(video_data)


# def get_video_frame(request, video_id, frame_idx):
#     def get_tracking_id_text(id):
#         return id[:TRACKING_ID_SUBSTRING_LENGTH]

#     def get_color_from_track_id(id):
#         spl = id.split("-")
#         return (int(spl[0], base=16) % 255, int(spl[1], base=16) % 255, int(spl[2], base=16) % 255)

#     def draw_text_box(box, text, border_color, universal_mask, text_bg_color=(0, 0, 0), do_scale_box=True, do_draw_text=True,
#                       orig_box_width=None, orig_box_height=None):
#         nonlocal base_image, draw
#         # draw box

#         if do_draw_text:
#             box_padding_hor = int(10 * math.sqrt(x_ratio))
#             box_padding_ver = 5
#             scaled_box = scale_box_2(box, orig_width=orig_box_width, orig_height=orig_box_height) if do_scale_box else box
#             draw.rectangle(scaled_box, outline=border_color, width=2)

#             bbox = font.getbbox(text)
#             ibbox = [int(coord) for coord in bbox]

#             for vert_offset_idx in range(100):
#                 offset = (int(scaled_box[0]), max(0, int(scaled_box[1]) - (ibbox[3]-ibbox[1]+2*box_padding_ver+5) + vert_offset_idx * 20))
#                 mask_x_start = max(0, offset[1]+ibbox[1]-box_padding_ver)
#                 mask_x_end = min(universal_mask.shape[0], offset[1]+ibbox[3]+box_padding_ver)
#                 mask_y_start = max(0, offset[0]+ibbox[0]-box_padding_hor)
#                 mask_y_end = min(universal_mask.shape[1], offset[0]+ibbox[2]+box_padding_hor)

#                 if np.any(universal_mask[mask_x_start:mask_x_end, mask_y_start:mask_y_end]):
#                     continue

#                 mask = np.zeros((base_image.height, base_image.width), dtype=bool)
#                 mask[mask_x_start:mask_x_end, mask_y_start:mask_y_end] = True
#                 break

#             universal_mask |= mask

#             base_image = superimpose_colored_mask(base_image, mask, color=text_bg_color, alpha=128)
#             draw = ImageDraw.Draw(base_image, "RGBA")
#             draw.text((offset[0], offset[1]), text, font=font)

#     available_hos_versions = get_available_hos_versions()

#     channels = [strp for s in request.GET["channels"].split(",") if len(strp := s.strip()) > 0 ] if "channels" in request.GET else []
#     tracking_bbox_channels = [strp for s in request.GET["tracking_channels"].split(",") if len(strp := s.strip()) > 0 ] if "tracking_channels" in request.GET else []
    
#     frame_id = fmt_frame(video_id, frame_idx)
    
#     base_image = None
#     reader = None
#     try:
#         reader = VideoReader(get_video_path(video_id), get_extracted_frame_dir_path(video_id), assumed_fps=EK_ASSUMED_FPS, chunk_size=1)
#         orig_image_width = reader.video_width
#         orig_image_height = reader.video_height
#     except Exception as ex:
#         print(ex)
#         orig_image_width = 1920
#         orig_image_height = 1080

#     print(f"{channels=}")
#     if "image" in channels:
#         image_version = request.GET.get("image_version", "image")
#         if image_version.startswith("inpainted_") or image_version.startswith("depth_"):
#             path_funct_channel = "inpainted" if image_version.startswith("inpaint") else "depth_postprocessing"

#             version = "_".join(image_version.split("_")[1:])
#             img_path = CHANNEL_FRAME_PATH_FUNCTS[path_funct_channel](video_id, frame_idx, frame_id, version)
#             if isfile(img_path):
#                 try:
#                     with Image.open(img_path) as img:
#                         base_image = img.copy().convert("RGBA")
#                 except Exception as ex:
#                     print(ex)
#                     pass

#             if "depth" in image_version:
#                 depth_top_path = join(ROOT_PATH, "data", "EK_depth", video_id)
#                 print(f"{depth_top_path=}")
#                 _, original_frame_idx = reader.get_frame(frame_idx, return_real_frame_idx=True)
                
#                 for depth_top_subdir in os.listdir(depth_top_path):
#                     depth_top_subdir_path = join(depth_top_path, depth_top_subdir)
#                     if not isdir(depth_top_subdir_path):
#                         continue


#                     original_frame_delta_idx = next((int(spl[2:]) for spl in depth_top_subdir.split("_") if spl.startswith("OS")), None)
#                     aggregated_output_dirname = next(filter(lambda n: n.startswith("R"), os.listdir(depth_top_subdir_path)), None)
#                     if aggregated_output_dirname is not None and original_frame_delta_idx is not None:
#                         print(f"{depth_top_subdir_path=} {aggregated_output_dirname=}")
#                         frame_depth_path = join(depth_top_subdir_path, aggregated_output_dirname,
#                                                 "StD100.0_StR1.0_SmD0_SmR0.0", "depth_e0000", "e0000_filtered", "depth",
#                                                 f"frame_{'%06d' % (original_frame_idx-original_frame_delta_idx)}.raw")
#                         if isfile(frame_depth_path):
#                             print(f"{frame_depth_path=} {depth_top_subdir=} {original_frame_delta_idx=}")
#                             disparity_data = load_raw_float32_image(frame_depth_path)
#                             depth_w = disparity_data.shape[1]
#                             depth_h = disparity_data.shape[0]

#                             depth = ((1.0 / disparity_data) * (3.0 / 255.0) * 65535.0)
                            
#                             depth_out_path = join(ROOT_PATH, "data", "EK_temp_depth", video_id, f"{frame_id}.png")
#                             os.makedirs(dirname(depth_out_path), exist_ok=True)

#                             depth_res = cv2.resize(depth, (853, 480))
#                             imageio.imwrite(depth_out_path, depth_res.astype(np.uint16))
#         else:
#             try:
#                 base_image = Image.fromarray(reader.get_frame(frame_idx)).convert("RGBA")
#             except Exception as ex:
#                 print(ex)
#                 pass
#     else:
#         base_image = Image.new(mode="RGBA", size=(orig_image_width, orig_image_height), color=(51, 51, 51))

#     if base_image is None:
#         base_image = Image.fromarray(cv2.imread(join(ROOT_PATH, "data", "etc", "no_image.png"))).convert("RGBA")
    
#     x_ratio = base_image.width / orig_image_width
#     y_ratio = base_image.height / orig_image_height

#     font = ImageFont.truetype(join("fonts", "DejaVuSans.ttf"), int(40 * math.sqrt(x_ratio)))

#     def scale_box_2(box, orig_width=None, orig_height=None):
#         xr = base_image.width / (orig_width or orig_image_width)
#         yr = base_image.height / (orig_height or orig_image_height)
#         return (box[0] * xr, box[1] * yr, box[2] * xr, box[3] * yr)

#     # keep this here
#     universal_mask = np.zeros((base_image.height, base_image.width), dtype=bool)

#     if "tracking_bbox" in channels:
#         tracking_bbox_filter = [st.lower() for s in request.GET.get("tracking_bbox_filter", "").split(",") if len(st := s.strip()) > 0]
#         tracking_bbox_data_path = CHANNEL_FRAME_PATH_FUNCTS["tracking_bbox"](video_id, frame_idx, frame_id)
#         if isfile(tracking_bbox_data_path):
#             tracking_bbox_data = read_pkl(tracking_bbox_data_path)
#         else:
#             tracking_bbox_data = None
#     else:
#         tracking_bbox_filter = []
#         tracking_bbox_data = None

#     if "tracking_mask" in channels:
#         rgb_img_out_path = join(ROOT_PATH, "data", "EK_temp_rgb", video_id, f"{frame_id}.png")
#         os.makedirs(join(dirname(rgb_img_out_path)), exist_ok=True)
#         base_image.resize((853, 480)).save(rgb_img_out_path)

#         def get_track_data(postprocessing_data, postprocessing_summary_data):
#             top_iou_class = None
#             tortuosity = float(postprocessing_data.get("tortuosity", "NaN"))
#             tortuosity_ema = float(postprocessing_data.get("tortuosity_ema", "NaN"))
#             suffix = f'T: {"%.2f" % tortuosity_ema}'
#             if postprocessing_summary_data is not None and "track_intersection_bbox_ious" in postprocessing_summary_data:
#                 ious = postprocessing_summary_data["track_intersection_bbox_ious"]
#                 ious_keys_sorted = [k for k in sorted(list(ious.keys()), key=lambda k: np.sum(ious[k])) if k != "person"]
#                 # ious_sorted = {k: ious[k] for k in ious_keys_sorted}
#                 top_iou_class = ious_keys_sorted[-1]
#                 suffix += "; CLS: " + top_iou_class
#             cd = float(postprocessing_data["cd_avg"]) if "cd_avg" in postprocessing_data else np.nan
#             cd_90 = float(postprocessing_data["cd_avg_quantiles_ema"][90]) if "cd_avg_quantiles_ema" in postprocessing_data else np.nan
            
#             hands_iou = val if (val := postprocessing_data.get("track_hos_hands_intersection_iou", np.nan)) is not None else np.nan
#             hands_ioa = val if (val := postprocessing_data.get("track_hos_hands_intersection_ioa", np.nan)) is not None else np.nan
#             hands_iou_ema = val if (val := postprocessing_data.get("track_hos_hands_intersection_iou_ema", np.nan)) is not None else np.nan
#             hands_ioa_ema = val if (val := postprocessing_data.get("track_hos_hands_intersection_ioa_ema", np.nan)) is not None else np.nan

#             object_iou = val if (val := postprocessing_data.get("track_hos_object_intersection_iou", np.nan)) is not None else np.nan
#             object_ioa = val if (val := postprocessing_data.get("track_hos_object_intersection_ioa", np.nan)) is not None else np.nan
#             object_iou_ema = val if (val := postprocessing_data.get("track_hos_object_intersection_iou_ema", np.nan)) is not None else np.nan
#             object_ioa_ema = val if (val := postprocessing_data.get("track_hos_object_intersection_ioa_ema", np.nan)) is not None else np.nan

#             if not np.isnan(hands_ioa_ema):
#                 suffix += f'; HIoA: {"%.4f" % hands_ioa_ema}'

#             if not np.isnan(cd_90):
#                 suffix += f'; CDQ90: {"%.4f" % cd_90}'
                
#             filter_passed = not (np.isnan(tortuosity) or np.isnan(tortuosity_ema) or tortuosity_ema > max_tortuosity
#                                  or cd_90 > max_cd or (not np.isnan(hands_ioa_ema) and hands_ioa_ema > max_ema_hand_ioa))
            
#             return tortuosity, tortuosity_ema, top_iou_class, cd, cd_90,\
#                    hands_iou, hands_ioa, hands_iou_ema, hands_ioa_ema,\
#                    object_iou, object_ioa, object_iou_ema, object_ioa_ema,\
#                    suffix, filter_passed

#         tracking_mask_filter = [st.lower() for s in request.GET.get("tracking_mask_filter", "").split(",") if len(st := s.strip()) > 0]
#         tracking_mask_types = [st.lower() for s in request.GET.get("tracking_mask_types", "object").split(",") if len(st := s.strip()) > 0]
#         hos_version = request.GET.get("hos_version", DEFAULT_HOS_VERSION)
#         image_version = request.GET.get("image_version", "image")
#         min_length = request.GET.get("tracking_mask_min_length", DEFAULT_TRACKING_MASK_MIN_LENGTH)
#         max_tortuosity = float(request.GET.get("tracking_mask_max_tortuosity", 1.0))
#         max_cd = float(request.GET.get("tracking_mask_max_cd", np.inf))
#         merge_tracking_mask_tracks = int(request.GET.get("tracking_mask_merge_tracks", 0)) == 1
#         max_ema_hand_ioa = float(request.GET.get("tracking_mask_max_ema_hand_ioa", np.inf))
#         # build merging dict
#         tracking_mask_merge_dict = {}
#         tracking_mask_merge_dict_undir = {}
#         tracking_summaries = {}
#         tracking_postprocessing_data = {}
#         tracking_data = {}
#         tracks_with_nonzero_masks = {}  # tracking_type: set()
#         track_ids = set()

#         tracking_mask_img = Image.new("L", (853, 480), (0))
#         tracking_mask_draw = ImageDraw.Draw(tracking_mask_img)

#         for track_type in tracking_mask_types:
#             tracks_with_nonzero_masks[track_type] = set()
#             tracking_mask_path = CHANNEL_FRAME_PATH_FUNCTS["tracking_mask"](video_id, frame_idx, frame_id, image_version, hos_version, min_length, track_type)
#             if isdir(tracking_mask_path):
#                 for root, dirs, files in os.walk(tracking_mask_path):
#                     for dir_name in sorted(dirs):
#                         track_ids.add(dir_name)
#                         # dir_name is track ID
                        
#                         # don't apply filtering to merging
#                         #if len(tracking_mask_filter) > 0 and not any((dir_name.startswith(fs) for fs in tracking_mask_filter)):
#                         #    continue

#                         dir_path = join(tracking_mask_path, dir_name)
#                         file_path = join(dir_path, f"{frame_id}__{dir_name}.pkl.zip")
#                         color = get_color_from_track_id(dir_name)
#                         if isfile(file_path):
#                             tracking_mask_data = read_pkl(file_path)
#                             mask = tracking_mask_data["masks"] > 0
#                             if mask.max():
#                                 tracks_with_nonzero_masks[track_type].add(dir_name)
#                             tracking_data[dir_name] = tracking_mask_data

#                         postprocessing_dir = CHANNEL_FRAME_PATH_FUNCTS["tracking_mask_postprocessing"](video_id, frame_idx, frame_id, image_version, hos_version, min_length, track_type)
#                         postprocessing_frame_path = join(postprocessing_dir, dir_name, f"{frame_id}__{dir_name}.pkl")
#                         postprocessing_summary_path = join(postprocessing_dir, dir_name, "_summary.pkl")


#                         if isfile(postprocessing_frame_path):
#                             tracking_postprocessing_data[dir_name] = read_pkl(postprocessing_frame_path)

#                         if isfile(postprocessing_summary_path):
#                             summary = read_pkl(postprocessing_summary_path)
#                             tracking_summaries[dir_name] = summary

#                             # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#                             # if "merge_track_into" in summary:
#                             #     tracking_mask_merge_dict[dir_name] = set(summary["merge_track_into"])

#                             #     if dir_name not in tracking_mask_merge_dict_undir:
#                             #         tracking_mask_merge_dict_undir[dir_name] = set()
#                             #     tracking_mask_merge_dict_undir[dir_name].update(summary["merge_track_into"])
#                             #     for target_id in summary["merge_track_into"]:
#                             #         if target_id not in tracking_mask_merge_dict_undir:
#                             #             tracking_mask_merge_dict_undir[target_id] = set()
#                             #         tracking_mask_merge_dict_undir[target_id].add(dir_name)
#                             # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

#         # calculate which tracks to merge
        
#         print(f"{tracking_summaries.keys()=}")

#         tracking_mask_merging_scheme = [strp for s in request.GET.get("tracking_mask_merging_scheme", "iou").split(",") if len(strp := s.strip()) > 0]
#         min_iou = float(request.GET.get("tracking_mask_merging_min_iou", DEFAULT_TRACKING_MASK_MERGING_OVERLAP_FRAME_IOU_FRACTION))
#         min_ioa = float(request.GET.get("tracking_mask_merging_min_ioa", DEFAULT_TRACKING_MASK_MERGING_OVERLAP_FRAME_IOA_FRACTION))
#         min_frame_fraction = float(request.GET.get("tracking_mask_merging_min_fraction", DEFAULT_TRACKING_MASK_MERGING_OVERLAP_FRAME_COUNT_FRACTION))
#         print(f"{merge_tracking_mask_tracks=}")
#         if merge_tracking_mask_tracks:
#             for track_id, tracking_summary in tracking_summaries.items():
#                 if "track_mask_appearances_filter2_passed" not in tracking_summary:
#                     continue
#                 this_track_appearance_count = len(tracking_summary["track_mask_appearances_filter2_passed"])
#                 print(f'{tracking_summary["track_track_intersection_ious"].items()=}')

#                 for other_track_id, frames_ious in tracking_summary["track_track_intersection_ious"].items():
#                     if "track_track_intersection_ioas" in tracking_summary:
#                         frames_ioas = tracking_summary["track_track_intersection_ioas"].get(other_track_id, {})
#                     else:
#                         frames_ioas = {}

#                     print(f"{track_id=} {other_track_id=}")
                    
#                     num_iou_overlaps = sum((1 for v in frames_ious.values() if v >= min_iou))
#                     num_ioa_overlaps = sum((1 for v in frames_ioas.values() if v >= min_ioa))
#                     num_iou_ioa_overlaps = sum((1 for k, iou in frames_ious.items() if iou >= min_iou and k in frames_ioas and frames_ioas[k] >= min_ioa))

#                     print(f"{tracking_mask_merging_scheme=}")

#                     if "iou" in tracking_mask_merging_scheme and "ioa" in tracking_mask_merging_scheme:
#                         num_filtered_overlaps = num_iou_ioa_overlaps
#                     elif "iou" in tracking_mask_merging_scheme:
#                         num_filtered_overlaps = num_iou_overlaps
#                     elif "ioa" in tracking_mask_merging_scheme:
#                         num_filtered_overlaps = num_ioa_overlaps
#                     else:
#                         num_filtered_overlaps = len(frames_ious)

#                     overlap_fraction = num_filtered_overlaps / max(1, this_track_appearance_count)
#                     overlap_fraction_reverse = -1.0 

#                     if overlap_fraction >= min_frame_fraction:
#                         if dir_name not in tracking_mask_merge_dict:
#                             tracking_mask_merge_dict[dir_name] = set()
#                         tracking_mask_merge_dict[dir_name].add(other_track_id)

#                         print(f"Merging {dir_name} into {other_track_id}")
                    
#                     if other_track_id in tracking_summaries:
#                         other_tracking_summary = tracking_summaries[other_track_id]
#                         if "track_mask_appearances_filter2_passed" not in other_tracking_summary:
#                             continue

#                         other_track_appearance_count = len(other_tracking_summary["track_mask_appearances_filter2_passed"])
#                         overlap_fraction_reverse = num_filtered_overlaps / max(1, other_track_appearance_count)
#                         if overlap_fraction_reverse >= min_frame_fraction:
#                             if other_track_id not in tracking_mask_merge_dict:
#                                 tracking_mask_merge_dict[other_track_id] = set()
#                             tracking_mask_merge_dict[other_track_id].add(dir_name)

#                     if max(overlap_fraction, overlap_fraction_reverse) >= min_frame_fraction:
#                         if dir_name not in tracking_mask_merge_dict_undir:
#                             tracking_mask_merge_dict_undir[dir_name] = set()
#                         tracking_mask_merge_dict_undir[dir_name].add(other_track_id)

#                         if other_track_id not in tracking_mask_merge_dict_undir:
#                             tracking_mask_merge_dict_undir[other_track_id] = set()
#                         tracking_mask_merge_dict_undir[other_track_id].add(dir_name)

#         print(f"{tracking_mask_merge_dict_undir=}")
#         union_find_links = {}

#         def find(item_id):
#             while union_find_links.get(item_id, item_id) != item_id:
#                 item_id = union_find_links.get(item_id, item_id)
#             return item_id

#         def union(id_1, id_2):
#             id_1_f = find(id_1)
#             id_2_f = find(id_2)
#             if id_1_f == id_2_f:
#                 return
#             if id_1_f < id_2_f:
#                 union_find_links[id_2_f] = id_1_f
#             else:
#                 union_find_links[id_1_f] = id_2_f

#         for track_id, merge_with in tracking_mask_merge_dict_undir.items():
#             for other_track_id in merge_with:
#                 union(other_track_id, track_id)

#         print(f"{union_find_links=}")

#         for track_type in tracking_mask_types:
#             tracking_mask_path = CHANNEL_FRAME_PATH_FUNCTS["tracking_mask"](video_id, frame_idx, frame_id, image_version, hos_version, min_length, track_type)
#             if isdir(tracking_mask_path):
#                 #for root, dirs, files in os.walk(tracking_mask_path):
#                 #    for dir_name in sorted(dirs):
#                 for dir_name in tracking_data.keys():
#                         # dir_name is track ID
#                         dir_path = join(tracking_mask_path, dir_name)
#                         file_path = join(dir_path, f"{frame_id}__{dir_name}.pkl.zip")
#                         color = get_color_from_track_id(dir_name)
#                         if dir_name in tracking_data:
#                             #tracking_mask_data = read_pkl(file_path)
#                             tracking_mask_data = tracking_data[dir_name]

#                             exclude_main = False
#                             mask = np.zeros(tracking_mask_data["masks"].shape, dtype=bool)
                            
#                             if len(tracking_mask_filter) > 0 and not any((dir_name.startswith(fs) for fs in tracking_mask_filter)):
#                                 exclude_main = True
                            
#                             tracks_merged_into_this = []
#                             if merge_tracking_mask_tracks:
#                                 merge_this_track_into = find(dir_name)
#                                 if merge_this_track_into != dir_name:
#                                     print(f"Merged {dir_name} into {merge_this_track_into}; ignoring")
#                                     continue

#                                 if not (len(tracking_mask_filter) > 0 and not any((dir_name.startswith(fs) for fs in tracking_mask_filter))):
#                                     # check which tracks were merged into this track
#                                     for candidate_track_id in union_find_links.keys():
#                                         if find(candidate_track_id) == dir_name:
#                                             candidate_track_in_this_frame = (candidate_track_id in tracking_data
#                                                                             and candidate_track_id in tracking_summaries
#                                                                             and frame_id in tracking_summaries[candidate_track_id].get("track_mask_appearances_filter2_passed", []))
                                            
#                                             tortuosity, tortuosity_ema, top_iou_class, cd, cd_90,\
#                                             hands_iou, hands_ioa, hands_iou_ema, hands_ioa_ema,\
#                                             object_iou, object_ioa, object_iou_ema, object_ioa_ema,\
#                                             suffix, filter_passed = get_track_data(tracking_postprocessing_data.get(candidate_track_id, {}),
#                                                                                    tracking_summaries.get(candidate_track_id, {}))
                                            
#                                             if not filter_passed:
#                                                 continue

#                                             tracks_merged_into_this.append(("*" if candidate_track_in_this_frame else "") + candidate_track_id)
#                                             if candidate_track_in_this_frame:
#                                                 print(f"candidate_track_in_this_frame: {candidate_track_id} for current {dir_name}")
#                                                 mask |= tracking_data[candidate_track_id]["masks"] > 0

#                             # check if we have tortuosity info
#                             # lambda video_id, frame_idx, frame_id, image_version, hos_version, min_length, track_type:
#                             # join(TRACKING_MASK_POSTPROCESSING_DATA_DIR, image_version.replace("inpainted_", ""), hos_version, f"min-length={min_length}", video_id, track_type),
                            
#                             # postprocessing_dir = CHANNEL_FRAME_PATH_FUNCTS["tracking_mask_postprocessing"](video_id, frame_idx, frame_id, image_version, hos_version, min_length, track_type)
#                             # postprocessing_summary_path = join(postprocessing_dir, dir_name, "_summary.pkl")
#                             # if isfile(postprocessing_summary_path):
#                             #     postprocessing_summary_data = read_pkl(postprocessing_summary_path)
#                             # else:
#                             #     postprocessing_summary_data = None

#                             if dir_name in tracking_summaries:
#                                 postprocessing_summary_data = tracking_summaries[dir_name]
#                             else:
#                                 postprocessing_summary_data = None

#                             postprocessing_data_path = join(postprocessing_dir, dir_name, f"{frame_id}__{dir_name}.pkl")
#                             suffix = ""
#                             if len(tracks_merged_into_this) > 0:
#                                 suffix = ", ".join([track_id[:(TRACKING_ID_SUBSTRING_LENGTH + (1 if (track_id + " ")[0] == "*" else 0))] for track_id in tracks_merged_into_this])
#                             #if isfile(postprocessing_data_path):
#                             #    postprocessing_data = read_pkl(postprocessing_data_path)
#                             if dir_name in tracking_postprocessing_data:
#                                 postprocessing_data = tracking_postprocessing_data[dir_name]
#                                 # TODO: refactor into new function (also in postprocess_tracking_masks.py)

#                                 tortuosity, tortuosity_ema, top_iou_class, cd, cd_90,\
#                                 hands_iou, hands_ioa, hands_iou_ema, hands_ioa_ema,\
#                                 object_iou, object_ioa, object_iou_ema, object_ioa_ema,\
#                                 suffix, filter_passed = get_track_data(postprocessing_data, postprocessing_summary_data)

#                                 if not filter_passed:
#                                     exclude_main = True

#                                 #q10_bottom = postprocessing_summary_data["track_mask_bbox_bottoms_quantiles_rel"][10] if postprocessing_summary_data is not None else -np.inf
#                                 #if q10_bottom >= 0.0:
#                                 #    suffix += f'; BTQ10: {"%.2f" % q10_bottom}'
#                                 #q10_bottom = postprocessing_summary_data[""]
#                             else:
#                                 tortuosity = None
#                                 top_iou_class = None

#                             if not exclude_main:
#                                 mask |= tracking_mask_data["masks"] > 0

#                             if not mask.max():
#                                 continue

#                             base_image = superimpose_colored_mask(base_image, mask, color=color, alpha=128)
#                             tracking_mask_img = superimpose_colored_mask(tracking_mask_img, mask, color=(255, 255, 255), alpha=255).convert("L")
#                             draw = ImageDraw.Draw(base_image, "RGBA")
#                             box = bbox_from_mask(mask)
                        
#                             tracking_mask_width_ratio = base_image.width / mask.shape[1]
#                             tracking_mask_height_ratio = base_image.height / mask.shape[0]
#                             scaled_box = [box[0] * tracking_mask_width_ratio,
#                                           box[1] * tracking_mask_height_ratio,
#                                           box[2] * tracking_mask_width_ratio,
#                                           box[3] * tracking_mask_height_ratio]
                        
#                             if suffix not in ["", None]:
#                                 suffix = f" ({suffix})"

#                             type_abbr = track_type[0].upper()
#                             draw_text_box(scaled_box,
#                                           (f"{type_abbr}: {get_tracking_id_text(dir_name)}"
#                                            + suffix),
#                                            color, universal_mask, color, do_scale_box=False,
#                                            do_draw_text="box_labels" in channels)
#             else:
#                 print("not isdir")

#         tracking_mask_img_out_path = join(ROOT_PATH, "data", "EK_temp_tracking_masks", video_id, f"{frame_id}.png")
#         os.makedirs(join(dirname(tracking_mask_img_out_path)), exist_ok=True)
#         tracking_mask_img.save(tracking_mask_img_out_path)


#     draw = ImageDraw.Draw(base_image, "RGBA")

#     hand_clearance_box = None

#     def recompute_hand_clearance_box():
#         nonlocal hand_clearance_box
#         if "filter_hand_border_clearance" not in request.GET:
#             return
#         filter_hand_border_clearance = float(request.GET["filter_hand_border_clearance"])
#         if filter_hand_border_clearance == 0.0:
#             return
        
#         min_dim = min(base_image.width, base_image.height)
#         hand_clearance_box = (filter_hand_border_clearance * min_dim, filter_hand_border_clearance * min_dim,
#                 base_image.width - filter_hand_border_clearance * min_dim,
#                 base_image.height - filter_hand_border_clearance * min_dim)

#     if "filter_hand_border_clearance" in request.GET:
#         recompute_hand_clearance_box()
    
#     if "hand_mesh" in channels:
#         hand_mesh_version = request.GET.get("hand_mesh_version", DEFAULT_HAND_MESH_VERSION)
#         vis_path = CHANNEL_FRAME_PATH_FUNCTS["hand_mesh_vis"](video_id, frame_idx, frame_id, hand_mesh_version)
#         if not isfile(vis_path):
#             vis_path = vis_path.replace(".jpg", ".png")
#         data_path = CHANNEL_FRAME_PATH_FUNCTS["hand_mesh"](video_id, frame_idx, frame_id, hand_mesh_version)
#         if isfile(vis_path) and isfile(data_path):
#             img_array = cv2.imread(vis_path)[:,:,::-1]
#             array_mesh = img_array[:,img_array.shape[1]//2:]
#             if array_mesh.shape[-1] == 3:
#                 array_mesh = np.concatenate((array_mesh, 255 * np.ones((*array_mesh.shape[:-1], 1))), axis=-1)
            
#             array_mesh[np.amax(array_mesh[:,:,:-1], axis=-1) < 10] = 0

#             mesh_img = Image.fromarray(array_mesh.astype(np.uint8)).convert("RGBA")

#             base_image = base_image.convert("RGBA")
#             if base_image.size != mesh_img.size:
#                 mesh_img = mesh_img.resize(base_image.size, Image.BILINEAR)
            
#             recompute_hand_clearance_box()

#             base_image = Image.alpha_composite(base_image, mesh_img)
#             base_image = base_image
#             draw = ImageDraw.Draw(base_image, "RGBA")

#             try:
#                 data = read_pkl(data_path)
#                 if isinstance(data["pred_output_list"], list):
#                     pred_output_list = data["pred_output_list"][0]
#                 elif isinstance(data["pred_output_list"], dict):
#                     pred_output_list = data["pred_output_list"]
#                 else:
#                     pred_output_list = None
#             except ToggleableException as ex:
#                 print(f"Exception reading {data_path}:", ex)
#                 pred_output_list = None

            
#             if pred_output_list is not None:
#                 for hand_side in pred_output_list.keys():
#                     if pred_output_list[hand_side].get("outlier", False) and "bbox_top_left" in pred_output_list[hand_side]:
#                         # "bbox_scale_ratio": scale factor to convert from original to cropped
#                         x1 = pred_output_list[hand_side]["bbox_top_left"][0]
#                         y1 = pred_output_list[hand_side]["bbox_top_left"][1]
#                         orig_w = pred_output_list[hand_side]["img_cropped"].shape[0] / pred_output_list[hand_side]["bbox_scale_ratio"]
#                         orig_h = pred_output_list[hand_side]["img_cropped"].shape[1] / pred_output_list[hand_side]["bbox_scale_ratio"]
#                         orig_rescaled_bbox = [x1, y1, x1 + orig_w, y1 + orig_h]
#                         # original image was thumbnailed
#                         orig_rescaled_width = data.get("image_width", 1280)
#                         orig_rescaled_height = data.get("image_height", 720)

#                         orig_backscaled_bbox = scale_box(orig_rescaled_bbox, orig_rescaled_width, orig_rescaled_height,
#                                                          target_width=base_image.width, target_height=base_image.height)
#                         draw.rectangle(orig_backscaled_bbox, outline=(255, 0, 0), width=2)

#             if ("hand_mesh_history" in request.GET and "hand_mesh_step" in request.GET
#                 and any(hand_str in request.GET and request.GET[hand_str] == "1" for hand_str in ["hand_mesh_track_left", "hand_mesh_track_right"])):
#                 hand_mesh_history = int(request.GET["hand_mesh_history"])
#                 hand_mesh_step = int(request.GET["hand_mesh_step"])

#                 pred_joints_img_history = {"left_hand": [], "right_hand": []}

#                 for step_idx in range(0, hand_mesh_history):
#                     offset = step_idx * hand_mesh_step
#                     found_left = False
#                     found_right = False
#                     for extra_offset in range(0, hand_mesh_step):
#                         target_frame_idx = frame_idx - offset - extra_offset
#                         if target_frame_idx < 0:
#                             break

#                         target_hand_frame_id = fmt_frame(video_id, target_frame_idx)
#                         target_hand_info_path = CHANNEL_FRAME_PATH_FUNCTS["hand_mesh"](video_id, target_frame_idx, target_hand_frame_id, hand_mesh_version)

#                         if not isfile(target_hand_info_path):
#                             continue

#                         pkl = read_pkl(target_hand_info_path)

#                         for hand_data_container in pkl["pred_output_list"]:
#                             for hand in ["left_hand", "right_hand"]:
#                                 if hand in hand_data_container and hand_data_container[hand] is not None and "pred_joints_img" in hand_data_container[hand]:
#                                     if hand == "left_hand":
#                                         if found_left:
#                                             continue
#                                         else:
#                                             found_left = True
#                                     elif hand == "right_hand":
#                                         if found_right:
#                                             continue
#                                         else:
#                                             found_right = True
#                                     pred_joints_img_history[hand].append((offset+extra_offset, hand_data_container[hand]["pred_joints_img"]))
                        
#                         if found_left and found_right:
#                             break

#                 for hand, hand_color in [("left_hand", np.array([0, 0, 255])), ("right_hand", np.array([255, 0, 0]))]:
#                     hand_str = "hand_mesh_track_" + hand.split('_')[0]
#                     if not (hand_str in request.GET and request.GET[hand_str] == "1"):
#                         continue
                    
#                     hist = pred_joints_img_history[hand]
#                     hist_length = len(hist)

#                     overlay = Image.new("RGBA", base_image.size, (0, 0, 0, 0))
#                     overlay_draw = ImageDraw.Draw(overlay)

#                     if hist_length > 0:
#                         prev_offset, prev_pred_joints_img = (None, None)
#                         for step_idx, hist_entry in enumerate(hist):
#                             offset, pred_joints_img = hist_entry
#                             # base_image already scaled to size that was used for this estimate
#                             # thumb to little finger: 4, 8, 12, 16, 20
#                             # "#0673B0", "#069E73", "#5BB4E4", "#5AB4E2", "#E5A023"
                            
#                             for joint_name, joint_color in HAND_JOINT_COLORS.items():
#                                 joint_idx = HAND_JOINT_INDICES[joint_name]
#                                 pred_joints_img[joint_idx][0] *= base_image.width / 1280
#                                 pred_joints_img[joint_idx][1] *= base_image.height / 720
#                                 coords = [int(pred_joints_img[joint_idx][0]), int(pred_joints_img[joint_idx][1])]
#                                 ellipse_radius = 8
#                                 ellipse_radius_ext = 10

#                                 alpha = int(255 - ((step_idx) / max(1, hist_length - 1)) * 127)

#                                 if prev_pred_joints_img is not None:
#                                     overlay_draw.line((*prev_pred_joints_img[joint_idx][:2], *coords), fill=(*hex_to_rgb(joint_color), alpha))

#                                 overlay_draw.ellipse((coords[0]-ellipse_radius_ext, coords[1]-ellipse_radius_ext,
#                                                      coords[0]+ellipse_radius_ext, coords[1]+ellipse_radius_ext), fill=(*tuple(hand_color.tolist()), alpha))
#                                 overlay_draw.ellipse((coords[0]-ellipse_radius, coords[1]-ellipse_radius,
#                                                      coords[0]+ellipse_radius, coords[1]+ellipse_radius), fill=(*hex_to_rgb(joint_color), alpha))
                            
#                             prev_offset, prev_pred_joints_img = offset, pred_joints_img

#                     base_image = Image.alpha_composite(base_image, overlay)
#                     draw = ImageDraw.Draw(base_image, "RGBA")

#     recompute_hand_clearance_box()

#     if "hos" in channels or "hos_hands" in channels or "hos_object" in channels:
#         hos_version = request.GET.get("hos_version", DEFAULT_HOS_VERSION)

#         left_color = (0, 0, 255)
#         right_color = (255, 0, 0)
#         object_color = (255, 255, 0)

#         if hos_version in ["egohos", "ek-gt", "ek-gt-sparse", "ek-gt-dense"]:
#             path_hands = CHANNEL_FRAME_PATH_FUNCTS["hos_hands"](video_id, frame_idx, frame_id, hos_version)
#             path_object = CHANNEL_FRAME_PATH_FUNCTS["hos_object"](video_id, frame_idx, frame_id, hos_version)

#             if isfile(path_hands):
#                 pkl = read_pkl(path_hands)
#                 sum_left = (pkl == 1).sum()
#                 sum_right = (pkl == 2).sum()

#                 if sum_left > 0:
#                     base_image = superimpose_colored_mask(base_image, (pkl == 1), left_color)
#                     draw = ImageDraw.Draw(base_image, "RGBA")
                
#                 if sum_right > 0:
#                     base_image = superimpose_colored_mask(base_image, (pkl == 2), right_color)
#                     draw = ImageDraw.Draw(base_image, "RGBA")

#             print(f"{path_object=} {isfile(path_object)=}")
#             if isfile(path_object):
#                 pkl = read_pkl(path_object)
#                 sum_obj = (pkl > 0).sum()
                
#                 if sum_obj > 0:
#                     base_image = superimpose_colored_mask(base_image, (pkl > 0), object_color)
#                     draw = ImageDraw.Draw(base_image, "RGBA")

#             # TODO: think about how to incorporate tracking bbox data for EgoHOS
#         else:
#             path = CHANNEL_FRAME_PATH_FUNCTS["hos"](video_id, frame_idx, frame_id, hos_version)

#             if isfile(path):
#                 pkl = read_pkl(path)

#                 idx = -1
#                 for cls, handside, mask, box in zip(pkl["instances"].pred_classes, pkl["instances"].pred_handsides, pkl["instances"].pred_masks, pkl["instances"].pred_boxes):
#                     idx += 1
#                     if cls == 0:  # 0: hand
#                         # 0: left; 1: right

#                         if not ("hos" in channels or "hos_hands" in channels):
#                             continue

#                         handside = handside.argmax().item()
#                         color = left_color if handside == 0 else right_color
#                         tbd_channel = "hos_left_hand" if handside == 0 else "hos_right_hand"
#                     else:  # 1: object
#                         if not ("hos" in channels or "hos_object" in channels):
#                             continue

#                         color = object_color
#                         tbd_channel = "hos_object"

#                     tbd_channel += f"_{hos_version}"
                    
#                     if tracking_bbox_data is not None and "tracking_bbox" in channels:  # "hos" in tracking_bbox_channels:
#                         tbd = tracking_bbox_data["frame_original_box_idxs_to_tracks"][tbd_channel]
#                         if idx in tbd:
#                             if len(tracking_bbox_filter) > 0 and not any([t.startswith(c) for t in tbd[idx] for c in tracking_bbox_filter]):
#                                 continue
#                             text = "ID: " + ", ".join(map(get_tracking_id_text, tbd[idx]))
#                             draw_text_box(box, text, color, universal_mask,
#                                           do_draw_text="box_labels" in channels)
#                         elif len(tracking_bbox_filter) > 0:
#                             continue

#                     # resize the mask
#                     base_image = superimpose_colored_mask(base_image, mask, color)
#                     draw = ImageDraw.Draw(base_image, "RGBA")

#     if "hand_bbox" in channels:
#         path = CHANNEL_FRAME_PATH_FUNCTS["hand_bbox"](video_id, frame_idx, frame_id)
#         if isfile(path):
#             with open(path, "rb") as f:
#                 pkl_data = pickle.load(f)
#                 hand_data = pkl_data[2]
#                 if isinstance(hand_data, list):
#                     hand_data = hand_data[0]
                
#                 if isinstance(hand_data, dict):
#                     for hand, hand_color in [("left_hand", "blue"), ("right_hand", "red")]:
#                         if hand in hand_data and hand_data[hand] is not None:
#                             d = hand_data[hand]
                            
#                             box = (d[0], d[1], d[0] + d[2], d[1] + d[3])
#                             text = hand[0].upper() + (" | VoL: %.2f" % hand_data[hand + "_laplacian_var"] if (hand + "_laplacian_var") in hand_data else "")

#                             if tracking_bbox_data is not None and "tracking_bbox" in channels:  # "hand_bbox" in tracking_bbox_channels:
#                                 tbd = tracking_bbox_data["frame_original_box_idxs_to_tracks"]["hand_bbox_" + hand.split("_")[0]]
#                                 if hand in tbd:
#                                     text += " | ID: " + ", ".join(map(get_tracking_id_text, tbd[hand]))

#                                     if len(tracking_bbox_filter) > 0 and not any([t.startswith(c) for t in tbd[hand] for c in tracking_bbox_filter]):
#                                         continue
#                                 elif len(tracking_bbox_filter) > 0:
#                                     continue


#                             scaled_box = scale_box_2(box)
#                             print()
#                             print(f"{hand_clearance_box=}")
#                             print(f"{scaled_box=}")
#                             print()
#                             if hand_clearance_box is not None and not box_a_in_box_b(scaled_box, hand_clearance_box):
#                                 text += " (ign.)"
#                             draw_text_box(box, text, hand_color, universal_mask,
#                                           do_draw_text="box_labels" in channels)

#     if "segmentation_mask" in channels:
#         segmentation_mask_version = request.GET.get("segmentation_mask_version", DEFAULT_HOS_VERSION)
#         channel_name = f"segmentation_mask_{segmentation_mask_version}"
#         path = CHANNEL_FRAME_PATH_FUNCTS["segmentation_mask"](video_id, frame_idx, frame_id, segmentation_mask_version)

#         if isfile(path):
#             pkl = read_pkl(path)

#             keep_box_idx = -1
#             keep_box_area = np.inf

#             for box_idx, box_data in enumerate(pkl):
#                 cls = box_data["cls"]
#                 if cls not in [1, -1]:
#                     continue
#                 box_area = (box_data["box"][2] - box_data["box"][0]) * (box_data["box"][3] - box_data["box"][1])
#                 if box_area < keep_box_area:
#                     keep_box_idx = box_idx
#                     keep_box_area = box_area

#             min_area = int(request.GET.get("filter_min_mask_area", 0))

#             for box_idx, box_data in enumerate(pkl):
#             #if keep_box_idx != -1:
#                 # box_data = pkl[keep_box_idx]
#                 cls = box_data["cls"]
#                 masks = box_data["masks"]
#                 orig_width = box_data.get("image_width", masks[0]["segmentation"].shape[1] if len(masks) > 0 else None)
#                 orig_height = box_data.get("image_height", masks[0]["segmentation"].shape[0] if len(masks) > 0 else None)
#                 if cls in [1, -1]:
#                     outer_box = box_data["box"]

#                     scaled_outer_box = scale_box_2(outer_box,
#                                                    orig_width=orig_width,
#                                                    orig_height=orig_height)
#                     extended_box = (int(scaled_outer_box[0]), int(scaled_outer_box[1]),
#                                     int(math.ceil(scaled_outer_box[2])), int(math.ceil(scaled_outer_box[3])))

#                     # get box
#                     box_img = base_image.crop(extended_box)
                    
#                     texts_to_draw = []

#                     # draw masks
#                     print(f"{outer_box=} {len(masks)=}")
#                     for mask_idx, mask in enumerate(masks):
#                         part_ids = None
#                         #  "segmentation_mask" in tracking_bbox_channels:
#                         if (tracking_bbox_data is not None and "tracking_bbox" in channels
#                             and channel_name in tracking_bbox_data["frame_original_box_idxs_to_tracks"]):
#                             tbd = tracking_bbox_data["frame_original_box_idxs_to_tracks"][channel_name]
#                             if (box_idx, mask_idx) in tbd:
#                                 if len(tracking_bbox_filter) > 0 and not any([t.startswith(c)
#                                        for t in tbd[(box_idx, mask_idx)] for c in tracking_bbox_filter]):
#                                     continue
#                                 part_ids = tbd[(box_idx, mask_idx)]
#                             elif len(tracking_bbox_filter) > 0:
#                                 continue
                        
#                         if mask["area"] < min_area:
#                             continue

#                         seg_mask = mask["segmentation"].toarray()
#                         area = seg_mask.astype(int).sum()
#                         if part_ids is not None and len(part_ids) > 0:
#                             color = get_color_from_track_id(part_ids[0])
#                         else:
#                             color = tuple(np.clip(np.array(hex_to_rgb(PALETTE[(area // 200) % len(PALETTE)])) + np.random.randint(-20, 20, size=3), 0, 255))
                        
#                         box_img = superimpose_colored_mask(box_img, seg_mask, color)
                        
#                         if part_ids is not None:
#                             inner_box = np.array(mask["bbox"])
#                             inner_box[2] += inner_box[0]
#                             inner_box[3] += inner_box[1]
#                             part_bbox_global = np.array([outer_box[0] + inner_box[0], outer_box[1] + inner_box[1],
#                                                          outer_box[0] + inner_box[2], outer_box[1] + inner_box[3]])
#                             border_color = color
#                             texts_to_draw.append({"text": ", ".join(map(get_tracking_id_text, part_ids)),
#                                                   "box": scale_box_2(part_bbox_global,
#                                                                      orig_width=orig_width,
#                                                                      orig_height=orig_height),
#                                                   "border_color": border_color,
#                                                   "use_zero_mask": True})


#                     base_image.paste(box_img.convert(base_image.format), extended_box)

#                     for text_to_draw_data in texts_to_draw:
#                         draw_text_box(text_to_draw_data["box"], text_to_draw_data["text"],
#                                       text_to_draw_data["border_color"],
#                                       (np.zeros_like(universal_mask)
#                                        if text_to_draw_data["use_zero_mask"] else universal_mask),
#                                       do_draw_text="box_labels" in channels)

#                     # draw box
#                     color = (255, 255, 0)
#                     draw.rectangle(scaled_outer_box, outline=color, width=2)

#     if "object_bbox" in channels:
#         object_bbox_version = request.GET.get("object_bbox_version", DEFAULT_OBJECT_BBOX_VERSION)
#         path = CHANNEL_FRAME_PATH_FUNCTS["object_bbox"](video_id, frame_idx, frame_id, object_bbox_version)
#         min_confidence = float(request.GET["filter_object_confidence"]) if "filter_object_confidence" in request.GET else 0.0

#         if isfile(path):
#             with open(path, "rb") as f:
#                 l = pickle.load(f)
#                 print(l)
#                 if "scores" in l:
#                     # draw bboxes
#                     idx = -1
#                     for cls, box, score in zip(l["classes"][0], l["boxes"][0], l["scores"][0]):
#                         idx += 1
#                         if score < min_confidence:
#                             continue
                        
#                         text = cls + " | %.2f" % score

#                         if tracking_bbox_data is not None and "tracking_bbox" in channels: #  "object_bbox" in tracking_bbox_channels:
#                             tbd = tracking_bbox_data["frame_original_box_idxs_to_tracks"]["object_bbox"]
#                             if (0, idx) in tbd:
#                                 text += " | ID: " + ", ".join(map(get_tracking_id_text, tbd[(0, idx)]))

#                                 if len(tracking_bbox_filter) > 0 and not any([t.startswith(c) for t in tbd[(0, idx)] for c in tracking_bbox_filter]):
#                                     continue
#                             elif len(tracking_bbox_filter) > 0:
#                                 continue

#                         border_color = (255, 255, 0)
#                         draw_text_box(box, text, border_color, universal_mask,
#                                       do_draw_text="box_labels" in channels,
#                                       orig_box_width=l.get("image_width"), orig_box_height=l.get("image_height"))
                        
#         else:
#             print(f"Not found: {path}")

#     if tracking_bbox_data is not None and "tracking_bbox" in channels:
#         hos_version = request.GET.get("hos_version", DEFAULT_HOS_VERSION)
        
#         tracking_bbox_channels_repl = [*tracking_bbox_channels]

#         if "hos_hands" in tracking_bbox_channels_repl:
#             tracking_bbox_channels_repl = [*[d for d in tracking_bbox_channels_repl if d != "hos_hands"],
#                                             f"hos_left_hand_{hos_version}"]

#         if "hos" in tracking_bbox_channels_repl:
#             tracking_bbox_channels_repl = [*[d for d in tracking_bbox_channels_repl if d != "hos"],
#                                              f"hos_left_hand_{hos_version}",
#                                              f"hos_right_hand_{hos_version}",
#                                              f"hos_object_{hos_version}"]

#         for tracking_channel in tracking_bbox_channels_repl:
#             if tracking_channel not in tracking_bbox_data["tracks"]:
#                 continue
            
#             tch = tracking_bbox_data["tracks"][tracking_channel]
#             for track_id, track_data in tch.items():
#                 if len(tracking_bbox_filter) > 0 and not any([track_id.startswith(c) for c in tracking_bbox_filter]):
#                     continue

#                 draw_text_box(track_data["box"], tracking_channel + " | ID: " + get_tracking_id_text(track_id),
#                               get_color_from_track_id(track_id),
#                               np.zeros_like(universal_mask),
#                               do_draw_text="box_labels" in channels)

#     if hand_clearance_box is not None:
#         draw.rectangle(hand_clearance_box, outline=(128, 0, 0), width=2)

#     base_image_width = base_image.width
#     base_image_height = base_image.height

#     max_image_width = int(request.GET.get("max_width", LARGE))
#     max_image_height = int(request.GET.get("max_height", LARGE))

#     if max_image_width > base_image_width or max_image_height > base_image_height:
#         base_image.thumbnail((max_image_width, max_image_height))

#     response = HttpResponse(content_type="image/jpeg")

#     base_image.convert("RGB").save(response, "JPEG")
#     return response


# def get_video_frame_data(request, video_id, frame_idx):
#     channels = [strp for s in request.GET["channels"].split(",") if len(strp := s.strip()) > 0 ] if "channels" in request.GET else []
#     frame_id = fmt_frame(video_id, frame_idx)
#     frame_data = {}

#     print(f"{channels=}")

#     if "focus" in channels:
#         with open(FOCUS_DATA_PATH, "rb") as f:
#             focus_dict = pickle.load(f)
#         if frame_id in focus_dict:
#             frame_data["focus"] = focus_dict[frame_id]

#     if "hand_bbox" in channels:
#         hand_bbox_data_path = join(HAND_BBOX_DATA_DIR, video_id, f"{frame_id}.pkl")
#         print(f"{hand_bbox_data_path=}")
#         if isfile(hand_bbox_data_path):
#             with open(hand_bbox_data_path, "rb") as f:
#                 hand_bbox_data = pickle.load(f)[2][0]
#                 frame_data["hand_bbox"] = {k: v.tolist() if isinstance(v, np.ndarray) else v
#                                            for k, v in hand_bbox_data.items()}
    
#     if "gt_activity" in channels:
#         val_csv = get_dataset_csv("val")
#         row = val_csv[(val_csv["video_id"] == video_id) & (val_csv["start_frame"] <= frame_idx) & (frame_idx < val_csv["stop_frame"])]
#         if len(row) > 0:
#             noun = row.iloc[0]["noun"]
#             verb = row.iloc[0]["verb"]
#             frame_data["gt_activity_noun"] = noun
#             frame_data["gt_activity_verb"] = verb
#             frame_data["gt_activity"] = f"{verb} {noun}"

#     return JsonResponse(frame_data)
