import os
from os.path import dirname, isdir, isfile, join
import pickle
import time

from data_handling.specific.ek100 import *
import utils.globals
from utils.globals import *
from utils.io import read_pkl


def get_all_channels():
    available_hos_versions = get_available_hos_versions()
    available_image_versions = get_available_image_versions()
    available_min_lengths = get_available_tracking_mask_min_lengths()
    available_segmentation_mask_versions = get_available_segmentation_mask_versions()
    available_hand_mesh_versions = get_available_hand_mesh_versions()
    available_object_bbox_versions = get_available_object_bbox_versions()
    return set(
        [
            "hand_bbox",
            "tracking_bbox",
            "tracking_bbox_boundaries",
            "focus",
            *[f"object_bbox_{v}" for v in available_object_bbox_versions],
            *[f"hand_mesh_{v}" for v in available_hand_mesh_versions],
            *[f"segmentation_mask_{v}" for v in available_segmentation_mask_versions],
            *[
                f"tracking_mask_track=left-hand_image-version={i.replace('_', '-')}_min-length={l}_{v}"
                for v in available_hos_versions
                for i in available_image_versions
                for l in available_min_lengths
            ],
            *[
                f"tracking_mask_track=right-hand_image-version={i.replace('_', '-')}_min-length={l}_{v}"
                for v in available_hos_versions
                for i in available_image_versions
                for l in available_min_lengths
            ],
            *[
                f"tracking_mask_track=object_image-version={i.replace('_', '-')}_min-length={l}_{v}"
                for v in available_hos_versions
                for i in available_image_versions
                for l in available_min_lengths
            ],
            *[f"hos_{v}" for v in available_hos_versions],
            *[f"hos_hands_{v}" for v in available_hos_versions],
            *[f"hos_object_{v}" for v in available_hos_versions],
            *[
                v if v == "image" or v.startswith("depth") else f"inpainted_{v}"
                for v in available_image_versions
            ],
        ]
    )


def get_available_hos_versions():
    available_versions = []
    for dirname in os.listdir(HOS_DATA_DIR):
        if isdir(join(HOS_DATA_DIR, dirname)):
            if dirname.startswith(f"threshold="):
                available_versions.append(
                    dirname
                )  # float(dirname.replace("_", "=").split("=")[1])
            elif dirname in ["egohos", "ek-gt", "ek-gt-sparse", "ek-gt-dense"]:
                available_versions.append(dirname)
    return available_versions


def get_available_object_bbox_versions():
    available_versions = []
    for dirname in os.listdir(OBJECT_BBOX_DATA_DIR):
        if isdir(join(OBJECT_BBOX_DATA_DIR, dirname)):
            available_versions.append(dirname)
    return available_versions


def get_available_image_versions():
    available_versions = ["image"]
    for dirname in os.listdir(INPAINTING_DATA_DIR):
        if isdir(join(INPAINTING_DATA_DIR, dirname)):
            available_versions.append(dirname)
    if isdir(DEPTH_ESTIMATION_DATA_DIR):
        available_versions.append("depth_midas2_section_scale")
        available_versions.append("depth_midas2_video_scale")
        available_versions.append("depth_cvd_section_scale")
        available_versions.append("depth_cvd_video_scale")
    return available_versions


def get_available_tracking_bbox_versions():
    available_versions = []
    for root, dirs, files in os.walk(TRACKING_BBOX_DATA_DIR):
        for dirname in dirs:
            if dirname.startswith(f"threshold="):
                available_versions.append(
                    dirname
                )  # float(dirname.replace("_", "=").split("=")[1])
            elif dirname == "egohos":
                available_versions.append(dirname)
    return available_versions


def get_available_segmentation_mask_versions():
    available_versions = []
    for dirname in os.listdir(SEGMENTATION_MASK_DATA_DIR):
        available_versions.append(dirname)
        # if dirname.startswith(f"threshold=") or dirname in ["egohos"]:
        #     available_versions.append(dirname)  # float(dirname.replace("_", "=").split("=")[1])
        # elif dirname == "egohos":
        #     available_versions.append(dirname)
        # elif dirname == "full_image":
        #     available_versions.append(dirname)
    return available_versions


def get_available_hand_mesh_versions():
    available_versions = []
    for dirname in os.listdir(HAND_MESH_DATA_DIR):
        available_versions.append(dirname)
    return available_versions


def get_available_tracking_mask_min_lengths():
    available_lengths = set([])
    for root, dirs, files in os.walk(TRACKING_MASK_DATA_DIR):
        for dirname in dirs:
            if dirname.startswith("min-length="):
                available_lengths.add(int(dirname.split("=", 1)[1]))
    return list(available_lengths)


def get_file_count_and_last_mod_timestamp(
    channel, video_id, start_frame=None, end_frame=None
):
    if channel == "image":
        return -1, -1
    if channel == "tracking_bbox_boundaries":
        channel = "tracking_bbox"

    file_count = 0
    max_time = 0
    if channel.startswith("inpaint") or channel.startswith("depth"):
        path_funct_channel = (
            "inpainted" if channel.startswith("inpaint") else "depth_postprocessing"
        )
        spl = channel.split("_")
        version = "_".join(spl[1:])
        dir_path = CHANNEL_VIDEO_PATH_FUNCTS[path_funct_channel](video_id, version)
    elif (
        channel.startswith("hos") or "tracking_mask" in channel
    ) and not channel.startswith("segmentation_mask_"):
        spl = channel.split("_")
        image_version = next(
            (
                s.split("=", 1)[1].replace("-", "_")
                for s in spl
                if s.startswith("image-version=")
            ),
            "image",
        )
        hos_version = next(
            (
                s
                for s in spl
                if s.startswith("threshold=")
                or s in ["egohos", "ek", "gt", "ek-gt", "ek-gt-sparse", "ek-gt-dense"]
            ),
            None,
        )  # float(s.split("=")[-1]) for ...
        track_type = next(
            (
                s.split("=", 1)[1].replace("-", "_")
                for s in spl
                if s.startswith("track") and "=" in s
            ),
            None,
        )
        channel_name = "_".join(
            [
                s
                for s in spl
                if "=" not in s
                and s
                not in ["egohos", "ek", "gt", "ek-gt", "ek-gt-sparse", "ek-gt-dense"]
            ]
        )  # must split ek_gt
        min_length = next(
            (float(s.split("=", 1)[1]) for s in spl if s.startswith("min-length=")),
            None,
        )

        if not channel.startswith("tracking_mask"):
            dir_path = CHANNEL_VIDEO_PATH_FUNCTS[channel_name](video_id, hos_version)
        else:
            if track_type is not None:
                dir_path = CHANNEL_VIDEO_PATH_FUNCTS[channel_name](
                    video_id, image_version, hos_version, min_length, track_type
                )
            else:
                dir_path = CHANNEL_VIDEO_PATH_FUNCTS[channel_name](
                    video_id, image_version, hos_version, min_length
                )
    elif channel.startswith("segmentation_mask"):
        spl = channel.split("_", 2)
        version = spl[-1] if len(spl) >= 3 else DEFAULT_HOS_VERSION
        dir_path = CHANNEL_VIDEO_PATH_FUNCTS["segmentation_mask"](video_id, version)
    elif channel.startswith("hand_mesh"):
        spl = channel.split("_", 2)
        version = spl[-1] if len(spl) >= 3 else DEFAULT_HAND_MESH_VERSION
        dir_path = CHANNEL_VIDEO_PATH_FUNCTS["hand_mesh"](video_id, version)
    elif channel.startswith("object_bbox"):
        spl = channel.split("_", 2)
        version = spl[-1] if len(spl) >= 3 else DEFAULT_OBJECT_BBOX_VERSION
        dir_path = CHANNEL_VIDEO_PATH_FUNCTS["object_bbox"](video_id, version)
    else:
        dir_path = CHANNEL_VIDEO_PATH_FUNCTS[channel](video_id)

    # disabled; affects performance too much:
    if False:
        for root, dirs, files in os.walk(dir_path):
            for fn in files:
                frame_idx = next(
                    (
                        int(ispl)
                        for i in fn.split("_")[::-1]
                        if (ispl := i.split(".")[0]).isnumeric()
                    ),
                    None,
                )
                if (
                    not any((e is None for e in [frame_idx, start_frame, end_frame]))
                    and (start_frame or -1) > frame_idx
                    or (end_frame or LARGE) < frame_idx
                ):
                    continue

                file_path = join(root, fn)

                max_time = max(max_time, os.path.getmtime(file_path))

                file_count += 1

    return file_count, max_time


def get_availability_data(video_id, start_frame=0, end_frame=LARGE, request=None):
    def get_channel_cache_path(channel):
        return join(
            EK_DETAILED_AVAILABILITY_CACHE_DIR,
            video_id,
            f"{channel}__start={start_frame}__end={end_frame}.pkl",
        )

    with open(FOCUS_DATA_PATH, "rb") as f:
        focus_dict = pickle.load(f)

    tracking_bbox_channels = (
        [
            strp
            for s in request.GET["tracking_channels"].split(",")
            if len(strp := s.strip()) > 0
        ]
        if request is not None and "tracking_channels" in request.GET
        else []
    )
    available_hos_versions = get_available_hos_versions()
    available_image_versions = get_available_image_versions()
    available_tracking_mask_min_lengths = get_available_tracking_mask_min_lengths()
    available_segmentation_mask_versions = get_available_segmentation_mask_versions()
    available_object_bbox_versions = get_available_object_bbox_versions()
    available_hand_mesh_versions = get_available_hand_mesh_versions()
    print(f"{available_tracking_mask_min_lengths=}")
    print(f"{available_hos_versions=}")
    print(f"{available_image_versions=}")
    print(f"{available_segmentation_mask_versions=}")
    print(f"{available_object_bbox_versions=}")
    print(f"{available_hand_mesh_versions=}")

    handled_channels = set()

    all_channels = get_all_channels()
    all_channels_filtered = set([s for s in all_channels if s not in EXCLUDED_CHANNELS])

    av = {c: [] for c in all_channels}

    channel_fc_lm = {}
    for channel in all_channels:
        if channel in EXCLUDED_CHANNELS:
            continue

        channel_cache_path = get_channel_cache_path(channel)
        file_count, last_mod_timestamp = get_file_count_and_last_mod_timestamp(
            channel, video_id, start_frame, end_frame
        )
        channel_fc_lm[channel] = (file_count, last_mod_timestamp)
        if isfile(channel_cache_path):
            channel_data = read_pkl(channel_cache_path)
            if (
                "file_count" in channel_data
                and "last_mod_timestamp" in channel_data
                and channel_data["file_count"] == file_count
                and channel_data["last_mod_timestamp"] == last_mod_timestamp
            ):
                av[channel] = channel_data["data"]
                handled_channels.add(channel)

    # !!!!!!!!!!
    if True or len(all_channels_filtered.difference(handled_channels)) > 0:
        reader = VideoReader(
            get_video_path(video_id),
            get_extracted_frame_dir_path(video_id),
            assumed_fps=EK_ASSUMED_FPS,
        )

        if "image" not in handled_channels:
            av["image"].append([0, reader.get_virtual_frame_count()])

        if "tracking_bbox_boundaries" not in handled_channels:
            av["tracking_bbox_boundaries"].append(
                [0, reader.get_virtual_frame_count(), {}]
            )

        for frame_idx in range(
            start_frame, min(end_frame + 1, reader.get_virtual_frame_count())
        ):
            frame_id = fmt_frame(video_id, frame_idx)

            for version in available_hand_mesh_versions:
                channel = f"hand_mesh_{version}"
                if channel not in handled_channels:
                    hand_mesh_path = CHANNEL_FRAME_PATH_FUNCTS["hand_mesh"](
                        video_id, frame_idx, frame_id, version
                    )
                    hand_mesh_vis_path = CHANNEL_FRAME_PATH_FUNCTS["hand_mesh_vis"](
                        video_id, frame_idx, frame_id, version
                    )
                    available = ""
                    if isfile(hand_mesh_path):
                        data = read_pkl(hand_mesh_path)
                        if "hand_bbox_list" in data:
                            for hand_data in data["hand_bbox_list"]:
                                for hand in ["left_hand", "right_hand"]:
                                    if (
                                        hand in hand_data
                                        and type(hand_data[hand]) is np.ndarray
                                    ):
                                        available += hand[0].upper()

                    if isfile(hand_mesh_vis_path):
                        available += "I"

                    if (
                        len(av[channel]) == 0
                        or av[channel][-1][1] != frame_idx
                        or av[channel][-1][2] != available
                    ):
                        av[channel].append([frame_idx, frame_idx + 1, available])
                    else:
                        av[channel][-1][1] = frame_idx + 1

            if "hand_bbox" not in handled_channels:
                hand_bbox_path = CHANNEL_FRAME_PATH_FUNCTS["hand_bbox"](
                    video_id, frame_idx, frame_id
                )
                if isfile(hand_bbox_path):
                    available = ""

                    with open(hand_bbox_path, "rb") as f:
                        pkl_data = pickle.load(f)
                        hand_data = pkl_data[2]
                        if isinstance(hand_data, list):
                            hand_data = hand_data[0]

                        if isinstance(hand_data, dict):
                            for hand in ["left_hand", "right_hand"]:
                                if hand in hand_data and hand_data[hand] is not None:
                                    available += hand[0].upper()

                    if (
                        len(av["hand_bbox"]) == 0
                        or av["hand_bbox"][-1][1] != frame_idx
                        or av["hand_bbox"][-1][2] != available
                    ):
                        av["hand_bbox"].append([frame_idx, frame_idx + 1, available])
                    else:
                        av["hand_bbox"][-1][1] = frame_idx + 1

            for version in available_hos_versions:
                for super_channel in ["hos", "hos_hands", "hos_object"]:
                    channel = f"{super_channel}_{version}"
                    if channel not in handled_channels:
                        available = ""
                        if channel not in av:
                            av[channel] = []

                        hos_path = CHANNEL_FRAME_PATH_FUNCTS[super_channel](
                            video_id, frame_idx, frame_id, version
                        )
                        if isfile(hos_path):
                            hos_data = read_pkl(hos_path)
                            if version == "egohos":
                                # see https://github.com/owenzlz/EgoHOS/tree/main#datasets
                                sum_left = (hos_data == 1).sum()
                                sum_right = (hos_data == 2).sum()
                                sum_obj = (hos_data == 3).sum()
                                if sum_left > 0:
                                    available += "L"
                                if sum_right > 0:
                                    available += "R"
                                if sum_obj > 0:
                                    available += "O"
                            else:
                                if "instances" not in hos_data or not hasattr(
                                    hos_data["instances"], pred_handsides
                                ):
                                    continue

                                for cls, handside, mask, box in zip(
                                    hos_data["instances"].pred_classes,
                                    hos_data["instances"].pred_handsides,
                                    hos_data["instances"].pred_masks,
                                    hos_data["instances"].pred_boxes,
                                ):
                                    if cls == 0:  # 0: hand
                                        # 0: left; 1: right
                                        handside = handside.argmax().item()
                                        available += "LR"[handside]
                                    else:  # 1: object
                                        available += "O"

                            available = "".join(sorted(available))
                            if (
                                len(av[channel]) == 0
                                or av[channel][-1][1] != frame_idx
                                or av[channel][-1][2] != available
                            ):
                                av[channel].append(
                                    [frame_idx, frame_idx + 1, available]
                                )
                            else:
                                av[channel][-1][1] = frame_idx + 1

            for image_version in available_image_versions:
                for hos_version in available_hos_versions:
                    for min_length in available_tracking_mask_min_lengths:
                        for track_type in ["object", "left_hand", "right_hand"]:
                            channel = f"tracking_mask_track={track_type.replace('_', '-')}_image-version={image_version.replace('_', '-')}_min-length={min_length}_{hos_version}"
                            if channel in handled_channels:
                                continue

                            if channel not in av:
                                av[channel] = []

                            track_list = []
                            tracking_mask_path = CHANNEL_FRAME_PATH_FUNCTS[
                                "tracking_mask"
                            ](
                                video_id,
                                frame_idx,
                                frame_id,
                                image_version,
                                hos_version,
                                min_length,
                                track_type,
                            )

                            if isdir(tracking_mask_path):
                                for dir_name in os.listdir(join(tracking_mask_path)):
                                    dir_path = join(tracking_mask_path, dir_name)
                                    file_path = join(
                                        dir_path, f"{frame_id}__{dir_name}.pkl.zip"
                                    )
                                    if isfile(file_path):
                                        track_list.append(dir_name)

                            track_list.sort()

                            if len(track_list) > 0:
                                if (
                                    len(av[channel]) == 0
                                    or av[channel][-1][1] != frame_idx
                                    or av[channel][-1][-1] != track_list
                                ):
                                    av[channel].append(
                                        [frame_idx, frame_idx + 1, track_list]
                                    )
                                else:
                                    av[channel][-1][1] = frame_idx + 1

            for segmentation_mask_version in available_segmentation_mask_versions:
                channel = f"segmentation_mask_{segmentation_mask_version}"
                if channel not in handled_channels:
                    segmentation_mask_path = CHANNEL_FRAME_PATH_FUNCTS[
                        "segmentation_mask"
                    ](video_id, frame_idx, frame_id, segmentation_mask_version)
                    if isfile(segmentation_mask_path):
                        available = ""
                        pkl_data = read_pkl(segmentation_mask_path)
                        for box_idx, box_data in enumerate(pkl_data):
                            # if keep_box_idx != -1:
                            # box_data = pkl[keep_box_idx]
                            cls = box_data["cls"]
                            if cls == -1:
                                available += "I"
                            else:
                                available += "B"
                                if cls == 1:
                                    available += "O"
                                    outer_box = box_data["box"]
                                    masks = box_data["masks"]

                        if (
                            len(av[channel]) == 0
                            or av[channel][-1][1] != frame_idx
                            or av[channel][-1][2] != available
                        ):
                            av[channel].append([frame_idx, frame_idx + 1, available])
                        else:
                            av[channel][-1][1] = frame_idx + 1

                for object_bbox_version in available_object_bbox_versions:
                    channel = f"object_bbox_{object_bbox_version}"
                    if channel not in handled_channels:
                        object_bbox_path = CHANNEL_FRAME_PATH_FUNCTS["object_bbox"](
                            video_id, frame_idx, frame_id, object_bbox_version
                        )
                        if isfile(object_bbox_path):
                            available = False
                            with open(object_bbox_path, "rb") as f:
                                l = pickle.load(f)
                                if "scores" in l and len(l) > 0:
                                    available = True

                            if (
                                len(av[channel]) == 0
                                or av[channel][-1][1] != frame_idx
                                or av[channel][-1][2] != available
                            ):
                                av[channel].append(
                                    [frame_idx, frame_idx + 1, available]
                                )
                            else:
                                av[channel][-1][1] = frame_idx + 1

            if (
                "tracking_bbox" not in handled_channels
                or "tracking_bbox_boundaries" not in handled_channels
            ):
                tracking_bbox_path = CHANNEL_FRAME_PATH_FUNCTS["tracking_bbox"](
                    video_id, frame_idx, frame_id
                )
                if isfile(tracking_bbox_path):
                    tracking_bbox_data = read_pkl(tracking_bbox_path)

                    available = {
                        k: list(v.keys())
                        for k, v in tracking_bbox_data["tracks"].items()
                    }

                    if "tracking_bbox" not in handled_channels:
                        if (
                            len(av["tracking_bbox"]) == 0
                            or av["tracking_bbox"][-1][1] != frame_idx
                            or av["tracking_bbox"][-1][2] != available
                        ):
                            av["tracking_bbox"].append(
                                [frame_idx, frame_idx + 1, available]
                            )
                        else:
                            av["tracking_bbox"][-1][1] = frame_idx + 1

                if "tracking_bbox_boundaries" not in handled_channels:
                    for k, v in tracking_bbox_data["tracks"].items():
                        for bbox_id in list(v.keys()):
                            if bbox_id not in av["tracking_bbox_boundaries"][0][2]:
                                av["tracking_bbox_boundaries"][0][2][bbox_id] = [
                                    frame_idx,
                                    frame_idx + 1,
                                ]
                            else:
                                av["tracking_bbox_boundaries"][0][2][bbox_id][-1] = (
                                    frame_idx + 1
                                )

            if "focus" not in handled_channels:
                if frame_id in focus_dict:
                    if len(av["focus"]) == 0 or av["focus"][-1][-1] != frame_idx:
                        av["focus"].append([frame_idx, frame_idx + 1])
                    else:
                        av["focus"][-1][-1] = frame_idx + 1

    for channel in all_channels:
        if channel in handled_channels or channel == "focus":
            continue

        file_count, last_mod_timestamp = channel_fc_lm[channel]
        channel_cache_path = get_channel_cache_path(channel)
        os.makedirs(dirname(channel_cache_path), exist_ok=True)
        with open(channel_cache_path, "wb") as f:
            pickle.dump(
                {
                    "file_count": file_count,
                    "last_mod_timestamp": last_mod_timestamp,
                    "data": av[channel],
                },
                f,
            )

    return av
