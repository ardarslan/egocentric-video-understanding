from os.path import join
import platform


LARGE = 1e9
CURRENT_SERVER = platform.node().split(".")[0]
DATA_ROOT_PATH = {
    "ait-server-03": "/data/data3/agavryushin/Datasets/",
    "ait-server-04": "/data/agavryushin/Datasets/",
    "ait-server-05": "/data/agavryushin/Datasets/",
}.get(CURRENT_SERVER, "/pub/scratch/agavryushin/Datasets/")
# DATA_ROOT_PATH = "/pub/scratch/agavryushin/Datasets/"
CHECKPOINTS_PATH = "/pub/scratch/agavryushin/Thesis/checkpoints"
DEFAULT_DEVICE = "cuda:6"

DEFAULT_FOCUS_THRESHOLD = 80.0

ROOT_PATH = "/mnt/scratch/agavryushin/Thesis/"
FRANKMOCAP_PATH = "/mnt/scratch/agavryushin/frankmocap/"
UNIDET_PATH = "/mnt/scratch/agavryushin/UniDet/"
SHIFTNET_PATH = "/mnt/scratch/agavryushin/Shift-Net/"
TRACK_ANYTHING_PATH = "/mnt/scratch/agavryushin/Track-Anything/"
VISOR_PATH = "/mnt/scratch/agavryushin/VISOR-HOS/"
EGOHOS_PATH = "/mnt/scratch/agavryushin/EgoHOS/mmsegmentation/"
E2FGVI_PATH = "/mnt/scratch/agavryushin/E2FGVI/"
CVD2_PATH = "/mnt/scratch/agavryushin/cvd2/"
VIDEO_READER_CHUNK_SIZE = 10
VIDEO_READER_MAX_CHUNK_COUNT = 1
EK_ASSUMED_FPS = 60
DETAILED_AVAILABILITY_DATA_CHUNK_SIZE = 600
DEFAULT_HOS_THRESHOLD = 0.5
DEFAULT_HOS_VERSION = f"threshold={DEFAULT_HOS_THRESHOLD}"
DEFAULT_HAND_MESH_VERSION = "frankmocap"
DEFAULT_HAND_MESH_VERSION_NAME = "FrankMocap"
DEFAULT_TRACKING_MASK_MIN_LENGTH = 30
DEFAULT_OBJECT_BBOX_VERSION = "unidet_image"
DEFAULT_TRACKING_MASK_MERGING_OVERLAP_FRAME_COUNT_FRACTION = 0.75
DEFAULT_TRACKING_MASK_MERGING_OVERLAP_FRAME_IOU_FRACTION = 0.9
DEFAULT_TRACKING_MASK_MERGING_OVERLAP_FRAME_IOA_FRACTION = 0.9
DEFAULT_TRACKING_MASK_MAX_TORTUOSITY = 0.35
DEFAULT_TRACKING_MASK_MAX_CD_Q90 = 0.015
DEFAULT_TRACKING_MASK_MAX_HAND_IOA = 0.10

EXCLUDED_CHANNELS = ["focus"]
HAND_MESH_DATA_DIR = join(ROOT_PATH, "data", "EK_hand_mesh")
HOS_DATA_DIR = join(ROOT_PATH, "data", "EK_HOS")
HAND_BBOX_DATA_DIR = join(ROOT_PATH, "data", "EK_hand_bbox")
SEGMENTATION_MASK_DATA_DIR = join(ROOT_PATH, "data", "EK_sam_mask_outputs")
TRACKING_BBOX_DATA_DIR = join(ROOT_PATH, "data", "EK_bbox_tracks")
TRACKING_MASK_DATA_DIR = join(ROOT_PATH, "data", "EK_mask_tracks")
TRACKING_MASK_POSTPROCESSING_DATA_DIR = join(
    ROOT_PATH, "data", "EK_mask_tracks_postprocessing"
)
SEGMENTATION_MASK_PALETTE = [
    "#D65E26",
    "#E4A120",
]
OBJECT_BBOX_DATA_DIR = join(ROOT_PATH, "data", "EK_object_bbox")
EK_AVAILABILITY_CACHE_DIR = join(ROOT_PATH, "data", "EK_availability_cache")
EK_DETAILED_AVAILABILITY_CACHE_DIR = join(
    ROOT_PATH, "data", "EK_detailed_availability_cache"
)
INPAINTING_DATA_DIR = join(ROOT_PATH, "data", "EK_inpainting")
FOCUS_DATA_PATH = join(ROOT_PATH, "data", "ek_variance_of_laplacian_1686167060.pkl")
COMMENT_DATA_DIR = join(ROOT_PATH, "data", "EK_comments")
ACTIONWISE_VIDEO_DATA_DIR = join(ROOT_PATH, "data", "EK_actionwise_videos")
DEPTH_ESTIMATION_DATA_DIR = join(ROOT_PATH, "data", "EK_depth")
DEPTH_ESTIMATION_POSTPROCESSING_DATA_DIR = join(
    ROOT_PATH, "data", "EK_depth_postprocessing"
)
POINT_DATA_DIR = join(DATA_ROOT_PATH, "EPIC-Fields")
FRAME_FEATURE_EXTRACTION_STRIDE = 6

CHANNEL_VIDEO_PATH_FUNCTS = {
    "image": None,
    "gt_activity": None,
    "inpainted": lambda video_id, version: join(INPAINTING_DATA_DIR, version, video_id),
    "depth_postprocessing": lambda video_id, version: join(
        DEPTH_ESTIMATION_POSTPROCESSING_DATA_DIR, version, video_id
    ),
    "hand_mesh": lambda video_id, version: join(HAND_MESH_DATA_DIR, version),
    "hand_mesh_vis": lambda video_id, version: join(
        HAND_MESH_DATA_DIR, version, "rendered"
    ),
    "hand_mesh_vis_nobg": lambda video_id, version: join(
        HAND_MESH_DATA_DIR, version, "rendered_nobg"
    ),
    "hand_bbox": lambda video_id: join(HAND_BBOX_DATA_DIR, video_id),
    "hos": lambda video_id, version: (
        join(HOS_DATA_DIR, version, "object", video_id, "pkls")
        if version == ""
        else join(HOS_DATA_DIR, version, video_id, "pkls")
    ),
    "hos_hands": lambda video_id, version: (
        join(HOS_DATA_DIR, version, "hands", video_id, "pkls")
        if version in ["egohos", "ek-gt", "ek-gt-sparse", "ek-gt-dense"]
        else join(HOS_DATA_DIR, version, video_id, "pkls")
    ),
    "hos_object": lambda video_id, version: (
        join(HOS_DATA_DIR, version, "object", video_id, "pkls")
        if version in ["egohos", "ek-gt", "ek-gt-sparse", "ek-gt-dense"]
        else join(HOS_DATA_DIR, version, video_id, "pkls")
    ),
    "segmentation_mask": lambda video_id, version: join(
        SEGMENTATION_MASK_DATA_DIR, version.replace("inpainted_", ""), video_id
    ),
    "object_bbox": lambda video_id, version=DEFAULT_OBJECT_BBOX_VERSION: join(
        OBJECT_BBOX_DATA_DIR, version, video_id
    ),
    "focus": None,
    "tracking_bbox": lambda video_id: join(TRACKING_BBOX_DATA_DIR, video_id),
    "tracking_mask": lambda video_id, image_version, hos_version, min_length, track_type: join(
        TRACKING_MASK_DATA_DIR,
        image_version.replace("inpainted_", ""),
        hos_version,
        f"min-length={min_length}",
        video_id,
        track_type,
    ),
    "tracking_mask_postprocessing": lambda video_id, image_version, hos_version, min_length, track_type: join(
        TRACKING_MASK_POSTPROCESSING_DATA_DIR,
        image_version.replace("inpainted_", ""),
        hos_version,
        f"min-length={min_length}",
        video_id,
        track_type,
    ),
    "comment": (lambda video_id: join(COMMENT_DATA_DIR, video_id)),
    "actionwise_video": lambda video_id: join(ACTIONWISE_VIDEO_DATA_DIR, video_id),
}

CHANNEL_FRAME_PATH_FUNCTS = {
    "image": None,
    "gt_activity": None,
    "inpainted": lambda video_id, frame_idx, frame_id, version: join(
        INPAINTING_DATA_DIR, version, video_id, f"{frame_id}.jpg"
    ),
    "depth_postprocessing": lambda video_id, frame_idx, frame_id, version: join(
        DEPTH_ESTIMATION_POSTPROCESSING_DATA_DIR, version, video_id, f"{frame_id}.png"
    ),
    "hand_mesh": lambda video_id, frame_idx, frame_id, version: join(
        HAND_MESH_DATA_DIR, version, "mocap", f"{frame_id}_prediction_result.pkl.zip"
    ),
    "hand_mesh_vis": lambda video_id, frame_idx, frame_id, version: join(
        HAND_MESH_DATA_DIR, version, "rendered", f"{frame_id}.jpg"
    ),
    "hand_mesh_vis_nobg": lambda video_id, frame_idx, frame_id, version: join(
        HAND_MESH_DATA_DIR, version, "rendered_nobg", f"{frame_id}.png"
    ),  # png here!
    "hand_bbox": lambda video_id, frame_idx, frame_id: join(
        HAND_BBOX_DATA_DIR, video_id, f"{frame_id}.pkl"
    ),
    "hos": lambda video_id, frame_idx, frame_id, version: (
        join(HOS_DATA_DIR, version, "object", video_id, "pkls", f"{frame_id}.pkl.zip")
        if version in ["egohos", "ek-gt", "ek-gt-sparse", "ek-gt-dense"]
        else join(HOS_DATA_DIR, version, video_id, "pkls", f"{frame_id}.pkl.zip")
    ),
    "hos_hands": (
        lambda video_id, frame_idx, frame_id, version: (
            join(
                HOS_DATA_DIR, version, "hands", video_id, "pkls", f"{frame_id}.pkl.zip"
            )
            if version in ["egohos", "ek-gt", "ek-gt-sparse", "ek-gt-dense"]
            else join(HOS_DATA_DIR, version, video_id, "pkls", f"{frame_id}.pkl.zip")
        )
    ),
    "hos_object": (
        lambda video_id, frame_idx, frame_id, version: (
            join(
                HOS_DATA_DIR, version, "object", video_id, "pkls", f"{frame_id}.pkl.zip"
            )
            if version in ["egohos", "ek-gt", "ek-gt-sparse", "ek-gt-dense"]
            else join(HOS_DATA_DIR, version, video_id, "pkls", f"{frame_id}.pkl.zip")
        )
    ),
    "segmentation_mask": lambda video_id, frame_idx, frame_id, version: join(
        SEGMENTATION_MASK_DATA_DIR,
        version.replace("inpainted_", "") if version != "image" else "full_image",
        video_id,
        f"{frame_id}.pkl.zip",
    ),
    "object_bbox": lambda video_id, frame_idx, frame_id, version=DEFAULT_OBJECT_BBOX_VERSION: join(
        OBJECT_BBOX_DATA_DIR, version, video_id, f"{frame_id}.pkl"
    ),
    "focus": None,
    "tracking_bbox": (
        lambda video_id, frame_idx, frame_id: (
            join(TRACKING_BBOX_DATA_DIR, f"{video_id}.pkl")
            if frame_idx < 0
            else join(TRACKING_BBOX_DATA_DIR, video_id, f"{frame_id}.pkl")
        )
    ),
    "tracking_mask": (
        lambda video_id, frame_idx, frame_id, image_version, hos_version, min_length, track_type: join(
            TRACKING_MASK_DATA_DIR,
            image_version.replace("inpainted_", ""),
            hos_version,
            f"min-length={min_length}",
            video_id,
            track_type,
        )
    ),
    "tracking_mask_postprocessing": lambda video_id, frame_idx, frame_id, image_version, hos_version, min_length, track_type: join(
        TRACKING_MASK_POSTPROCESSING_DATA_DIR,
        image_version.replace("inpainted_", ""),
        hos_version,
        f"min-length={min_length}",
        video_id,
        track_type,
    ),
    "comment": (
        lambda video_id, frame_idx, frame_id: join(COMMENT_DATA_DIR, video_id, frame_id)
    ),
    "actionwise_video": lambda video_id, frame_idx, frame_id: join(
        ACTIONWISE_VIDEO_DATA_DIR, video_id
    ),
}

CHANNEL_NAME_DICT = {
    "image": "Image",
    "gt_activity": "GT activity",
    "inpainted": "Inpainted image",
    "hand_mesh": "Hand mesh",
    "hand_mesh_vis": "Hand mesh",
    "hand_mesh_vis_nobg": "Hand mesh",
    "hand_bbox": "Hand bounding box",
    "hos": "HOS",
    "hos_hands": "HOS",
    "hos_object": "HOS",
    "segmentation_mask": "Segmentation mask (HOS bbox)",
    "object_bbox": "Object bounding box",
    "focus": "Focus",
    "tracking_bbox": "Tracking bounding box",
    "tracking_mask": "Tracking mask",
}
CHANNEL_NAME_DICT_STR = str(CHANNEL_NAME_DICT)
PALETTE = ["#0673B0", "#069E73", "#5BB4E4", "#5AB4E2", "#E5A023", "#F1E444", "#D16124"]
PALETTE_STR = ", ".join([f'"{color}"' for color in PALETTE])
HAND_JOINT_INDICES = {"THUMB": 4, "INDEX": 8, "MIDDLE": 12, "RING": 16, "LITTLE": 20}
HAND_JOINT_COLORS = {
    "THUMB": "#D65E26",
    "INDEX": "#E4A120",
    "MIDDLE": "#069E73",
    "RING": "#5BB4E4",
    "LITTLE": "#0673B0",
}
TRACKING_ID_SUBSTRING_LENGTH = 4
UNIDET_IGNORE_CATEGORIES = [
    "foot",
    "human foot",
    "footwear",
    "sandal",
    "sandals",
    "sock",
    "boot",
    "boots",
    "leather shoes",
    "ski_boot",
    "leg",
    "human leg",
    "slipper",
    "slippers",
    "man",
    "woman",
    "foot_superclass",
    "ignore_superclass",
]
