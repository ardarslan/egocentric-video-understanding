# %%

import argparse
from collections import deque
import numpy as np
import os
from os.path import basename, dirname, isfile, join
import pickle
from PIL import Image, ImageDraw, ImageFont
import pytorch3d
import sys
import time
from scipy.interpolate import UnivariateSpline
import ssl
import sys
import torch
from tqdm import tqdm
import zipfile

ssl._create_default_https_context = ssl._create_unverified_context

sys.path.append(dirname(dirname(__file__)))

from data_handling.specific.ek100 import *
from utils.args import arg_dict_to_list
from utils.globals import *
from utils.io import read_pkl

os.chdir(FRANKMOCAP_PATH)

from demo.demo_options import DemoOptions
from mocap_utils.coordconv import convert_smpl_to_bbox, convert_bbox_to_oriIm
from handmocap.hand_modules.h3dw_model import extract_hand_output
import mocap_utils.general_utils as gnu
import mocap_utils.demo_utils as demo_utils

from handmocap.hand_mocap_api import HandMocap
from handmocap.hand_bbox_detector import HandBboxDetector

import renderer.image_utils as imu
from renderer.viewer2D import ImShow


# from https://stackoverflow.com/a/52998713
def ewma_vectorized(data, alpha, offset=None, dtype=None, order="C", out=None):
    """
    Calculates the exponential moving average over a vector.
    Will fail for large inputs.
    :param data: Input data
    :param alpha: scalar float in range (0,1)
        The alpha parameter for the moving average.
    :param offset: optional
        The offset for the moving average, scalar. Defaults to data[0].
    :param dtype: optional
        Data type used for calculations. Defaults to float64 unless
        data.dtype is float32, then it will use float32.
    :param order: {'C', 'F', 'A'}, optional
        Order to use when flattening the data. Defaults to 'C'.
    :param out: ndarray, or None, optional
        A location into which the result is stored. If provided, it must have
        the same shape as the input. If not provided or `None`,
        a freshly-allocated array is returned.
    """
    data = np.array(data, copy=False)

    if dtype is None:
        if data.dtype == np.float32:
            dtype = np.float32
        else:
            dtype = np.float64
    else:
        dtype = np.dtype(dtype)

    if data.ndim > 1:
        # flatten input
        data = data.reshape(-1, order)

    if out is None:
        out = np.empty_like(data, dtype=dtype)
    else:
        assert out.shape == data.shape
        assert out.dtype == dtype

    if data.size < 1:
        # empty input, return empty array
        return out

    if offset is None:
        offset = data[0]

    alpha = np.array(alpha, copy=False).astype(dtype, copy=False)

    # scaling_factors -> 0 as len(data) gets large
    # this leads to divide-by-zeros below
    scaling_factors = np.power(
        1.0 - alpha, np.arange(data.size + 1, dtype=dtype), dtype=dtype
    )
    # create cumulative sum array
    np.multiply(
        data, (alpha * scaling_factors[-2]) / scaling_factors[:-1], dtype=dtype, out=out
    )
    np.cumsum(out, dtype=dtype, out=out)

    # cumsums / scaling
    out /= scaling_factors[-2::-1]

    if offset != 0:
        offset = np.array(offset, copy=False).astype(dtype, copy=False)
        # add offsets
        out += offset * scaling_factors[1:]

    return out


# from https://stackoverflow.com/a/52998713
def ewma_vectorized_2d(
    data, alpha, axis=None, offset=None, dtype=None, order="C", out=None
):
    """
    Calculates the exponential moving average over a given axis.
    :param data: Input data, must be 1D or 2D array.
    :param alpha: scalar float in range (0,1)
        The alpha parameter for the moving average.
        Approximation: alpha = 1 - np.exp(np.log(0.5) / ewm_span_abs)
    :param axis: The axis to apply the moving average on.
        If axis==None, the data is flattened.
    :param offset: optional
        The offset for the moving average. Must be scalar or a
        vector with one element for each row of data. If set to None,
        defaults to the first value of each row.
    :param dtype: optional
        Data type used for calculations. Defaults to float64 unless
        data.dtype is float32, then it will use float32.
    :param order: {'C', 'F', 'A'}, optional
        Order to use when flattening the data. Ignored if axis is not None.
    :param out: ndarray, or None, optional
        A location into which the result is stored. If provided, it must have
        the same shape as the desired output. If not provided or `None`,
        a freshly-allocated array is returned.
    """
    data = np.array(data, copy=False)

    assert data.ndim <= 2

    if dtype is None:
        if data.dtype == np.float32:
            dtype = np.float32
        else:
            dtype = np.float64
    else:
        dtype = np.dtype(dtype)

    if out is None:
        out = np.empty_like(data, dtype=dtype)
    else:
        assert out.shape == data.shape
        assert out.dtype == dtype

    if data.size < 1:
        # empty input, return empty array
        return out

    if axis is None or data.ndim < 2:
        # use 1D version
        if isinstance(offset, np.ndarray):
            offset = offset[0]
        return ewma_vectorized(data, alpha, offset, dtype=dtype, order=order, out=out)

    assert -data.ndim <= axis < data.ndim

    # create reshaped data views
    out_view = out
    if axis < 0:
        axis = data.ndim - int(axis)

    if axis == 0:
        # transpose data views so columns are treated as rows
        data = data.T
        out_view = out_view.T

    if offset is None:
        # use the first element of each row as the offset
        offset = np.copy(data[:, 0])
    elif np.size(offset) == 1:
        offset = np.reshape(offset, (1,))

    alpha = np.array(alpha, copy=False).astype(dtype, copy=False)

    # calculate the moving average
    row_size = data.shape[1]
    row_n = data.shape[0]
    scaling_factors = np.power(
        1.0 - alpha, np.arange(row_size + 1, dtype=dtype), dtype=dtype
    )
    # create a scaled cumulative sum array
    np.multiply(
        data,
        np.multiply(
            alpha * scaling_factors[-2], np.ones((row_n, 1), dtype=dtype), dtype=dtype
        )
        / scaling_factors[np.newaxis, :-1],
        dtype=dtype,
        out=out_view,
    )
    np.cumsum(out_view, axis=1, dtype=dtype, out=out_view)
    out_view /= scaling_factors[np.newaxis, -2::-1]

    if not (np.size(offset) == 1 and offset == 0):
        offset = offset.astype(dtype, copy=False)
        # add the offsets to the scaled cumulative sums
        out_view += offset[:, np.newaxis] * scaling_factors[np.newaxis, 1:]

    return out


def __calc_hand_mesh(hand_type, pose_params, betas, smplx_model, device):
    # adapted from frankmocap/demo/demo_visualize_prediction.py
    hand_rotation = pose_params[:, :3]
    hand_pose = pose_params[:, 3:]
    body_pose = torch.zeros((1, 63), device=device).float()

    assert hand_type in ["left_hand", "right_hand"]
    if hand_type == "right_hand":
        body_pose[:, 60:] = hand_rotation  # set right hand rotation
        right_hand_pose = hand_pose
        left_hand_pose = torch.zeros((1, 45), dtype=torch.float32, device=device)
    else:
        body_pose[:, 57:60] = hand_rotation  # set right hand rotation
        left_hand_pose = hand_pose
        right_hand_pose = torch.zeros((1, 45), dtype=torch.float32, device=device)

    output = smplx_model(
        global_orient=torch.zeros((1, 3), device=device),
        body_pose=body_pose,
        betas=betas,
        left_hand_pose=left_hand_pose,
        right_hand_pose=right_hand_pose,
        return_verts=True,
    )

    hand_info_file = "extra_data/hand_module/SMPLX_HAND_INFO.pkl"
    hand_info = gnu.load_pkl(hand_info_file)
    hand_output = extract_hand_output(
        output,
        hand_type=hand_type.split("_")[0],
        hand_info=hand_info,
        top_finger_joints_type="ave",
        use_cuda=False,
    )

    pred_verts = hand_output["hand_vertices_shift"].detach().cpu().numpy()
    faces = hand_info[f"{hand_type}_faces_local"]
    return pred_verts[0], faces


def run_hand_mocap(args, bbox_detector, hand_mocap, visualizer, generator_videos=None):
    if generator_videos is None:
        generator_videos = get_video_list()
    else:
        generator_videos = [s.strip() for v in generator_videos for s in v.split(",")]

    device = args.device or "cpu"
    ewm_span_abs = round(EK_ASSUMED_FPS * args.ewm_span_rel_to_fps)
    alpha = 1 - np.exp(np.log(0.5) / ewm_span_abs)

    font = ImageFont.truetype(
        "/mnt/scratch/agavryushin/Thesis/webserver/fonts/DejaVuSans.ttf", 25
    )

    for video_id in generator_videos:
        if args.out_dir in [None, ""]:
            out_dir = CHANNEL_VIDEO_PATH_FUNCTS["hand_mesh"](video_id, args.version)
        else:
            out_dir = args.out_dir

        sample_vis_dir = (
            f"/mnt/scratch/agavryushin/Thesis/data/vis_test_frankmocap/{video_id}/"
        )
        os.makedirs(sample_vis_dir, exist_ok=True)

        reader = VideoReader(
            get_video_path(video_id),
            get_extracted_frame_dir_path(video_id),
            assumed_fps=EK_ASSUMED_FPS,
            max_width=args.max_width,
            max_height=args.max_height,
        )
        virtual_frame_count = reader.get_virtual_frame_count()
        range_obj = range(virtual_frame_count)

        if virtual_frame_count == 0:
            continue

        first_frame = reader.get_frame(0)
        img_width = first_frame.shape[1]
        img_height = first_frame.shape[0]

        print()
        print("*** Step 1: reading data ***")
        print()

        betas = {"left_hand": [], "right_hand": []}
        thetas = {"left_hand": [], "right_hand": []}
        cameras = {"left_hand": [], "right_hand": []}
        frame_idxs = {"left_hand": [], "right_hand": []}
        pred_joints_smpl = {"left_hand": {}, "right_hand": {}}  # frame idx as key
        pred_joints_smpl_list = {"left_hand": [], "right_hand": []}  # frame idx as key

        for frame_idx in tqdm(range_obj):
            frame_id = fmt_frame(video_id, frame_idx)
            pkl_path = join(out_dir, "mocap", f"{frame_id}_prediction_result.pkl")

            if not isfile(pkl_path):
                pkl_path = pkl_path + ".zip"

            if not isfile(pkl_path):
                continue

            try:
                result_data = read_pkl(pkl_path)
                if isinstance(result_data["pred_output_list"], list):
                    pred_output_list = result_data["pred_output_list"][0]
                elif isinstance(result_data["pred_output_list"], dict):
                    pred_output_list = result_data["pred_output_list"]
                else:
                    raise NotImplementedError()
            except ToggleableException as ex:
                print(f"Error while processing {frame_id}:", ex)
                continue

            for hand_side_idx, hand_side in enumerate(["left_hand", "right_hand"]):
                if hand_side not in pred_output_list:
                    continue
                hand_side_data = pred_output_list[hand_side]
                if hand_side_data is None or len(hand_side_data) == 0:
                    pred_output_list[hand_side] = None  # needed for __calc_hand_mesh
                    continue

                betas[hand_side].append(hand_side_data["pred_hand_betas"].squeeze())
                thetas[hand_side].append(hand_side_data["pred_hand_pose"].squeeze())
                cameras[hand_side].append(hand_side_data["pred_camera"])
                pred_joints_smpl[hand_side][frame_idx] = hand_side_data[
                    "pred_joints_smpl"
                ]
                pred_joints_smpl_list[hand_side].append(
                    hand_side_data["pred_joints_smpl"]
                )

                # WARNING: some outputs may be None and will not be added to the lists above!
                frame_idxs[hand_side].append(frame_idx)

            # !!!!!!!!!!
            if frame_idx >= 9000:
                break

        window_size = 4  # # of frames to consider on each side, current one excluded
        similarity_threshold = 0.2

        new_joints = {"left_hand": {}, "right_hand": {}}
        votes = {"left_hand": {}, "right_hand": {}}

        for hand_side in ["left_hand", "right_hand"]:
            for frame_idx in tqdm(range_obj):
                # !!!!!!!!!!
                if frame_idx >= 9000:
                    break

                if frame_idx not in pred_joints_smpl[hand_side]:
                    continue

                # construct joints to test
                neighbor_joints = []
                neighbor_joints_dict = {}
                for idx in range(frame_idx - window_size, frame_idx + window_size + 1):
                    if idx < 0 or idx >= virtual_frame_count:
                        continue

                    # TODO: allow usage of interpolated results?
                    if idx in pred_joints_smpl[hand_side]:
                        neighbor_joints.append((idx, pred_joints_smpl[hand_side][idx]))
                        neighbor_joints_dict[idx] = pred_joints_smpl[hand_side][idx]

                if len(neighbor_joints) < 3:
                    continue

                # inlier check
                link = {}  # default: self
                component_sizes = {}  # default: 1

                def find(identifier):
                    while link.get(identifier, identifier) != identifier:
                        identifier = link.get(identifier, identifier)
                    return identifier

                def union(identifier_1, identifier_2):
                    identifier_1 = find(identifier_1)
                    identifier_2 = find(identifier_2)
                    cs1 = component_sizes.get(identifier_1, 1)
                    cs2 = component_sizes.get(identifier_2, 1)
                    if cs1 >= cs2:
                        component_sizes[identifier_1] = cs1 + cs2
                        link[identifier_2] = identifier_1
                        if identifier_2 in component_sizes:
                            del component_sizes[identifier_2]
                    else:
                        component_sizes[identifier_2] = cs1 + cs2
                        link[identifier_1] = identifier_2
                        if identifier_1 in component_sizes:
                            del component_sizes[identifier_1]

                # determine relationships between neighbors
                for neigh_idx_1 in range(len(neighbor_joints)):
                    for neigh_idx_2 in range(neigh_idx_1 + 1, len(neighbor_joints)):
                        dist = np.linalg.norm(
                            neighbor_joints[neigh_idx_1][1]
                            - neighbor_joints[neigh_idx_2][1]
                        )
                        if dist <= similarity_threshold:
                            union(
                                neighbor_joints[neigh_idx_1][0],
                                neighbor_joints[neigh_idx_2][0],
                            )

                # find largest component
                components_sorted = sorted(
                    list(component_sizes.keys()), key=lambda k: component_sizes[k]
                )
                if len(components_sorted) == 0:
                    continue
                largest_component_id = components_sorted[-1]
                largest_component_members = []
                other_component_members = []
                for neighbor_data in neighbor_joints:
                    neigh_idx, _ = neighbor_data
                    if find(neigh_idx) == largest_component_id:
                        largest_component_members.append(neigh_idx)
                    else:
                        other_component_members.append(neigh_idx)
                if len(largest_component_members) == 0 or len(
                    largest_component_members
                ) == len(neighbor_joints):
                    continue
                largest_component_members.sort()

                if frame_idx not in largest_component_members or (
                    len(components_sorted) >= 2
                    and 2 * components_sorted[-2] >= components_sorted[-1]
                ):
                    print(f"Outlier frame: {frame_idx} for {hand_side}")

                    # fit splines
                    frame_new_joints = np.zeros_like(neighbor_joints[0])

                    # !!!!!!!!!!
                    if False:
                        for pred_joint_idx in range(neighbor_joints[0].shape[-2]):
                            for coord_idx in range(neighbor_joints[0].shape[-1]):
                                spline = UnivariateSpline(
                                    x=[
                                        m - largest_component_members[0]
                                        for m in largest_component_members
                                    ],
                                    y=[
                                        neighbor_joints_dict[m][pred_joint_idx][
                                            coord_idx
                                        ]
                                        for m in largest_component_members
                                    ],
                                )
                                # regress this value:
                                frame_new_joints[pred_joint_idx] = spline(frame_idx)

                    new_joints[hand_side][frame_idx] = frame_new_joints

                    # attempt to optimize to get beta

                    """
                    hand_mocap.model_regressor.smplx

                        def forward(
                            self,
                            betas: Optional[Tensor] = None,
                            global_orient: Optional[Tensor] = None,
                            body_pose: Optional[Tensor] = None,
                            left_hand_pose: Optional[Tensor] = None,
                            right_hand_pose: Optional[Tensor] = None,
                            transl: Optional[Tensor] = None,
                            expression: Optional[Tensor] = None,
                            jaw_pose: Optional[Tensor] = None,
                            leye_pose: Optional[Tensor] = None,
                            reye_pose: Optional[Tensor] = None,
                            return_verts: bool = True,
                            return_full_pose: bool = False,
                            pose2rot: bool = True,
                            return_shaped: bool = True,
                            **kwargs
                        ) -> SMPLXOutput:
                    """

                # perform voting
                # for lcm in largest_component_members:
                #     votes[hand_side][lcm] = votes[hand_side].get(lcm, 0) + 1

                for ocm in other_component_members:
                    votes[hand_side][ocm] = votes[hand_side].get(ocm, 0) - 1

        betas_ewm = {}
        thetas_ewm = {}
        cameras_ewm = {}
        for hand_side in ["left_hand", "right_hand"]:
            # ewma_vectorized_2d(data, alpha, axis=None, offset=None, dtype=None, order='C', out=None)

            betas_ewm[hand_side] = ewma_vectorized_2d(
                np.stack(betas[hand_side]), alpha, axis=1
            )
            thetas_ewm[hand_side] = ewma_vectorized_2d(
                np.stack(thetas[hand_side]), alpha, axis=1
            )
            cameras_ewm[hand_side] = ewma_vectorized_2d(
                np.stack(cameras[hand_side]), alpha, axis=1
            )

        print()
        print("*** Step 2: smoothing data ***")
        print()

        for frame_seq_idx, frame_idx in tqdm(enumerate(range_obj)):
            frame_id = fmt_frame(video_id, frame_idx)
            pkl_path = join(out_dir, "mocap", f"{frame_id}_prediction_result.pkl")

            if not isfile(pkl_path):
                pkl_path = pkl_path + ".zip"

            print(f"{pkl_path=} {isfile(pkl_path)=}")
            if not isfile(pkl_path):
                continue

            try:
                result_data = read_pkl(pkl_path)
                if isinstance(result_data["pred_output_list"], list):
                    pred_output_list = result_data["pred_output_list"][0]
                elif isinstance(result_data["pred_output_list"], dict):
                    pred_output_list = result_data["pred_output_list"]
                else:
                    raise NotImplementedError()
            except ToggleableException as ex:
                print(f"Error while processing {frame_id}:", ex)
                continue

            info_str = ""

            votes_left = votes["left_hand"].get(frame_idx, 0)
            votes_right = votes["right_hand"].get(frame_idx, 0)
            if votes_left < 0:
                info_str += f"\nOutlier (left side)"
            elif votes_right < 0:
                info_str += f"\nOutlier (right side)"
            # else:
            #    continue

            for hand_side in ["left_hand", "right_hand"]:
                if hand_side not in pred_output_list:
                    continue
                hand_side_data = pred_output_list[hand_side]
                if hand_side_data is None or len(hand_side_data) == 0:
                    pred_output_list[hand_side] = None  # needed for __calc_hand_mesh
                    continue

                # mark as outlier in PKL
                frame_id = fmt_frame(video_id, frame_idx)
                zip_path = CHANNEL_FRAME_PATH_FUNCTS["hand_mesh"](
                    video_id, frame_idx, frame_id, args.version
                )

                # !!!!!!!!!!
                frame_handside_votes = votes[hand_side].get(frame_idx, 0)
                if frame_handside_votes < 0:
                    try:
                        result_data_2 = read_pkl(zip_path)
                        if isinstance(result_data_2["pred_output_list"], list):
                            pred_output_list_2 = result_data_2["pred_output_list"][0]
                        elif isinstance(result_data_2["pred_output_list"], dict):
                            pred_output_list_2 = result_data_2["pred_output_list"]
                        else:
                            raise NotImplementedError()

                        if (
                            isinstance(pred_output_list_2, dict)
                            and hand_side in pred_output_list_2
                        ):
                            pred_output_list_2[hand_side]["outlier"] = True

                        with zipfile.ZipFile(
                            zip_path, "w", zipfile.ZIP_DEFLATED, False
                        ) as zip_file:
                            zip_file.writestr(
                                os.path.basename(zip_path), pickle.dumps(result_data_2)
                            )
                    except ToggleableException as ex:
                        print(f"Error while reading {zip_path}:", ex)

                ###########

                # adapted from frankmocap/handmocap/hand_mocap_api.py

                thetas_np = hand_side_data["pred_hand_pose"]
                betas_np = hand_side_data["pred_hand_betas"]
                cam = hand_side_data["pred_camera"]

                try:
                    left_idx = frame_idxs["left_hand"].index(frame_idx)
                except:
                    left_idx = -1

                try:
                    right_idx = frame_idxs["right_hand"].index(frame_idx)
                except:
                    right_idx = -1

                if left_idx > 2 and hand_side == "left_hand":
                    idx = left_idx
                    last_frame_idx = frame_idxs[hand_side][idx - 1]
                    if frame_idx - last_frame_idx < 10:
                        last_betas = betas[hand_side][idx - 1]
                        this_betas = betas[hand_side][idx]
                        last_thetas = thetas[hand_side][idx - 1]
                        this_thetas = thetas[hand_side][idx]
                        last_cam = cameras[hand_side][idx - 1]
                        this_cam = cameras[hand_side][idx]

                        last_joints_smpl = pred_joints_smpl_list[hand_side][idx - 1]
                        this_joints_smpl = pred_joints_smpl_list[hand_side][idx]

                        beta_change = np.linalg.norm(this_betas - last_betas)
                        theta_change = np.linalg.norm(this_thetas - last_thetas)
                        cam_change = np.linalg.norm(this_cam - last_cam)
                        joint_change = np.linalg.norm(
                            last_joints_smpl - this_joints_smpl
                        )

                        info_str += (
                            f"\nbeta_change ({hand_side}): {'%.4f' % beta_change}"
                        )
                        info_str += (
                            f"\ntheta_change ({hand_side}): {'%.4f' % theta_change}"
                        )
                        info_str += (
                            f"\njoint_change ({hand_side}): {'%.4f' % joint_change}"
                        )
                        info_str += f"\ncam_change ({hand_side}): {'%.4f' % cam_change}"

                        # works:
                        # if frame_idx - last_frame_idx == 2:
                        #     info_str += f"\ninterpol ({hand_side}): True"
                        #     betas_np = np.expand_dims((this_betas + last_betas) / 2, 0)
                        #     thetas_np = np.expand_dims((this_thetas + last_thetas) / 2, 0)
                        #     cam = (this_cam + last_cam) / 2

                if right_idx > 2 and hand_side == "right_hand":
                    idx = right_idx
                    last_frame_idx = frame_idxs[hand_side][idx - 1]
                    if frame_idx - last_frame_idx < 10:
                        last_betas = betas[hand_side][idx - 1]
                        this_betas = betas[hand_side][idx]
                        last_thetas = thetas[hand_side][idx - 1]
                        this_thetas = thetas[hand_side][idx]
                        last_cam = cameras[hand_side][idx - 1]
                        this_cam = cameras[hand_side][idx]

                        last_joints_smpl = pred_joints_smpl_list[hand_side][idx - 1]
                        this_joints_smpl = pred_joints_smpl_list[hand_side][idx]

                        beta_change = np.linalg.norm(this_betas - last_betas)
                        theta_change = np.linalg.norm(this_thetas - last_thetas)
                        cam_change = np.linalg.norm(this_cam - last_cam)
                        joint_change = np.linalg.norm(
                            last_joints_smpl - this_joints_smpl
                        )

                        info_str += (
                            f"\nbeta_change ({hand_side}): {'%.4f' % beta_change}"
                        )
                        info_str += (
                            f"\ntheta_change ({hand_side}): {'%.4f' % theta_change}"
                        )
                        info_str += (
                            f"\njoint_change ({hand_side}): {'%.4f' % joint_change}"
                        )
                        info_str += f"\ncam_change ({hand_side}): {'%.4f' % cam_change}"

                        # works:
                        # if frame_idx - last_frame_idx == 2:
                        #     info_str += f"\ninterpol ({hand_side}): True"
                        #     betas_np = np.expand_dims((this_betas + last_betas) / 2, 0)
                        #     thetas_np = np.expand_dims((this_thetas + last_thetas) / 2, 0)
                        #     cam = (this_cam + last_cam) / 2

                idx = frame_idxs[hand_side].index(frame_idx)
                # thetas_np = np.expand_dims(thetas_ewm[hand_side][idx], 0)
                # betas_np = np.expand_dims(betas_ewm[hand_side][idx], 0)

                thetas_torch = torch.from_numpy(thetas_np).to(device)
                betas_torch = torch.from_numpy(betas_np).to(device)

                vert_smplcoord, hand_faces = __calc_hand_mesh(
                    hand_side,
                    thetas_torch,
                    betas_torch,
                    hand_mocap.model_regressor.smplx,
                    device,
                )

                hand_side_data["pred_vertices_smpl"] = vert_smplcoord
                vert_smplcoord = hand_side_data["pred_vertices_smpl"]

                # cam = cameras_ewm[hand_side][frame_seq_idx]

                cam_scale = cam[0]
                cam_trans = cam[1:]
                pred_joints = hand_side_data["pred_joints_smpl"]
                joints_smplcoord = pred_joints.copy()

                vert_bboxcoord = convert_smpl_to_bbox(
                    vert_smplcoord, cam_scale, cam_trans, bAppTransFirst=True
                )  # SMPL space -> bbox space
                joints_bboxcoord = convert_smpl_to_bbox(
                    joints_smplcoord, cam_scale, cam_trans, bAppTransFirst=True
                )  # SMPL space -> bbox space

                hand_boxScale_o2n = hand_side_data["bbox_scale_ratio"]
                hand_bboxTopLeft = hand_side_data["bbox_top_left"]

                vert_imgcoord = convert_bbox_to_oriIm(
                    vert_bboxcoord,
                    hand_boxScale_o2n,
                    hand_bboxTopLeft,
                    img_width,
                    img_height,
                )
                hand_side_data["pred_vertices_img"] = vert_imgcoord

            img_original = reader.get_frame(frame_idx)
            if img_original is not None:
                img_original_bgr = img_original[:, :, ::-1]
                # extract mesh for rendering (vertices in image space and faces) from pred_output_list
                # restore original structure:
                pred_mesh_list = demo_utils.extract_mesh_from_output(
                    result_data["pred_output_list"]
                )
                hand_bbox_list = result_data["hand_bbox_list"]

                output_path = join(sample_vis_dir, f"{frame_id}.jpg")
                res_img = visualizer.visualize(
                    # !!!!!!!!!!
                    np.zeros_like(img_original_bgr),
                    pred_mesh_list=pred_mesh_list,
                    hand_bbox_list=hand_bbox_list,
                )

                # superimpose some info

                if info_str != "":
                    res_img_pil = Image.fromarray(res_img)
                    draw = ImageDraw.Draw(res_img_pil)
                    draw.text((10, 10), text=info_str, font=font)
                    res_img = np.array(res_img_pil)

                demo_utils.save_res_img(
                    dirname(output_path), basename(output_path), res_img
                )
                print(f"Saved {output_path}")


def main():
    with torch.no_grad():
        options = DemoOptions()
        options.parser.add_argument(
            "--generator_videos", action="append", type=str, default=None
        )
        options.parser.add_argument(
            "--version", type=str, default=DEFAULT_HAND_MESH_VERSION
        )
        options.parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
        options.parser.add_argument("--max_width", type=int, default=1280)
        options.parser.add_argument("--max_height", type=int, default=720)
        options.parser.add_argument(
            "--ewm_span_rel_to_fps", type=float, default=(1.0 / EK_ASSUMED_FPS)
        )
        #  ewm_span_abs = round(EK_ASSUMED_FPS * args.ewm_span_rel_to_fps)
        args = options.parse()
        args.use_smplx = True
        args.save_pred_pkl = True

        device = args.device or "cpu"
        assert torch.cuda.is_available(), "Current version only supports GPU"

        torch.cuda.set_device(device)

        # Set Bbox detector
        bbox_detector = HandBboxDetector(args.view_type, device=device)

        # Set Mocap regressor
        hand_mocap = HandMocap(args.checkpoint_hand, args.smpl_dir, device=device)

        # Set Visualizer
        if args.renderer_type in ["pytorch3d", "opendr"]:
            from renderer.screen_free_visualizer import Visualizer
        else:
            from renderer.visualizer import Visualizer
        visualizer = Visualizer(args.renderer_type)

        # run
        run_hand_mocap(
            args, bbox_detector, hand_mocap, visualizer, args.generator_videos
        )


if __name__ == "__main__":
    main()
