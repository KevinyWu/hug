"""MANO hand model wrapper with 6D rotation support."""

import warnings

import numpy as np
import torch
import torch.nn as nn
from manotorch.manolayer import ManoLayer
from manotorch.utils.geometry import rotation_to_axis_angle

from ..utils.data_keys import MANO_MODELS_FOLDER
from ..utils.transform_utils import six_d_to_rotation_matrix

warnings.filterwarnings(
    "ignore",
    message=r"Using torch\.cross without specifying the dim arg is deprecated\.",
    category=UserWarning,
)


class MANO(nn.Module):
    """Wrapper for MANO hand model.

    Handles conversion from 6D rotations to axis-angle and runs MANO forward pass.
    """

    def __init__(
        self,
        side: str = "right",
        use_pca: bool = False,
        flat_hand_mean: bool = True,
        ncomps: int = 45,
    ):
        super().__init__()
        self.mano_layer = ManoLayer(
            rot_mode="axisang",
            side=side,
            center_idx=0,  # Wrist at origin
            mano_assets_root=MANO_MODELS_FOLDER,
            use_pca=use_pca,
            flat_hand_mean=flat_hand_mean,
            ncomps=ncomps,
        )
        # Freeze MANO
        for param in self.mano_layer.parameters():
            param.requires_grad = False

    def decode_mano_params(self, mano_params: torch.Tensor):
        """Decode 99D MANO pose to components.

        Args:
            mano_params: (B, 99) = t(3) + R_6d(6) + pose_6d(90), translation in meters.

        Returns:
            t: (B, 3) metric wrist translation in camera frame.
            R_6d: (B, 6) wrist rotation 6D.
            pose_6d: (B, 15, 6) finger joint rotations 6D.
        """
        t = mano_params[:, :3]
        R_6d = mano_params[:, 3:9]
        pose_6d = mano_params[:, 9:].reshape(-1, 15, 6)
        return t, R_6d, pose_6d

    def forward(
        self,
        mano_params: torch.Tensor,
        betas: torch.Tensor = None,
    ):
        """Run MANO forward pass.

        Args:
            mano_params: (B, 99) = t(3) + R_6d(6) + pose_6d(90), translation in meters.
            betas: (B, 10) shape parameters (optional, defaults to zeros).

        Returns:
            dict with landmarks_3d (B, 21, 3), vertices (B, 778, 3), etc.
        """
        B = mano_params.shape[0]
        device = mano_params.device

        t, R_6d, pose_6d = self.decode_mano_params(mano_params)

        R_3x3 = six_d_to_rotation_matrix(R_6d)
        R_aa = rotation_to_axis_angle(R_3x3)
        pose_3x3 = six_d_to_rotation_matrix(pose_6d.reshape(-1, 6))
        pose_aa = rotation_to_axis_angle(pose_3x3)
        pose_aa = pose_aa.reshape(B, 15 * 3)
        pose_coeffs = torch.cat([R_aa, pose_aa], dim=-1)

        if betas is None:
            betas = torch.zeros(B, 10, device=device)

        mano_output = self.mano_layer(pose_coeffs, betas)

        return {
            "landmarks_3d": mano_output.joints,
            "vertices": mano_output.verts,
            "t": t,
            "R_3x3": R_3x3.reshape(B, 3, 3),
            "pose_3x3": pose_3x3.reshape(B, 15, 3, 3),
        }


def project_3d_to_2d(points_3d: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Project 3D points to 2D using camera intrinsics. (N,3) -> (N,2)."""
    proj = points_3d @ K.T
    return proj[:, :2] / (proj[:, 2:3] + 1e-8)


@torch.no_grad()
def mano_params_to_grasp_dict(
    mano_params: torch.Tensor,
    betas: torch.Tensor,
    mano_model: MANO,
    camera_K: np.ndarray,
    mesh_faces: np.ndarray,
) -> dict:
    """Convert 99D model output to full Grasp-format dict (all numpy).

    Args:
        mano_params: (99,) predicted MANO params.
        betas: (10,) shape params.
        mano_model: MANO instance.
        camera_K: (3, 3) camera intrinsics.
        mesh_faces: (F, 3) right hand mesh faces.
    """
    device = next(mano_model.buffers()).device
    params = mano_params.unsqueeze(0).to(device)
    shape = betas.unsqueeze(0).to(device)

    t, R_6d, pose_6d = mano_model.decode_mano_params(params)

    out = mano_model(params, betas=shape)
    joints_wrist = out["landmarks_3d"][0]
    verts_wrist = out["vertices"][0]
    R_3x3 = out["R_3x3"][0]
    pose_3x3 = out["pose_3x3"][0]

    translation = t[0]  # (3,) metric

    landmarks_3d = joints_wrist + translation
    verts_cam = verts_wrist + translation

    T_camera_wrist = np.eye(4, dtype=np.float32)
    T_camera_wrist[:3, :3] = R_3x3.cpu().numpy()
    T_camera_wrist[:3, 3] = translation.cpu().numpy()

    pose_aa = rotation_to_axis_angle(pose_3x3.reshape(-1, 3, 3))

    landmarks_3d_np = landmarks_3d.cpu().numpy()
    landmarks_2d = project_3d_to_2d(landmarks_3d_np, camera_K)

    return {
        "pose": pose_aa.cpu().numpy().reshape(1, 15, 3),
        "pose_6d": pose_6d[0].cpu().numpy().reshape(1, 15, 6),
        "shape": betas.cpu().numpy().reshape(1, 10),
        "landmarks_3d": landmarks_3d_np,
        "landmarks_2d": landmarks_2d,
        "T_camera_wrist": T_camera_wrist,
        "R_6d": R_6d[0].cpu().numpy().reshape(1, 6),
        "t": t[0].cpu().numpy().reshape(1, 3),
        "mesh_vertices": verts_cam.cpu().numpy(),
        "mesh_faces": mesh_faces,
    }


@torch.no_grad()
def mano_params_to_animation(
    mano_params: torch.Tensor,
    betas: torch.Tensor,
    mano_model: MANO,
    n_frames: int = 64,
    pre_offset_m: tuple[float, float] = (0.03, 0.03),
    thumb_pre_bend_rad: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Batched MANO verts + joints for the pre-grasp -> grasp lerp (phase A).

    Linearly interpolates an open pre-grasp into the predicted grasp with wrist
    orientation held fixed: the wrist position lerps from a point offset back
    along the wrist-local x/y to the grasp, and the fingers lerp from open (zeros,
    with a thumb pre-bend) to the predicted axis-angles. Mirrors phase A of
    `aria2mesh/grasping/utils/sim_plan.plan_from_world_wrist`.

    Args:
        mano_params: (99,) predicted MANO params (t + R_6d + pose_6d).
        betas: (10,) shape params.
        mano_model: MANO instance.
        n_frames: number of interpolation frames.
        pre_offset_m: wrist-local (x, y) pre-grasp offset in meters.
        thumb_pre_bend_rad: thumb-base bend held in the open pre-grasp.

    Returns:
        verts_seq: (n_frames, 778, 3) camera-frame vertices, frame 0 pre-grasp.
        joints_seq: (n_frames, 21, 3) camera-frame joints; frame -1 == grasp.
    """
    device = next(mano_model.buffers()).device
    params = mano_params.unsqueeze(0).to(device)
    shape = betas.unsqueeze(0).to(device)

    t, R_6d, pose_6d = mano_model.decode_mano_params(params)
    R_3x3 = six_d_to_rotation_matrix(R_6d)
    R_aa = rotation_to_axis_angle(R_3x3)  # (1, 3)
    pose_aa = rotation_to_axis_angle(
        six_d_to_rotation_matrix(pose_6d.reshape(-1, 6))
    ).reshape(15, 3)

    finger_start_aa = torch.zeros(15, 3, device=device)
    finger_start_aa[12, 0] = thumb_pre_bend_rad
    finger_start_aa[12, 1] = thumb_pre_bend_rad

    alphas = torch.linspace(0.0, 1.0, n_frames, device=device).reshape(-1, 1, 1)
    finger_seq = (1.0 - alphas) * finger_start_aa + alphas * pose_aa  # (n, 15, 3)

    pose_coeffs = torch.cat(
        [R_aa.expand(n_frames, 3), finger_seq.reshape(n_frames, 45)], dim=-1
    )
    out = mano_model.mano_layer(pose_coeffs, shape.expand(n_frames, -1))
    verts = out.verts  # (n, 778, 3), wrist frame
    joints = out.joints  # (n, 21, 3), wrist frame

    t_grasp = t[0]  # (3,)
    t_pre = t_grasp + R_3x3[0] @ torch.tensor(
        [pre_offset_m[0], pre_offset_m[1], 0.0], device=device
    )
    pos_seq = (1.0 - alphas.reshape(-1, 1)) * t_pre + alphas.reshape(-1, 1) * t_grasp

    verts_cam = verts + pos_seq.unsqueeze(1)
    joints_cam = joints + pos_seq.unsqueeze(1)

    return verts_cam.cpu().numpy(), joints_cam.cpu().numpy()
