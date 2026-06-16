"""GraspData from aria2mano with condition_point for point conditioning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class GraspData:
    """Grasp data class.

    Images are stored as encoded bytes for compact serialization:
    - image: JPEG-encoded 224x224 image (RGB or grayscale as 3-ch)
    - depth: PNG-encoded 224x224 depth map (uint16, 1mm units)
    - object_mask: PNG-encoded 224x224 binary mask

    Decode with cv2.imdecode(np.frombuffer(data, np.uint8), flag).
    """

    object_name: str  # Object name
    frame_index: int  # Frame index within the video
    grasp_index: int  # Grasp frame index within the video
    camera: CameraIntrinsics  # 224x224 camera intrinsics
    camera_original: CameraIntrinsics  # Original-resolution intrinsics (RGB:
    # FOV-cropped square; grayscale: stereo-left native)
    grasp: Optional[Grasp] = None  # Grasp label (right hand) in camera frame
    image: bytes = b""  # JPEG-encoded 224x224 image
    depth: bytes = b""  # PNG-encoded 224x224 depth (uint16, 1mm)
    object_mask: bytes = b""  # PNG-encoded 224x224 binary mask
    condition_point: Optional[np.ndarray] = None  # (2,) [u, v]: model
    # conditioning pixel in 224x224 image coords


@dataclass
class CameraIntrinsics:
    """Camera intrinsics data class."""

    K: np.ndarray  # Camera intrinsics matrix in pixel coordinates (3, 3)
    width: int  # Width of image
    height: int  # Height of image


@dataclass
class Grasp:
    """Grasp data class for a single right hand grasp of a single object."""

    pose: np.ndarray  # MANO pose parameters (1, 15, 3)
    pose_6d: np.ndarray  # MANO pose parameters in 6D representation (1, 15, 6)
    shape: np.ndarray  # MANO shape parameters (1, 10)
    landmarks_3d: np.ndarray  # 3D MANO hand landmarks in camera frame (21, 3)
    landmarks_2d: np.ndarray  # 2D MANO hand landmarks in 224x224 image (21, 2)
    T_camera_wrist: np.ndarray  # Wrist to camera transform (4, 4)
    R_6d: np.ndarray  # Wrist to camera rotation in 6D representation (1, 6)
    t: np.ndarray  # Wrist translation in camera frame (1, 3)
    mesh_vertices: np.ndarray  # MANO hand mesh vertices in camera frame (778, 3)
    mesh_faces: Optional[np.ndarray] = None  # MANO right hand mesh faces (1552, 3)
