"""Viser 3D visualization utilities."""

import base64
from collections import OrderedDict
from typing import Optional, Tuple

import cv2
import numpy as np
import viser
import viser.transforms as tf

from .visualization_utils import (
    MANO_KEYPOINT_COLOR,
    MANO_SKELETON_COLOR,
    MANO_SKELETON_PAIRS,
    OBJECT_POINTS_COLOR,
)

COLORS = [
    (90, 255, 90),  # Green
    (255, 100, 100),  # Red
    (100, 180, 255),  # Blue
    (255, 200, 60),  # Yellow
    (200, 130, 255),  # Purple
]


def _to_float(c: Tuple[int, int, int]) -> Tuple[float, float, float]:
    """Convert RGB (0-255) to float (0-1)."""
    return (c[0] / 255.0, c[1] / 255.0, c[2] / 255.0)


def get_skeleton_from_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """Convert 21 landmarks to (N_segments, 2, 3) line segments."""
    segments = np.stack(
        [
            np.stack([landmarks[a], landmarks[b]], axis=0)
            for a, b in MANO_SKELETON_PAIRS
        ],
        axis=0,
    )
    return segments


def add_hand_skeleton(
    server: viser.ViserServer,
    path: str,
    landmarks: np.ndarray,
    color: Tuple[int, int, int] = MANO_SKELETON_COLOR,
    line_width: float = 4.0,
    visible: bool = True,
) -> viser.LineSegmentsHandle:
    """Add hand skeleton to Viser scene."""
    segments = get_skeleton_from_landmarks(landmarks)
    handle = server.scene.add_line_segments(
        path,
        points=segments,
        colors=np.tile(color, (len(segments), 2, 1)).astype(np.uint8),
        line_width=line_width,
    )
    handle.visible = visible
    return handle


def add_hand_keypoints(
    server: viser.ViserServer,
    path: str,
    landmarks: np.ndarray,
    color: Tuple[int, int, int] = MANO_KEYPOINT_COLOR,
    point_size: float = 0.005,
    visible: bool = True,
) -> viser.PointCloudHandle:
    """Add hand keypoints to Viser scene."""
    handle = server.scene.add_point_cloud(
        path,
        points=landmarks,
        colors=np.tile(color, (len(landmarks), 1)).astype(np.uint8),
        point_size=point_size,
        point_shape="rounded",
    )
    handle.visible = visible
    return handle


def add_mano_mesh(
    server: viser.ViserServer,
    path: str,
    vertices: np.ndarray,
    faces: np.ndarray,
    color: Tuple[float, float, float] = (0.5, 0.5, 0.5),
    opacity: float = 0.7,
    visible: bool = True,
) -> viser.MeshHandle:
    """Add MANO mesh to Viser scene."""
    handle = server.scene.add_mesh_simple(
        name=path,
        vertices=vertices,
        faces=faces,
        color=_to_float(color),
    )
    handle.opacity = opacity
    handle.visible = visible
    return handle


def add_wrist_frame(
    server: viser.ViserServer,
    path: str,
    T_camera_wrist: np.ndarray,
    axes_length: float = 0.12,
    axes_radius: float = 0.004,
    visible: bool = False,
) -> viser.FrameHandle:
    """Add wrist frame to Viser scene."""
    handle = server.scene.add_frame(
        name=path,
        wxyz=tf.SO3.from_matrix(T_camera_wrist[:3, :3]).wxyz,
        position=T_camera_wrist[:3, 3],
        axes_length=axes_length,
        axes_radius=axes_radius,
    )
    handle.visible = visible
    return handle


def add_object_point_cloud(
    server: viser.ViserServer,
    path: str,
    points: np.ndarray,
    point_size: float = 0.01,
    color: Tuple[int, int, int] = OBJECT_POINTS_COLOR,
    visible: bool = True,
) -> Optional[viser.PointCloudHandle]:
    """Add object point cloud to Viser scene."""
    if points is None or len(points) == 0:
        return None
    handle = server.scene.add_point_cloud(
        path,
        points=points,
        colors=np.tile(color, (len(points), 1)).astype(np.uint8),
        point_size=point_size,
        point_shape="rounded",
    )
    handle.visible = visible
    return handle


class PredictionStore:
    """Manages persisted grasp predictions in the 3D scene."""

    def __init__(self):
        self.predictions: OrderedDict[str, list] = OrderedDict()
        self.wrist_frames: OrderedDict[str, object] = OrderedDict()
        self.point_handles: OrderedDict[str, object] = OrderedDict()
        self._color_idx = 0

    def next_color(self) -> tuple[int, int, int]:
        """Get next prediction color."""
        color = COLORS[self._color_idx % len(COLORS)]
        self._color_idx += 1
        return color

    def add(self, name: str, handles: list, wrist_handle=None, point_handle=None):
        """Add prediction to store."""
        self.predictions[name] = handles
        if wrist_handle is not None:
            self.wrist_frames[name] = wrist_handle
        if point_handle is not None:
            self.point_handles[name] = point_handle

    def set_visible(
        self, visible: bool, show_wrist: bool = False, show_point: bool = True
    ):
        """Set visibility of predictions, wrist frames, and point handles."""
        for handles in self.predictions.values():
            for h in handles:
                try:
                    h.visible = visible
                except Exception:
                    pass
        for h in self.wrist_frames.values():
            try:
                h.visible = visible and show_wrist
            except Exception:
                pass
        for h in self.point_handles.values():
            try:
                h.visible = visible and show_point
            except Exception:
                pass

    def set_wrist_visible(self, visible: bool, show_pred: bool = True):
        """Set visibility of wrist frames."""
        for h in self.wrist_frames.values():
            try:
                h.visible = visible and show_pred
            except Exception:
                pass

    def set_point_visible(self, visible: bool, show_pred: bool = True):
        """Set visibility of point condition handles."""
        for h in self.point_handles.values():
            try:
                h.visible = visible and show_pred
            except Exception:
                pass

    def set_mesh_opacity(self, opacity: float):
        """Set opacity of predictions."""
        for handles in self.predictions.values():
            for h in handles:
                if hasattr(h, "opacity"):
                    try:
                        h.opacity = opacity
                    except Exception:
                        pass

    def clear_last(self):
        """Clear last prediction."""
        if not self.predictions:
            return
        name, handles = self.predictions.popitem(last=True)
        for h in handles:
            try:
                h.remove()
            except Exception:
                pass
        wrist = self.wrist_frames.pop(name, None)
        if wrist is not None:
            try:
                wrist.remove()
            except Exception:
                pass
        point = self.point_handles.pop(name, None)
        if point is not None:
            try:
                point.remove()
            except Exception:
                pass

    def clear_all(self):
        """Clear all predictions."""
        for handles in self.predictions.values():
            for h in handles:
                try:
                    h.remove()
                except Exception:
                    pass
        for h in self.wrist_frames.values():
            try:
                h.remove()
            except Exception:
                pass
        for h in self.point_handles.values():
            try:
                h.remove()
            except Exception:
                pass
        self.predictions.clear()
        self.wrist_frames.clear()
        self.point_handles.clear()


def backproject_depth_to_point_cloud(
    depth_image: np.ndarray,
    rgb_image: np.ndarray,
    camera_K: np.ndarray,
    cam_w: float,
    cam_h: float,
    max_depth: float,
    mask: Optional[np.ndarray] = None,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Backproject depth to 3D point cloud with RGB colors.

    Args:
        depth_image: (H, W) uint16 depth in millimeters.
        rgb_image: (H, W, 3) uint8 RGB at same resolution as depth.
        camera_K: (3, 3) intrinsics at original crop resolution.
        cam_w, cam_h: original crop resolution for scaling K.
        max_depth: max depth in meters to keep.
        mask: optional (H, W) binary mask for additional filtering.
    """
    depth_h, depth_w = depth_image.shape[:2]
    scale_x = depth_w / cam_w
    scale_y = depth_h / cam_h
    K = camera_K.copy()
    K[0, 0] *= scale_x
    K[1, 1] *= scale_y
    K[0, 2] *= scale_x
    K[1, 2] *= scale_y

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    u, v = np.meshgrid(np.arange(depth_w), np.arange(depth_h))
    z = depth_image.astype(np.float32) / 1000.0
    valid = (z > 0) & (z < max_depth)
    if mask is not None:
        valid = valid & (mask > 0)
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    points = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    colors = rgb_image.reshape(-1, 3)
    valid_flat = valid.flatten()
    if valid_flat.sum() == 0:
        return None
    return points[valid_flat], colors[valid_flat]


def image_to_data_url(image_rgb: np.ndarray) -> str:
    """Convert RGB image to data URL."""
    _, buf = cv2.imencode(".jpg", cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


def make_clickable_image_html(image_rgb: np.ndarray, bridge_label: str) -> str:
    """Make clickable image HTML."""
    data_url = image_to_data_url(image_rgb)
    bridge_label_js = bridge_label.replace("\\", "\\\\").replace("'", "\\'")
    return (
        f'<img src="{data_url}" '
        f'style="width:100%;display:block;cursor:crosshair;" '
        f'onload="(function(){{'
        f"var label=Array.from(document.querySelectorAll('label')).find(function(el){{"
        f"return el.textContent==='{bridge_label_js}';"
        f"}});"
        f"if(!label) return;"
        f"var row=label.closest('.mantine-Flex-root');"
        f"if(row) row.style.display='none';"
        f'}})()" '
        f'onclick="(function(e){{'
        f"var label=Array.from(document.querySelectorAll('label')).find(function(el){{"
        f"return el.textContent==='{bridge_label_js}';"
        f"}});"
        f"if(!label) return;"
        f"var row=label.closest('.mantine-Flex-root');"
        f"if(row) row.style.display='none';"
        f"var input=row?row.querySelector('input,textarea'):null;"
        f"if(!input) return;"
        f"var r=e.target.getBoundingClientRect();"
        f"var u=(e.clientX-r.left)/r.width;"
        f"var v=(e.clientY-r.top)/r.height;"
        f"var setter=Object.getOwnPropertyDescriptor("
        f"window.HTMLInputElement.prototype,'value').set;"
        f"setter.call(input, u.toFixed(6)+','+v.toFixed(6));"
        f"input.dispatchEvent(new Event('input',{{bubbles:true}}));"
        f'}})(event)">'
    )


def make_hidden_bridge_bootstrap_html(bridge_label: str) -> str:
    """Make hidden bridge bootstrap HTML."""
    bridge_label_js = bridge_label.replace("\\", "\\\\").replace("'", "\\'")
    return (
        '<img src="x" style="display:none" '
        'onerror="(function(){'
        "var label=Array.from(document.querySelectorAll('label')).find(function(el){"
        f"return el.textContent==='{bridge_label_js}';"
        "});"
        "if(!label) return;"
        "var row=label.closest('.mantine-Flex-root');"
        "if(row) row.style.display='none';"
        '})()">'
    )
