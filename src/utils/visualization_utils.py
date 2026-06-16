"""Visualization utilities."""

from typing import Optional, Tuple

import cv2
import numpy as np
import plotly.graph_objects as go
import torch
from rich import box
from rich.table import Table

from ..models.mano import project_3d_to_2d

# Colors (RGB 0-255). Converted to BGR at cv2 call sites via _bgr().
MANO_MESH_COLOR = (90, 200, 255)  # Light blue
MANO_BORDER_COLOR = (90, 200, 255)
MANO_FINGERTIP_COLOR = (255, 0, 255)  # Magenta
MANO_KEYPOINT_COLOR = (0, 128, 255)  # Blue
MANO_SKELETON_COLOR = (150, 200, 255)  # Light blue
MASK_COLOR = (200, 200, 0)  # Yellow
MASK_BORDER_COLOR = (255, 255, 0)  # Bright yellow
PRED_MESH_COLOR = (90, 255, 90)  # Green
PRED_SKELETON_COLOR = (50, 255, 100)  # Green
PRED_KEYPOINT_COLOR = (0, 200, 50)  # Dark green
OBJECT_POINTS_COLOR = (255, 255, 0)  # Yellow
POINT_MARKER_COLOR = (255, 0, 255)  # Magenta
LANDMARK_RADIUS = 5
SKELETON_THICKNESS = 5

MANO_FINGERTIP_INDICES = [4, 8, 12, 16, 20]
MANO_SKELETON_PAIRS = [
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 4],
    [0, 5],
    [5, 6],
    [6, 7],
    [7, 8],
    [9, 10],
    [10, 11],
    [11, 12],
    [13, 14],
    [14, 15],
    [15, 16],
    [0, 17],
    [17, 18],
    [18, 19],
    [19, 20],
    [5, 9],
    [9, 13],
    [13, 17],
]

# Short display names for loss columns
LOSS_SHORT_NAMES = {
    "landmarks_3d": "lmk_3d",
}

EPOCH_COL_WIDTH = 14
LOSS_COL_WIDTH = 10


def render_loss_chart(lambdas, rows):
    """Render a full Rich Table: heavy-head border, \u03bb row above header separator, all accumulated rows below."""
    table = Table(
        box=box.HEAVY_HEAD,
        show_header=True,
        header_style="bold cyan",
        pad_edge=False,
    )
    # Epoch column is left-justified so data rows (step right-just, phase left-just)
    # align across TRAIN/VAL. Rich strips trailing whitespace under center-justify, so
    # we pre-center the header label manually.
    table.add_column(
        f"{'epoch':^{EPOCH_COL_WIDTH}}",
        width=EPOCH_COL_WIDTH,
        justify="left",
        no_wrap=True,
    )
    table.add_column("total", width=LOSS_COL_WIDTH, justify="center", no_wrap=True)
    for name in lambdas:
        short = LOSS_SHORT_NAMES.get(name, name)
        table.add_column(short, width=LOSS_COL_WIDTH, justify="center", no_wrap=True)
    lam_cells = ["", ""] + [f"[magenta]\u03bb={lam:g}[/]" for lam in lambdas.values()]
    table.add_row(*lam_cells, end_section=True)
    for step, phase, losses in rows:
        phase_color = {"TRAIN": "yellow", "VAL": "green"}.get(phase, "white")
        # Fixed-width 14-char cell: step right-justified in 8 chars, space, phase
        # left-justified in 5 chars. Keeps TRAIN/VAL rows aligned under each other.
        epoch_cell = f"{step:>8d} [bold {phase_color}]{phase:<5s}[/]"
        cells = [epoch_cell, f"[bold]{losses['total']:.4f}[/]"]
        for name in lambdas:
            cells.append(f"{losses.get(name, 0.0):.5f}")
        table.add_row(*cells)
    return table


def _bgr(rgb: Tuple[int, int, int]) -> Tuple[int, int, int]:
    return (rgb[2], rgb[1], rgb[0])


def draw_point_marker(
    image: np.ndarray,
    u: int,
    v: int,
    color: Tuple[int, int, int] = POINT_MARKER_COLOR,
    radius: int = 4,
) -> np.ndarray:
    """Draw two concentric filled circles at (u, v): colored inner, white halo."""
    c = _bgr(color)
    w = (255, 255, 255)
    halo = max(1, radius // 3)
    cv2.circle(image, (u, v), radius + halo, w, -1, cv2.LINE_AA)
    cv2.circle(image, (u, v), radius, c, -1, cv2.LINE_AA)
    return image


def draw_mask_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int] = MASK_COLOR,
    alpha: float = 0.3,
    border_color: Tuple[int, int, int] = MASK_BORDER_COLOR,
    border_thickness: int = 4,
) -> np.ndarray:
    """Draw semi-transparent mask overlay with border."""
    result = image.copy()
    overlay = image.copy()
    mask_bool = mask > 0
    overlay[mask_bool] = _bgr(color)
    cv2.addWeighted(overlay, alpha, result, 1 - alpha, 0, result)

    if border_thickness > 0:
        mask_uint8 = (mask > 0).astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if contours:
            cv2.drawContours(
                result, contours, -1, _bgr(border_color), border_thickness, cv2.LINE_AA
            )
    return result


def draw_mano_mesh(
    image: np.ndarray,
    mesh_vertices_2d: np.ndarray,
    mesh_vertices_camera: np.ndarray,
    mesh_faces: np.ndarray,
    landmarks_2d: Optional[np.ndarray] = None,
    color: Tuple[int, int, int] = MANO_MESH_COLOR,
    alpha: float = 0.60,
    border_color: Tuple[int, int, int] = MANO_BORDER_COLOR,
    border_thickness: int = 0,
    skeleton_color: Optional[Tuple[int, int, int]] = None,
    keypoint_color: Optional[Tuple[int, int, int]] = None,
    skeleton_thickness: int = SKELETON_THICKNESS,
    landmark_radius: int = LANDMARK_RADIUS,
) -> np.ndarray:
    """Draw shaded MANO mesh with depth sorting and optional skeleton."""
    color_bgr = _bgr(color)
    v0 = mesh_vertices_camera[mesh_faces[:, 0]]
    v1 = mesh_vertices_camera[mesh_faces[:, 1]]
    v2 = mesh_vertices_camera[mesh_faces[:, 2]]

    centroids_z = (v0[:, 2] + v1[:, 2] + v2[:, 2]) / 3.0
    order = np.argsort(-centroids_z)

    edge1 = v1 - v0
    edge2 = v2 - v0
    normals = np.cross(edge1, edge2)
    norm = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / (norm + 1e-6)

    light_dir = np.array([0, 0, -1])
    shading = np.sum(normals * light_dir, axis=1)
    shading = 0.5 + 0.5 * np.clip(shading, 0.0, 1.0)

    overlay = image.copy()
    h, w = image.shape[:2]

    for i in order:
        face = mesh_faces[i]
        pts = mesh_vertices_2d[face].astype(np.int32)
        if not (
            np.all(pts[:, 0] >= 0)
            and np.all(pts[:, 0] < w)
            and np.all(pts[:, 1] >= 0)
            and np.all(pts[:, 1] < h)
        ):
            continue
        s = shading[i]
        sc = (int(color_bgr[0] * s), int(color_bgr[1] * s), int(color_bgr[2] * s))
        cv2.fillConvexPoly(overlay, pts, sc, cv2.LINE_AA)

    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

    if border_thickness > 0:
        mask = np.zeros((h, w), dtype=np.uint8)
        for face in mesh_faces:
            pts_float = mesh_vertices_2d[face]
            if np.any(np.isnan(pts_float)):
                continue
            pts = pts_float.astype(np.int32)
            cv2.fillConvexPoly(mask, pts, 255, cv2.LINE_AA)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cv2.drawContours(
                image, contours, -1, _bgr(border_color), border_thickness, cv2.LINE_AA
            )

    if landmarks_2d is not None:
        skel_c = _bgr(
            skeleton_color if skeleton_color is not None else MANO_SKELETON_COLOR
        )
        kp_c = _bgr(
            keypoint_color if keypoint_color is not None else MANO_KEYPOINT_COLOR
        )
        ft_c = _bgr(MANO_FINGERTIP_COLOR)
        for i, j in MANO_SKELETON_PAIRS:
            pt1 = landmarks_2d[i].astype(int)
            pt2 = landmarks_2d[j].astype(int)
            if (
                0 <= pt1[0] < w
                and 0 <= pt1[1] < h
                and 0 <= pt2[0] < w
                and 0 <= pt2[1] < h
            ):
                cv2.line(image, tuple(pt1), tuple(pt2), skel_c, skeleton_thickness)
        for i, pt in enumerate(landmarks_2d):
            x, y = int(pt[0]), int(pt[1])
            if 0 <= x < w and 0 <= y < h:
                c = ft_c if i in MANO_FINGERTIP_INDICES else kp_c
                if i in MANO_FINGERTIP_INDICES:
                    cv2.circle(image, (x, y), landmark_radius * 2, c, -1)
                else:
                    cv2.circle(image, (x, y), landmark_radius, c, -1)

    return image


def _rgb_str(color: Tuple[int, int, int]) -> str:
    """Convert RGB tuple to string."""
    return f"rgb({color[0]},{color[1]},{color[2]})"


def _skeleton_lines(joints):
    """Build x/y/z lists with None separators for disconnected bone segments."""
    xs, ys, zs = [], [], []
    for i, j in MANO_SKELETON_PAIRS:
        xs.extend([joints[i, 0], joints[j, 0], None])
        ys.extend([joints[i, 1], joints[j, 1], None])
        zs.extend([joints[i, 2], joints[j, 2], None])
    return xs, ys, zs


def _backproject_depth(
    depth_image: np.ndarray,
    depth_rgb: np.ndarray,
    camera_K: np.ndarray,
    max_depth: float = 5.0,
) -> Optional[tuple]:
    """Back-project depth to 3D point cloud with RGB colors.

    Caller must pass a camera_K whose resolution matches depth_image.
    """
    h, w = depth_image.shape[:2]
    fx, fy = camera_K[0, 0], camera_K[1, 1]
    cx, cy = camera_K[0, 2], camera_K[1, 2]

    z = depth_image.astype(np.float32) / 1000.0
    valid = (z > 0) & (z < max_depth)

    u, v = np.meshgrid(np.arange(w), np.arange(h))
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    points = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    colors = depth_rgb.reshape(-1, 3)
    valid_flat = valid.flatten()

    if valid_flat.sum() == 0:
        return None
    return points[valid_flat], colors[valid_flat]


def create_3d_plotly(
    pred_mano_params: torch.Tensor,
    mano_model: torch.nn.Module,
    mesh_faces: np.ndarray,
    object_points: Optional[np.ndarray] = None,
    depth_image: Optional[np.ndarray] = None,
    depth_rgb: Optional[np.ndarray] = None,
    camera_K: Optional[np.ndarray] = None,
    betas: Optional[torch.Tensor] = None,
    point_uv: Optional[np.ndarray] = None,
) -> go.Figure:
    """Create interactive Plotly 3D figure with the predicted triangle mesh and skeleton."""
    betas_batch = betas.unsqueeze(0) if betas is not None else None
    with torch.no_grad():
        pred_out = mano_model(pred_mano_params.unsqueeze(0), betas=betas_batch)
        pred_verts = pred_out["vertices"][0].cpu().numpy()
        pred_landmarks = pred_out["landmarks_3d"][0].cpu().numpy()
        pred_t = pred_out["t"][0].cpu().numpy()
        pred_verts_cam = pred_verts + pred_t
        pred_joints = pred_landmarks + pred_t

    # Back-project depth
    depth_points = None
    depth_colors = None
    if depth_image is not None and depth_rgb is not None and camera_K is not None:
        result = _backproject_depth(depth_image, depth_rgb, camera_K)
        if result is not None:
            depth_points, depth_colors = result

    traces = []
    # Pred hand (mesh + skeleton + joints grouped together)
    traces.append(
        go.Mesh3d(
            x=pred_verts_cam[:, 0],
            y=pred_verts_cam[:, 1],
            z=pred_verts_cam[:, 2],
            i=mesh_faces[:, 0],
            j=mesh_faces[:, 1],
            k=mesh_faces[:, 2],
            color=_rgb_str(PRED_MESH_COLOR),
            opacity=0.6,
            name="Pred",
            showlegend=True,
            legendgroup="pred",
        )
    )
    px, py, pz = _skeleton_lines(pred_joints)
    traces.append(
        go.Scatter3d(
            x=px,
            y=py,
            z=pz,
            mode="lines",
            line=dict(color=_rgb_str(PRED_SKELETON_COLOR), width=5),
            showlegend=False,
            legendgroup="pred",
        )
    )
    traces.append(
        go.Scatter3d(
            x=pred_joints[:, 0],
            y=pred_joints[:, 1],
            z=pred_joints[:, 2],
            mode="markers",
            marker=dict(color=_rgb_str(PRED_KEYPOINT_COLOR), size=2),
            showlegend=False,
            legendgroup="pred",
        )
    )
    # Object points
    if object_points is not None:
        traces.append(
            go.Scatter3d(
                x=object_points[:, 0],
                y=object_points[:, 1],
                z=object_points[:, 2],
                mode="markers",
                marker=dict(color="yellow", size=1, opacity=1.0),
                name="Object",
                visible="legendonly",
            )
        )
    # Depth point cloud
    if depth_points is not None:
        color_strs = [f"rgb({r},{g},{b})" for r, g, b in depth_colors]
        traces.append(
            go.Scatter3d(
                x=depth_points[:, 0],
                y=depth_points[:, 1],
                z=depth_points[:, 2],
                mode="markers",
                marker=dict(color=color_strs, size=2, opacity=1.0),
                name="Depth",
            )
        )

    # Click point — unproject at the exact (u,v) pixel using the depth image.
    # point_uv, camera_K, and depth_image are all in the same (224) frame.
    if point_uv is not None and camera_K is not None and depth_image is not None:
        K = camera_K.cpu().numpy() if isinstance(camera_K, torch.Tensor) else camera_K
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        dh, dw = depth_image.shape[:2]
        du = int(np.clip(round(float(point_uv[0])), 0, dw - 1))
        dv = int(np.clip(round(float(point_uv[1])), 0, dh - 1))
        d_m = float(depth_image[dv, du]) / 1000.0
        x = (du - cx) * d_m / fx
        y = (dv - cy) * d_m / fy
        traces.append(
            go.Scatter3d(
                x=[x],
                y=[y],
                z=[d_m],
                mode="markers",
                marker=dict(
                    color=_rgb_str(POINT_MARKER_COLOR),
                    size=12,
                    line=dict(color="white", width=2),
                ),
                name="Click",
            )
        )

    # Head-on view down +z (camera frame, OpenCV) — depth cloud reads like a flat
    # RGB image. -y is screen-up because +y is down in OpenCV.
    eye = dict(x=0, y=0, z=-1.5)
    up = dict(x=0, y=-1, z=0)

    fig = go.Figure(data=traces)
    fig.update_layout(
        scene=dict(
            aspectmode="data",
            dragmode="orbit",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            bgcolor="rgb(30,30,30)",
            camera=dict(eye=eye, up=up, center=dict(x=0, y=0, z=0)),
        ),
        paper_bgcolor="rgb(30,30,30)",
        margin=dict(l=0, r=10, t=0, b=10),
        legend=dict(
            font=dict(color="white"), x=1, y=0, xanchor="right", yanchor="bottom"
        ),
    )
    return fig


def create_prediction_visualization(
    rgb_original: np.ndarray,
    mask_original: np.ndarray,
    pred_mano_params: torch.Tensor,
    gt_mano_params: torch.Tensor,
    mano_model: torch.nn.Module,
    camera_K: torch.Tensor,
    mesh_faces: np.ndarray,
    loss_dict: Optional[dict] = None,
    point_uv: Optional[np.ndarray] = None,
    betas: Optional[torch.Tensor] = None,
) -> np.ndarray:
    """Create side-by-side pred vs GT visualization.

    Args:
        rgb_original: (H, W, 3) uint8 RGB (original size, not resized)
        mask_original: (H, W) uint8 object mask (original size)
        pred_mano_params: (99,) predicted MANO params
        gt_mano_params: (99,) ground truth MANO params
        mano_model: MANO instance
        camera_K: (3, 3) camera intrinsics for original-size image
        mesh_faces: (F, 3) int array of mesh face indices
        loss_dict: optional dict of {name: value} to overlay on the pred image
        point_uv: optional (2,) point coords at original image scale

    Returns:
        (H, W*2, 3) uint8 numpy array in RGB
    """
    img_pred = cv2.cvtColor(rgb_original.copy(), cv2.COLOR_RGB2BGR)
    img_gt = cv2.cvtColor(rgb_original.copy(), cv2.COLOR_RGB2BGR)

    # Scale all draw params to image height (1440 = reference resolution).
    scale = img_pred.shape[0] / 1440
    mask_border_thickness = max(1, round(3 * scale))
    skel_thickness = max(1, round(4 * scale))
    lm_radius = max(1, round(4 * scale))
    pt_radius = max(3, img_pred.shape[0] // 100)

    mask = (
        mask_original
        if isinstance(mask_original, np.ndarray)
        else mask_original.cpu().numpy()
    )
    mask = (mask > 0).astype(np.uint8) * 255

    img_pred = draw_mask_overlay(img_pred, mask, border_thickness=mask_border_thickness)
    img_gt = draw_mask_overlay(img_gt, mask, border_thickness=mask_border_thickness)

    if point_uv is not None:
        u, v = int(point_uv[0]), int(point_uv[1])
        draw_point_marker(img_pred, u, v, radius=pt_radius)
        draw_point_marker(img_gt, u, v, radius=pt_radius)

    K = camera_K.cpu().numpy() if isinstance(camera_K, torch.Tensor) else camera_K

    # Render pred
    betas_batch = betas.unsqueeze(0) if betas is not None else None
    with torch.no_grad():
        pred_out = mano_model(pred_mano_params.unsqueeze(0), betas=betas_batch)
        pred_verts = pred_out["vertices"][0].cpu().numpy()
        pred_landmarks = pred_out["landmarks_3d"][0].cpu().numpy()
        pred_t = pred_out["t"][0].cpu().numpy()

    pred_verts_cam = pred_verts + pred_t
    pred_landmarks_3d = pred_landmarks + pred_t
    img_pred = draw_mano_mesh(
        img_pred,
        project_3d_to_2d(pred_verts_cam, K),
        pred_verts_cam,
        mesh_faces,
        landmarks_2d=project_3d_to_2d(pred_landmarks_3d, K),
        color=PRED_MESH_COLOR,
        skeleton_color=PRED_SKELETON_COLOR,
        keypoint_color=PRED_KEYPOINT_COLOR,
        skeleton_thickness=skel_thickness,
        landmark_radius=lm_radius,
    )

    # Render GT
    with torch.no_grad():
        gt_out = mano_model(gt_mano_params.unsqueeze(0), betas=betas_batch)
        gt_verts = gt_out["vertices"][0].cpu().numpy()
        gt_landmarks = gt_out["landmarks_3d"][0].cpu().numpy()
        gt_t = gt_out["t"][0].cpu().numpy()

    gt_verts_cam = gt_verts + gt_t
    gt_landmarks_3d = gt_landmarks + gt_t
    img_gt = draw_mano_mesh(
        img_gt,
        project_3d_to_2d(gt_verts_cam, K),
        gt_verts_cam,
        mesh_faces,
        landmarks_2d=project_3d_to_2d(gt_landmarks_3d, K),
        skeleton_thickness=skel_thickness,
        landmark_radius=lm_radius,
    )

    h, w = img_pred.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5 * scale
    thickness = max(1, round(3 * scale))
    line_gap = max(1, round(56 * scale))
    title_scale = 3.75 * scale
    title_thickness = max(1, round(8 * scale))
    header_scale = 2.25 * scale
    header_thickness = max(1, round(5 * scale))
    sep_thickness = max(1, round(2 * scale))
    m15 = max(1, round(11 * scale))
    m20 = max(1, round(15 * scale))
    m10 = max(1, round(8 * scale))

    # Centered title labels
    for img, label in [(img_pred, "Pred"), (img_gt, "GT")]:
        (tw, th), _ = cv2.getTextSize(label, font, title_scale, title_thickness)
        cv2.putText(
            img,
            label,
            ((w - tw) // 2, th + m15),
            font,
            title_scale,
            (255, 255, 255),
            title_thickness,
        )

    # Wrist translation stats — bottom-left of pred image
    pred_t_mm = pred_t * 1000
    gt_t_mm = gt_t * 1000
    diff_mm = pred_t_mm - gt_t_mm
    dist_mm = np.linalg.norm(diff_mm)
    t_lines = [
        f"pred: ({pred_t_mm[0]:.0f}, {pred_t_mm[1]:.0f}, {pred_t_mm[2]:.0f}) mm",
        f"gt:   ({gt_t_mm[0]:.0f}, {gt_t_mm[1]:.0f}, {gt_t_mm[2]:.0f}) mm",
        f"diff: ({diff_mm[0]:.0f}, {diff_mm[1]:.0f}, {diff_mm[2]:.0f}) mm",
        f"dist: {dist_mm:.1f} mm",
    ]
    for i, line in enumerate(reversed(t_lines)):
        y_pos = h - m15 - i * line_gap
        cv2.putText(
            img_pred, line, (m10, y_pos), font, font_scale, (255, 255, 255), thickness
        )
    # Separator line + "Wrist" header
    sep_y = h - m15 - len(t_lines) * line_gap + m20
    (hw, _), _ = cv2.getTextSize("Wrist", font, header_scale, header_thickness)
    cv2.line(img_pred, (m10, sep_y), (hw + m10, sep_y), (255, 255, 255), sep_thickness)
    header_y = sep_y - m15
    cv2.putText(
        img_pred,
        "Wrist",
        (m10, header_y),
        font,
        header_scale,
        (255, 255, 255),
        header_thickness,
    )

    # Loss values — bottom-right of GT image
    if loss_dict:
        loss_lines = [f"{name}: {val:.4f}" for name, val in loss_dict.items()]
        for i, line in enumerate(reversed(loss_lines)):
            (tw, _), _ = cv2.getTextSize(line, font, font_scale, thickness)
            y_pos = h - m15 - i * line_gap
            cv2.putText(
                img_gt,
                line,
                (w - tw - m10, y_pos),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
            )
        # Separator line + "Loss" header
        sep_y = h - m15 - len(loss_lines) * line_gap + m20
        (lw, _), _ = cv2.getTextSize("Loss", font, header_scale, header_thickness)
        cv2.line(
            img_gt,
            (w - lw - m10, sep_y),
            (w - m10, sep_y),
            (255, 255, 255),
            sep_thickness,
        )
        header_y = sep_y - m15
        cv2.putText(
            img_gt,
            "Loss",
            (w - lw - m10, header_y),
            font,
            header_scale,
            (255, 255, 255),
            header_thickness,
        )

    combined_bgr = np.concatenate([img_pred, img_gt], axis=1)
    combined_rgb = cv2.cvtColor(combined_bgr, cv2.COLOR_BGR2RGB)
    return combined_rgb
