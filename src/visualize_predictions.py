"""Visualize predicted grasps in 3D using Viser.

Displays RGB with the predicted hand (green).
"""

import pickle
import random
import re
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import tyro
import viser
from rich.console import Console

from .utils.data_keys import MANO_RIGHT_MESH_FACES_FILE
from .utils.pcl_utils import backproject_to_pcl
from .utils.viser_utils import (
    add_hand_keypoints,
    add_hand_skeleton,
    add_mano_mesh,
    add_wrist_frame,
    backproject_depth_to_point_cloud,
    image_to_data_url,
)
from .utils.visualization_utils import (
    POINT_MARKER_COLOR,
    PRED_KEYPOINT_COLOR,
    PRED_MESH_COLOR,
    PRED_SKELETON_COLOR,
    draw_point_marker,
)

console = Console()

# Crop-mode constants (formerly the "Crop Radius (m)" / "Background Dim" sliders)
CROP_RADIUS_M = 0.3
CROP_BG_DIM = 0.7


def visualize(
    dataset_path: Path,
    port: int = 8080,
    max_depth: float = 5.0,
    sample_name: Optional[str] = None,
    num_samples: int = 100,
    share: bool = False,
) -> None:
    """Visualize grasp predictions.

    Args:
        dataset_path: Path to prepared dataset folder.
        port: Web server port.
        max_depth: Maximum depth in meters for point cloud filtering.
        sample_name: Sample to visualize or path to .txt file with one stem per line.
        num_samples: Number of random samples to load. Ignored if sample_name is set.
        share: Generate shareable Viser URL.
    """
    server = viser.ViserServer(port=port, verbose=False)
    server.gui.configure_theme(dark_mode=True)
    if share:
        server.request_share_url()

    grasp_dir = dataset_path
    pred_dir = dataset_path / "grasp_pred"
    image_original_dir = dataset_path / "image_original"
    has_preds = pred_dir.exists()

    # Collect sample names from grasp_pred/ directory
    if sample_name is not None and sample_name.endswith(".txt"):
        sample_names = Path(sample_name).read_text().strip().splitlines()
    elif sample_name is not None:
        sample_names = [sample_name]
    else:
        # Recursive so nested scene/object layouts (grasp_pred/<scene>/<stem>.pkl)
        # written by app --save-pred are found; the name is the path relative to
        # grasp_pred, which resolves back via grasp_dir/pred_dir / f"{name}.pkl".
        all_samples = sorted(
            p.relative_to(pred_dir).with_suffix("").as_posix()
            for p in pred_dir.rglob("*.pkl")
        )
        if len(all_samples) > num_samples:
            sample_names = sorted(random.sample(all_samples, num_samples))
        else:
            sample_names = all_samples

    state = {"current_idx": 0, "scene_handles": []}
    server.scene.set_up_direction("-y")

    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        client.camera.up_direction = (0, -1, 0)
        client.camera.position = (0, 0, -0.1)
        client.camera.look_at = (0, 0, 1)

    # GUI — Hand
    with server.gui.add_folder("Hand"):
        gui_show_grasp = server.gui.add_checkbox("Show Pred", True)
        gui_show_wrist_frame = server.gui.add_checkbox("Wrist Frame", False)
        gui_mesh_opacity = server.gui.add_slider(
            "Mesh Opacity", min=0.0, max=1.0, step=0.05, initial_value=0.6
        )
        gui_show_point_condition = server.gui.add_checkbox("Show Point Condition", True)

    # GUI — Image
    with server.gui.add_folder("Image"):
        gui_dropdown = server.gui.add_dropdown(
            "Sample", options=sample_names, initial_value=sample_names[0]
        )
        gui_nav = server.gui.add_button_group("Nav", ("← Prev", "Next →"))
        gui_image = server.gui.add_html("")

    # GUI — Visibility
    with server.gui.add_folder("Visibility"):
        gui_show_camera = server.gui.add_checkbox("Camera", False)
        gui_show_camera_frame = server.gui.add_checkbox("Camera Frame", False)
        gui_depth_point_size = server.gui.add_slider(
            "Depth Point Size", min=0.0, max=0.01, step=0.0001, initial_value=0.004
        )
        gui_max_depth = server.gui.add_slider(
            "Max Depth (m)",
            min=0.5,
            max=20.0,
            step=0.1,
            initial_value=min(max_depth, 20.0),
        )
        gui_crop_only = server.gui.add_checkbox("Crop PCL", False)

    vis = {
        "frustum": None,
        "camera_frame": None,
        "pred_wrist_frame": None,
        "pred_mesh": None,
        "depth_point_cloud": None,
        "depth_colors": None,
        "crop_point_cloud": None,
        "crop_center": None,
        "crop_depth_m": None,
        "crop_rgb": None,
        "crop_K": None,
        "point_handle": None,
        "rgb_image": None,
        "point_uv_norm": None,
        "fov": None,
        "aspect": None,
        "pred_handles": [],
    }

    def update_sidebar_image():
        img = vis["rgb_image"]
        if img is None:
            return
        point_uv_norm = vis["point_uv_norm"]
        if point_uv_norm is not None and gui_show_point_condition.value:
            h_disp, w_disp = img.shape[:2]
            u_px = int(np.clip(point_uv_norm[0] * w_disp, 0, w_disp - 1))
            v_px = int(np.clip(point_uv_norm[1] * h_disp, 0, h_disp - 1))
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img_bgr = draw_point_marker(
                img_bgr, u_px, v_px, radius=max(3, h_disp // 100)
            )
            img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        gui_image.content = (
            f'<img src="{image_to_data_url(img)}" style="width:100%;display:block;">'
        )

    def load_grasp_pkl(path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        return data

    def update_scene(name: str) -> None:
        for h in state["scene_handles"]:
            try:
                h.remove()
            except Exception:
                pass
        state["scene_handles"] = []
        vis["pred_handles"] = []
        vis["point_handle"] = None
        vis["depth_colors"] = None
        if vis["crop_point_cloud"] is not None:
            try:
                vis["crop_point_cloud"].remove()
            except Exception:
                pass
        vis["crop_point_cloud"] = None
        vis["crop_center"] = None

        console.print(f"[cyan]Visualizing: {name}[/cyan]")

        # Load the source pkl for image/camera/depth (the app appends a _<datetime>
        # suffix per click, so strip it to find it); fall back to the pred pkl.
        src_stem = re.sub(r"_\d{8}_\d{6}_\d{3}$", "", name)
        src_path = grasp_dir / f"{src_stem}.pkl"
        pred_path = pred_dir / f"{name}.pkl"
        pred_data = (
            load_grasp_pkl(pred_path) if has_preds and pred_path.exists() else None
        )
        if src_path.exists():
            data = load_grasp_pkl(src_path)
        elif pred_data is not None:
            data = pred_data
        else:
            raise FileNotFoundError(f"No input pkl or grasp_pred entry for {name}")

        pred_grasp = pred_data["grasp"] if pred_data is not None else None

        camera_small = data["camera"]
        K_small = (
            camera_small["K"] if isinstance(camera_small, dict) else camera_small.K
        )
        cam_w_small = (
            camera_small["width"]
            if isinstance(camera_small, dict)
            else camera_small.width
        )
        cam_h_small = (
            camera_small["height"]
            if isinstance(camera_small, dict)
            else camera_small.height
        )

        # Decode embedded 224x224 image (always present in new format); used for
        # depth backprojection coloring which needs to match the depth resolution.
        rgb_224 = cv2.imdecode(np.frombuffer(data["image"], np.uint8), cv2.IMREAD_COLOR)
        rgb_224 = cv2.cvtColor(rgb_224, cv2.COLOR_BGR2RGB)

        # Prefer the high-res FOV-cropped image for sidebar + frustum when present.
        # Some stems are {capture}_{NNN}; fall back to the capture stem.
        image_stem = name
        if not (image_original_dir / f"{image_stem}.jpg").exists():
            parts = name.rsplit("_", 1)
            if len(parts) == 2 and parts[1].isdigit():
                image_stem = parts[0]
        image_original_path = image_original_dir / f"{image_stem}.jpg"
        if image_original_path.exists():
            rgb_high = cv2.cvtColor(
                cv2.imread(str(image_original_path)), cv2.COLOR_BGR2RGB
            )
        else:
            rgb_high = rgb_224.copy()

        # Optional grasp point: app.py / inference.py encode 8 bytes of float32
        # (u_norm, v_norm) into the pred's object_mask. Prefer the pred file so we
        # get the real point that was used for prediction; anything else (e.g. a
        # PNG mask) is ignored here.
        point_uv_norm = None
        point_bytes = (pred_data or {}).get("object_mask", b"") or data.get(
            "object_mask", b""
        )
        if point_bytes and len(point_bytes) == 8:
            point_uv_norm = np.frombuffer(point_bytes, dtype=np.float32).copy()

        vis["point_uv_norm"] = point_uv_norm

        # Load mesh faces from the prediction, fall back to default MANO faces
        mesh_faces = pred_grasp.get("mesh_faces") if pred_grasp is not None else None
        if mesh_faces is None:
            mesh_faces = np.load(MANO_RIGHT_MESH_FACES_FILE)

        vis["rgb_image"] = rgb_high
        vis["fov"] = 2 * np.arctan2(cam_w_small / 2, K_small[0, 0])
        vis["aspect"] = cam_w_small / cam_h_small

        # Camera frame
        camera_frame = server.scene.add_frame(
            "/camera_frame",
            axes_length=0.12,
            axes_radius=0.004,
        )
        camera_frame.visible = gui_show_camera_frame.value and gui_show_camera.value
        state["scene_handles"].append(camera_frame)
        vis["camera_frame"] = camera_frame

        # Camera frustum
        frustum = server.scene.add_camera_frustum(
            "/camera_frustum",
            fov=vis["fov"],
            aspect=vis["aspect"],
            scale=0.1,
            image=rgb_high,
            format="jpeg",
        )
        frustum.visible = gui_show_camera.value
        state["scene_handles"].append(frustum)
        vis["frustum"] = frustum

        # Depth point cloud — embedded 224 depth back-projected with 224-scale K.
        depth_pc_handle = None
        if data.get("depth"):
            depth_image = cv2.imdecode(
                np.frombuffer(data["depth"], np.uint8), cv2.IMREAD_UNCHANGED
            )
            result = backproject_depth_to_point_cloud(
                depth_image,
                rgb_224,
                K_small,
                cam_w_small,
                cam_h_small,
                gui_max_depth.value,
                mask=None,
            )
            if result is not None:
                points, colors = result
                vis["depth_colors"] = colors
                depth_pc_handle = server.scene.add_point_cloud(
                    "/depth_points",
                    points=points,
                    colors=colors,
                    point_size=gui_depth_point_size.value,
                    point_shape="rounded",
                )
                depth_pc_handle.visible = gui_depth_point_size.value > 0
                state["scene_handles"].append(depth_pc_handle)

            # 3D grasp point: back-project (u_224, v_224, depth) with K_small
            if point_uv_norm is not None:
                u_224 = int(np.clip(point_uv_norm[0] * cam_w_small, 0, cam_w_small - 1))
                v_224 = int(np.clip(point_uv_norm[1] * cam_h_small, 0, cam_h_small - 1))
                z_mm = int(depth_image[v_224, u_224])
                if z_mm > 0:
                    z = z_mm / 1000.0
                    fx, fy = K_small[0, 0], K_small[1, 1]
                    cx, cy = K_small[0, 2], K_small[1, 2]
                    x = (u_224 - cx) * z / fx
                    y = (v_224 - cy) * z / fy
                    point_handle = server.scene.add_icosphere(
                        "/pred/point_condition",
                        radius=0.01,
                        color=POINT_MARKER_COLOR,
                        position=(x, y, z),
                    )
                    point_handle.visible = (
                        gui_show_grasp.value and gui_show_point_condition.value
                    )
                    state["scene_handles"].append(point_handle)
                    vis["point_handle"] = point_handle

                    # Crop inputs: sphere around the grasp point
                    vis["crop_center"] = np.array([x, y, z], dtype=np.float32)
                    vis["crop_depth_m"] = depth_image.astype(np.float32) / 1000.0
                    vis["crop_rgb"] = rgb_224
                    vis["crop_K"] = K_small
        vis["depth_point_cloud"] = depth_pc_handle

        # Pred hand (green)
        if pred_grasp is not None:
            pred_wrist = add_wrist_frame(
                server,
                "/pred/wrist_frame",
                pred_grasp["T_camera_wrist"],
                visible=gui_show_wrist_frame.value and gui_show_grasp.value,
            )
            state["scene_handles"].append(pred_wrist)
            vis["pred_wrist_frame"] = pred_wrist
            vis["pred_handles"].append(pred_wrist)

            pred_skel = add_hand_skeleton(
                server,
                "/pred/skeleton",
                pred_grasp["landmarks_3d"],
                color=PRED_SKELETON_COLOR,
                visible=gui_show_grasp.value,
            )
            state["scene_handles"].append(pred_skel)
            vis["pred_handles"].append(pred_skel)

            pred_kp = add_hand_keypoints(
                server,
                "/pred/keypoints",
                pred_grasp["landmarks_3d"],
                color=PRED_KEYPOINT_COLOR,
                visible=gui_show_grasp.value,
            )
            state["scene_handles"].append(pred_kp)
            vis["pred_handles"].append(pred_kp)

            pred_mesh = add_mano_mesh(
                server,
                "/pred/mesh",
                pred_grasp["mesh_vertices"],
                mesh_faces,
                color=PRED_MESH_COLOR,
                opacity=gui_mesh_opacity.value,
                visible=gui_show_grasp.value,
            )
            state["scene_handles"].append(pred_mesh)
            vis["pred_mesh"] = pred_mesh
            vis["pred_handles"].append(pred_mesh)
        else:
            vis["pred_wrist_frame"] = None
            vis["pred_mesh"] = None
        update_sidebar_image()
        rebuild_crop()

    def apply_background_dim() -> None:
        """Set the full depth cloud's color/visibility.

        Outside crop mode it shows at full brightness. In crop mode it acts as a
        dimmed backdrop behind the bright crop cloud: colors are scaled by
        (1 - CROP_BG_DIM); at full dim the cloud is hidden entirely.
        """
        h = vis["depth_point_cloud"]
        if not h or vis["depth_colors"] is None:
            return
        base_visible = gui_depth_point_size.value > 0
        if gui_crop_only.value:
            bright = 1.0 - CROP_BG_DIM
            h.colors = (vis["depth_colors"].astype(np.float32) * bright).astype(
                np.uint8
            )
            h.visible = base_visible and bright > 0.0
        else:
            h.colors = vis["depth_colors"]
            h.visible = base_visible

    def rebuild_crop() -> None:
        """Draw the crop-radius PCL around the grasp point.

        When enabled, shows points within crop_radius of the grasp point at full
        brightness over a dimmed full cloud (see apply_background_dim); otherwise
        removes the crop cloud and restores the full cloud.
        """
        if vis["crop_point_cloud"] is not None:
            try:
                vis["crop_point_cloud"].remove()
            except Exception:
                pass
            vis["crop_point_cloud"] = None

        if gui_crop_only.value and vis["crop_center"] is not None:
            xyz, colors = backproject_to_pcl(
                vis["crop_depth_m"],
                vis["crop_rgb"],
                vis["crop_K"],
                max_depth=gui_max_depth.value,
                center=vis["crop_center"],
                crop_radius=CROP_RADIUS_M,
            )
            if xyz.shape[0] > 0:
                vis["crop_point_cloud"] = server.scene.add_point_cloud(
                    "/crop_points",
                    points=xyz.astype(np.float32),
                    colors=colors,
                    point_size=max(gui_depth_point_size.value, 0.001) * 1.5,
                    point_shape="rounded",
                )

        apply_background_dim()

    # Navigation
    def on_dropdown(_):
        state["current_idx"] = sample_names.index(gui_dropdown.value)
        update_scene(gui_dropdown.value)

    def on_nav(event: viser.GuiEvent):
        if event.target.value == "← Prev":
            state["current_idx"] = (state["current_idx"] - 1) % len(sample_names)
        else:
            state["current_idx"] = (state["current_idx"] + 1) % len(sample_names)
        gui_dropdown.value = sample_names[state["current_idx"]]

    gui_dropdown.on_update(on_dropdown)
    gui_nav.on_click(on_nav)

    # Visibility callbacks
    def on_show_rgb(_):
        if vis["frustum"]:
            vis["frustum"].visible = gui_show_camera.value
        if vis["camera_frame"]:
            vis["camera_frame"].visible = (
                gui_show_camera_frame.value and gui_show_camera.value
            )

    def on_mesh_opacity(_):
        if vis["pred_mesh"]:
            vis["pred_mesh"].opacity = gui_mesh_opacity.value

    def on_depth_point_size(_):
        if vis["depth_point_cloud"]:
            vis["depth_point_cloud"].point_size = gui_depth_point_size.value
        if vis["crop_point_cloud"]:
            vis["crop_point_cloud"].point_size = (
                max(gui_depth_point_size.value, 0.001) * 1.5
            )
        apply_background_dim()

    def on_crop(_):
        rebuild_crop()

    def on_show_camera_frame(_):
        if vis["camera_frame"]:
            vis["camera_frame"].visible = (
                gui_show_camera_frame.value and gui_show_camera.value
            )

    def on_show_wrist(_):
        show = gui_show_wrist_frame.value
        if vis["pred_wrist_frame"]:
            vis["pred_wrist_frame"].visible = show and gui_show_grasp.value

    def on_show_grasp(_):
        show = gui_show_grasp.value
        for h in vis["pred_handles"]:
            h.visible = show
        if vis["pred_wrist_frame"]:
            vis["pred_wrist_frame"].visible = show and gui_show_wrist_frame.value
        if vis["point_handle"]:
            vis["point_handle"].visible = show and gui_show_point_condition.value

    def on_show_point_condition(_):
        if vis["point_handle"]:
            vis["point_handle"].visible = (
                gui_show_grasp.value and gui_show_point_condition.value
            )
        update_sidebar_image()

    gui_show_camera.on_update(on_show_rgb)
    gui_mesh_opacity.on_update(on_mesh_opacity)
    gui_depth_point_size.on_update(on_depth_point_size)
    gui_show_camera_frame.on_update(on_show_camera_frame)
    gui_show_wrist_frame.on_update(on_show_wrist)
    gui_show_grasp.on_update(on_show_grasp)
    gui_show_point_condition.on_update(on_show_point_condition)
    gui_crop_only.on_update(on_crop)

    update_scene(sample_names[0])

    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    tyro.cli(visualize)
