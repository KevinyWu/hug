"""Interactive grasp prediction app.

Loads model + dataset, shows scene in Viser.
Click on 2D image in sidebar to run inference at that point.
"""

import pickle
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import tyro
import viser
from rich.console import Console

from .dataloader.data_classes import CameraIntrinsics, Grasp, GraspData
from .dataloader.grasp_dataset import GraspDataset
from .inference import load_model, resolve_checkpoint_path
from .models.mano import mano_params_to_animation, mano_params_to_grasp_dict
from .utils.pcl_utils import depth_to_pcl_tensors, pixel_to_xyz
from .utils.viser_utils import (
    PredictionStore,
    add_hand_keypoints,
    add_hand_skeleton,
    add_mano_mesh,
    add_wrist_frame,
    backproject_depth_to_point_cloud,
    get_skeleton_from_landmarks,
    make_clickable_image_html,
    make_hidden_bridge_bootstrap_html,
)
from .utils.visualization_utils import (
    POINT_MARKER_COLOR,
    draw_point_marker,
)

console = Console()


def app(
    checkpoint_path: Path,
    dataset_path: Path,
    port: int = 8080,
    use_ema: bool = True,
    sampling_steps: int = 50,
    sample_name: Optional[str] = None,
    num_samples: int = 100,
    max_depth: float = 5.0,
    share: bool = False,
    save_pred: bool = False,
) -> None:
    """Launch an interactive Viser app: click a pixel to predict a grasp there.

    Args:
        checkpoint_path: Checkpoint file or run directory (prefers the slim bf16 file).
        dataset_path: Folder of .pkl samples (searched recursively; grasp_pred/ skipped).
        port: Web server port.
        use_ema: Load EMA weights when present in the checkpoint.
        sampling_steps: Euler ODE steps for flow-matching sampling.
        sample_name: Single stem or path to a .txt of stems; overrides num_samples.
        num_samples: Number of random samples to load when sample_name is unset.
        max_depth: Max depth in meters for point-cloud display.
        share: Generate a shareable Viser URL.
        save_pred: Save each click to grasp_pred/<name>_<datetime_ms>.pkl (a new
            file per click) for later viewing with visualize_predictions.
    """
    checkpoint_path = resolve_checkpoint_path(checkpoint_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(checkpoint_path, use_ema, device)
    use_rgb = getattr(model, "use_rgb", True)
    use_depth = getattr(model, "use_depth", False)
    pcl_use_rgb = getattr(model, "pcl_use_rgb", False)

    # Resolve sample names. Pkls are found recursively under dataset_path
    # (grasp_pred/ excluded), so flat and nested scene/object layouts both work;
    # the sample name is the path relative to dataset_path (e.g. "large_1" or
    # "large_1/00000064"), which get_inference_data resolves via dataset_path /
    # f"{name}.pkl".
    all_stems = [
        p.relative_to(dataset_path).with_suffix("").as_posix()
        for p in GraspDataset.find_pkls(dataset_path)
    ]
    if sample_name is not None:
        if sample_name.endswith(".txt"):
            sample_names = Path(sample_name).read_text().strip().splitlines()
        else:
            sample_names = [sample_name]
    else:
        if len(all_stems) > num_samples:
            sample_names = sorted(random.sample(all_stems, num_samples))
        else:
            sample_names = all_stems

    dataset = GraspDataset(
        str(dataset_path),
        split="val",
        use_rgb=use_rgb,
        use_depth=use_depth,
    )

    # Viser setup
    server = viser.ViserServer(port=port, verbose=False)
    server.gui.configure_theme(dark_mode=True)
    if share:
        server.request_share_url()
    server.scene.set_up_direction("-y")

    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        client.camera.up_direction = (0, -1, 0)
        client.camera.position = (0, 0, -0.1)
        client.camera.look_at = (0, 0, 1)

    state = {"current_idx": 0, "scene_handles": []}
    pred_store = PredictionStore()

    # GUI — Hand
    with server.gui.add_folder("Hand"):
        gui_show_grasp = server.gui.add_checkbox("Show Pred", True)
        gui_show_wrist_frame = server.gui.add_checkbox("Wrist Frame", False)
        gui_mesh_opacity = server.gui.add_slider(
            "Mesh Opacity", min=0.0, max=1.0, step=0.05, initial_value=0.6
        )
        gui_animate = server.gui.add_checkbox("Animate Grasp", True)
        gui_anim_duration = server.gui.add_slider(
            "Anim Duration (s)", min=0.2, max=3.0, step=0.1, initial_value=1.0
        )
        gui_pre_offset = server.gui.add_slider(
            "Pre-grasp Offset (cm)", min=0.0, max=10.0, step=0.5, initial_value=5.0
        )
        gui_show_point_condition = server.gui.add_checkbox("Show Point Condition", True)
        gui_clear_btns = server.gui.add_button_group(
            "Clear", options=["Clear Last", "Clear All"]
        )

    # GUI — Image
    with server.gui.add_folder("Image"):
        gui_dropdown = server.gui.add_dropdown(
            "Sample", options=sample_names, initial_value=sample_names[0]
        )
        gui_nav = server.gui.add_button_group("Nav", ("← Prev", "Next →"))
        click_bridge_label = "__click_uv_bridge__"
        gui_click_bridge = server.gui.add_text(click_bridge_label, initial_value="")
        server.gui.add_html(make_hidden_bridge_bootstrap_html(click_bridge_label))
        gui_image = server.gui.add_html("")

    # GUI — Visibility
    with server.gui.add_folder("Visibility"):
        gui_show_rgb = server.gui.add_checkbox("Camera", False)
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

    vis = {
        "rgb_image": None,
        "width": None,
        "height": None,
        "camera_K": None,
        "mesh_faces": None,
        "depth_point_cloud": None,
        "camera_frustum": None,
        "camera_frame": None,
        "last_click_norm": None,
    }

    def _make_image_html(image_rgb: np.ndarray) -> str:
        return make_clickable_image_html(image_rgb, click_bridge_label)

    def add_pred_hand(pred_grasp, point_handle=None, frames=None):
        """Add predicted hand to 3D scene with rotating colors.

        When frames is a (verts_seq, joints_seq) tuple, the mesh/skeleton/keypoints
        are created at the pre-grasp (frame 0) and animated into the grasp before
        settling; otherwise they are placed statically at the predicted grasp.
        """
        mesh_faces = vis["mesh_faces"]
        show = gui_show_grasp.value
        color = pred_store.next_color()
        pred_name = f"pred_{int(time.time_ns())}"
        path = f"/predictions/{pred_name}"

        init_landmarks = (
            frames[1][0] if frames is not None else pred_grasp["landmarks_3d"]
        )
        init_verts = frames[0][0] if frames is not None else pred_grasp["mesh_vertices"]

        handles = []
        skel = add_hand_skeleton(
            server,
            f"{path}/skeleton",
            init_landmarks,
            color=color,
            visible=show,
        )
        handles.append(skel)
        kp = add_hand_keypoints(
            server,
            f"{path}/keypoints",
            init_landmarks,
            color=color,
            visible=show,
        )
        handles.append(kp)
        mesh = add_mano_mesh(
            server,
            f"{path}/mesh",
            init_verts,
            mesh_faces,
            color=color,
            opacity=gui_mesh_opacity.value,
            visible=show,
        )
        handles.append(mesh)
        wrist = add_wrist_frame(
            server,
            f"{path}/wrist_frame",
            pred_grasp["T_camera_wrist"],
            visible=show and gui_show_wrist_frame.value,
        )

        if frames is not None:
            verts_seq, joints_seq = frames
            dt = gui_anim_duration.value / len(verts_seq)
            for i in range(len(verts_seq)):
                mesh.vertices = verts_seq[i]
                kp.points = joints_seq[i]
                skel.points = get_skeleton_from_landmarks(joints_seq[i])
                wrist.position = joints_seq[i][0]
                time.sleep(dt)

        pred_store.add(
            pred_name, handles, wrist_handle=wrist, point_handle=point_handle
        )

    def update_gui_image():
        """Redraw 2D image, adding a click marker when point condition is on."""
        img_bgr = cv2.cvtColor(vis["rgb_image"].copy(), cv2.COLOR_RGB2BGR)
        click_norm = vis["last_click_norm"]
        if click_norm is not None and gui_show_point_condition.value:
            h, w = img_bgr.shape[:2]
            u_disp = int(np.clip(click_norm[0] * w, 0, w - 1))
            v_disp = int(np.clip(click_norm[1] * h, 0, h - 1))
            img_bgr = draw_point_marker(
                img_bgr,
                u_disp,
                v_disp,
                radius=max(3, h // 100),
            )
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        gui_image.content = _make_image_html(img_rgb)

    def save_prediction(
        name: str, pred_grasp: dict, u_224: float, v_224: float
    ) -> None:
        """Persist one prediction to grasp_pred/<name>_<datetime>.pkl (visualize_predictions format).

        Mirrors inference.py: reuses the source pkl's image/depth/camera, stores the
        predicted grasp, and encodes the normalized click (u, v) into object_mask. The
        path mirrors the input's (possibly nested) layout; each click writes a new
        timestamped file rather than overwriting.
        """
        orig_path = dataset_path / f"{name}.pkl"
        with open(orig_path, "rb") as f:
            orig = pickle.load(f)
        cam, cam_orig = orig["camera"], orig["camera_original"]
        point_norm = np.array([u_224 / 224.0, v_224 / 224.0], dtype=np.float32)
        grasp_data = GraspData(
            object_name=orig.get("object_name", ""),
            frame_index=orig.get("frame_index", 0),
            grasp_index=orig.get("grasp_index", 0),
            camera=CameraIntrinsics(**cam) if isinstance(cam, dict) else cam,
            camera_original=(
                CameraIntrinsics(**cam_orig) if isinstance(cam_orig, dict) else cam_orig
            ),
            grasp=Grasp(**pred_grasp),
            image=orig.get("image", b""),
            depth=orig.get("depth", b""),
            object_mask=point_norm.tobytes(),
        )
        now = time.time()
        timestamp = (
            time.strftime("%Y%m%d_%H%M%S", time.localtime(now))
            + f"_{int((now % 1) * 1000):03d}"
        )
        out_path = dataset_path / "grasp_pred" / f"{name}_{timestamp}.pkl"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "wb") as f:
            pickle.dump(asdict(grasp_data), f)
        console.print(f"[green]Saved pred -> {out_path}[/green]")

    def handle_click(u_norm: float, v_norm: float):
        """Handle click at normalized (0-1) coords, run inference."""
        u_224 = float(np.clip(u_norm * 224.0, 0, 223))
        v_224 = float(np.clip(v_norm * 224.0, 0, 223))

        name = sample_names[state["current_idx"]]
        sample = dataset.get_inference_data(name)

        depth_image = sample.get("depth_image")
        depth_m = 0.0
        if depth_image is not None:
            dh, dw = depth_image.shape[:2]
            du = int(np.clip(u_224 * dw / 224.0, 0, dw - 1))
            dv = int(np.clip(v_224 * dh / 224.0, 0, dh - 1))
            depth_m = float(depth_image[dv, du]) / 1000.0
        K_224 = vis["camera_K"]
        point_uv = torch.tensor(
            [[u_224, v_224, depth_m]], dtype=torch.float32, device=device
        )
        camera_K = torch.from_numpy(np.asarray(K_224)).float().unsqueeze(0).to(device)
        betas = model.fixed_betas.squeeze(0)
        rgb = sample["rgb"].unsqueeze(0).to(device) if use_rgb else None

        # Build PCL. With a crop, rebuild around the live click so it matches
        # the cropped distribution; otherwise reuse the PCL prepared in
        # get_inference_data (real colors, 224-res, identical to training).
        pcl_xyz = None
        pcl_rgb = None
        if use_depth:
            crop_r = getattr(model, "pcl_crop_radius", None)
            if crop_r is not None and depth_image is not None:
                depth_m_arr = depth_image.astype(np.float32) / 1000.0
                point_xyz = pixel_to_xyz(u_224, v_224, depth_m, np.asarray(K_224))
                xyz, rgb_pcl = depth_to_pcl_tensors(
                    depth_m_arr,
                    sample["rgb_original"],
                    np.asarray(K_224),
                    center=point_xyz,
                    crop_radius=crop_r,
                )
                pcl_xyz = xyz.unsqueeze(0).to(device)
                pcl_rgb = rgb_pcl.unsqueeze(0).to(device) if pcl_use_rgb else None
            else:
                pcl_xyz = sample["pcl_xyz"].unsqueeze(0).to(device)
                pcl_rgb = (
                    sample["pcl_rgb"].unsqueeze(0).to(device) if pcl_use_rgb else None
                )

        with torch.no_grad():
            preds = model.sample(
                point_uv,
                camera_K,
                steps=sampling_steps,
                rgb=rgb,
                pcl_xyz=pcl_xyz,
                pcl_rgb=pcl_rgb,
            )

        K = vis["camera_K"]
        pred_grasp = mano_params_to_grasp_dict(
            preds[0],
            betas,
            model.mano,
            K,
            model.mesh_faces,
        )

        point_handle = None
        if depth_image is not None and depth_m > 0:
            fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
            pt3d = np.array(
                [(u_224 - cx) * depth_m / fx, (v_224 - cy) * depth_m / fy, depth_m]
            )
            point_handle = server.scene.add_icosphere(
                f"/predictions/click_{int(time.time_ns())}",
                radius=0.01,
                color=POINT_MARKER_COLOR,
                position=pt3d,
            )
            point_handle.visible = (
                gui_show_grasp.value and gui_show_point_condition.value
            )

        frames = None
        if gui_animate.value:
            off = gui_pre_offset.value / 100.0
            n_frames = min(120, max(2, int(gui_anim_duration.value * 60)))
            frames = mano_params_to_animation(
                preds[0],
                betas,
                model.mano,
                n_frames=n_frames,
                pre_offset_m=(off, off),
            )

        add_pred_hand(pred_grasp, point_handle=point_handle, frames=frames)
        if save_pred:
            save_prediction(name, pred_grasp, u_224, v_224)
        vis["last_click_norm"] = (u_norm, v_norm)
        update_gui_image()

    @gui_click_bridge.on_update
    def _(_event) -> None:
        raw = gui_click_bridge.value.strip()
        if not raw:
            return
        try:
            u_str, v_str = raw.split(",", maxsplit=1)
            handle_click(float(u_str), float(v_str))
        except ValueError:
            return

    def update_scene(name: str) -> None:
        for h in state["scene_handles"]:
            try:
                h.remove()
            except Exception:
                pass
        state["scene_handles"] = []
        pred_store.clear_all()
        vis["last_click_norm"] = None

        console.print(f"[cyan]Visualizing: {name}[/cyan]")

        data = dataset.get_inference_data(name)
        K = data["camera_K"]
        width, height = data["width"], data["height"]
        vis["camera_K"] = K
        vis["width"] = width
        vis["height"] = height
        vis["mesh_faces"] = data["mesh_faces"]
        vis["rgb_image"] = data["rgb_original"]

        update_gui_image()

        # Depth point cloud
        vis["depth_point_cloud"] = None
        depth_image = data["depth_image"]
        if depth_image is not None:
            depth_h, depth_w = depth_image.shape[:2]
            rgb_small = cv2.resize(data["rgb_original"], (depth_w, depth_h))
            result = backproject_depth_to_point_cloud(
                depth_image, rgb_small, K, width, height, gui_max_depth.value
            )
            if result is not None:
                pts, colors = result
                dpc = server.scene.add_point_cloud(
                    "/depth_points",
                    points=pts,
                    colors=colors,
                    point_size=gui_depth_point_size.value,
                    point_shape="rounded",
                )
                dpc.visible = gui_depth_point_size.value > 0
                state["scene_handles"].append(dpc)
                vis["depth_point_cloud"] = dpc

        # Camera frustum and frame
        fov = 2 * np.arctan2(height / 2, K[1, 1])
        aspect = width / height
        frustum = server.scene.add_camera_frustum(
            "/scene/rgb_frustum",
            fov=fov,
            aspect=aspect,
            scale=0.1,
            image=data["rgb_original"],
            format="jpeg",
            wxyz=(1.0, 0.0, 0.0, 0.0),
            position=(0.0, 0.0, 0.0),
        )
        frustum.visible = gui_show_rgb.value
        state["scene_handles"].append(frustum)
        vis["camera_frustum"] = frustum

        frame = server.scene.add_frame(
            "/scene/camera_frame",
            wxyz=(1.0, 0.0, 0.0, 0.0),
            position=(0.0, 0.0, 0.0),
            axes_length=0.12,
            axes_radius=0.004,
        )
        frame.visible = gui_show_camera_frame.value
        state["scene_handles"].append(frame)
        vis["camera_frame"] = frame

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
    def on_mesh_opacity(_):
        pred_store.set_mesh_opacity(gui_mesh_opacity.value)

    def on_depth_point_size(_):
        if vis.get("depth_point_cloud"):
            vis["depth_point_cloud"].point_size = gui_depth_point_size.value
            vis["depth_point_cloud"].visible = gui_depth_point_size.value > 0

    def on_show_grasp(_):
        pred_store.set_visible(
            gui_show_grasp.value,
            gui_show_wrist_frame.value,
            gui_show_point_condition.value,
        )

    def on_show_point_condition(_):
        pred_store.set_point_visible(
            gui_show_point_condition.value, gui_show_grasp.value
        )
        update_gui_image()

    def on_show_wrist(_):
        pred_store.set_wrist_visible(gui_show_wrist_frame.value, gui_show_grasp.value)

    def on_show_rgb(_):
        show = gui_show_rgb.value
        h = vis.get("camera_frustum")
        if h:
            h.visible = show
        f = vis.get("camera_frame")
        if f:
            f.visible = show and gui_show_camera_frame.value

    def on_show_camera_frame(_):
        f = vis.get("camera_frame")
        if f:
            f.visible = gui_show_rgb.value and gui_show_camera_frame.value

    def on_clear(ev):
        if gui_clear_btns.value == "Clear Last":
            pred_store.clear_last()
        else:
            pred_store.clear_all()

    gui_mesh_opacity.on_update(on_mesh_opacity)
    gui_depth_point_size.on_update(on_depth_point_size)
    gui_show_grasp.on_update(on_show_grasp)
    gui_show_point_condition.on_update(on_show_point_condition)
    gui_show_wrist_frame.on_update(on_show_wrist)
    gui_show_rgb.on_update(on_show_rgb)
    gui_show_camera_frame.on_update(on_show_camera_frame)
    gui_clear_btns.on_click(on_clear)

    update_scene(sample_names[0])

    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    tyro.cli(app)
