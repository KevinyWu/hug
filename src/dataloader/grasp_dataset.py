"""Grasp dataset for training and evaluation."""

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from ..utils.data_keys import MANO_RIGHT_MESH_FACES_FILE, MANO_RIGHT_SHAPE_FILE
from ..utils.pcl_utils import depth_to_pcl_tensors, pixel_to_xyz

logger = logging.getLogger(__name__)

# Prediction output dir, excluded from input pkl discovery
PRED_DIRNAME = "grasp_pred"


class GraspDataset(Dataset):
    """Dataset for grasp prediction from RGB + object mask.

    Loads RGB images, object masks, and ground truth MANO parameters from any
    `.pkl` found recursively under dataset_path (the `grasp_pred/` output dir
    excluded). Output MANO pose: 99D = 3 (metric t in meters) +
    6 (wrist R_6d) + 90 (15 joints * 6D).
    """

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    def __init__(
        self,
        dataset_path: str,
        split: str = "train",
        indices: Optional[List[int]] = None,
        image_size: int = 224,
        use_rgb: bool = True,
        use_depth: bool = True,
        samples_filename: Optional[str] = None,
        n_points_input: int = 4096,
        pcl_crop_radius: Optional[float] = 0.3,
    ):
        self.dataset_path = Path(dataset_path)
        self.split = split
        self.image_size = image_size
        self.use_rgb = use_rgb
        self.use_depth = use_depth
        self.n_points_input = n_points_input
        self.pcl_crop_radius = pcl_crop_radius

        self.grasp_files = self._load_file_list(
            self.dataset_path, split, samples_filename
        )

        if indices is not None:
            self.grasp_files = [self.grasp_files[i] for i in indices]

        # Image transforms (ImageNet normalization for DINOv2)
        self.rgb_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD),
            ]
        )
        self.mask_transform = transforms.ToTensor()

    def __len__(self) -> int:
        return len(self.grasp_files)

    @staticmethod
    def find_pkls(root: Path) -> List[Path]:
        """All .pkl under root (recursive, sorted), excluding the grasp_pred/ output dir."""
        return sorted(
            p
            for p in root.rglob("*.pkl")
            if PRED_DIRNAME not in p.relative_to(root).parts
        )

    @staticmethod
    def _load_file_list(
        dataset_path: Path,
        split: str,
        samples_filename: Optional[str] = None,
    ) -> List[Path]:
        """Resolve list of pkl paths under dataset_path.

        Reads stems from samples.txt at the dataset root if present, else globs
        recursively (excluding `grasp_pred/`) and caches the list. Stems are
        paths relative to dataset_path (e.g. `scene/00000064` for a nested
        layout, or just `00000064` when flat). If `samples_filename` is provided
        that file must exist (subset request).
        """
        filename = samples_filename or "samples.txt"
        samples_file = dataset_path / filename
        if samples_filename is not None and not samples_file.exists():
            raise FileNotFoundError(f"Subset file not found: {samples_file}")
        if samples_file.exists():
            stems = samples_file.read_text().splitlines()
            files = [dataset_path / f"{stem}.pkl" for stem in stems if stem]
            logger.info(f"Loaded {len(files)} {split} files from {samples_file}")
            return files
        logger.info(
            f"Globbing {dataset_path} (no samples file, may take minutes on NFS)..."
        )
        files = GraspDataset.find_pkls(dataset_path)
        samples_file.parent.mkdir(parents=True, exist_ok=True)
        stems = "\n".join(
            p.relative_to(dataset_path).with_suffix("").as_posix() for p in files
        )
        samples_file.write_text(stems + "\n")
        logger.info(f"Wrote {len(files)} {split} stems to {samples_file}")
        return files

    def _load_grasp_data(self, grasp_path: Path) -> Dict:
        with open(grasp_path, "rb") as f:
            return pickle.load(f)

    def _get_mano_params(self, grasp_data) -> torch.Tensor:
        """Extract 99D MANO pose: t(3, metric meters) + R_6d(6) + pose_6d(90).

        Translation is metric [x, y, z], matching the model's PCL + 3D query
        point space.
        """
        grasp = grasp_data["grasp"]
        t = grasp["t"].flatten()
        R_6d = grasp["R_6d"].flatten()
        pose_6d = grasp["pose_6d"].flatten()
        mano_params = np.concatenate([t, R_6d, pose_6d], axis=0).astype(np.float32)
        return torch.from_numpy(mano_params)

    @staticmethod
    def _decode_image(image_bytes: bytes) -> np.ndarray:
        """Decode JPEG bytes -> (H,W,3) uint8 RGB."""
        arr = np.frombuffer(image_bytes, np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    @staticmethod
    def _decode_mask(mask_bytes: bytes) -> np.ndarray:
        """Decode PNG bytes -> (H,W) uint8 mask."""
        arr = np.frombuffer(mask_bytes, np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)

    @staticmethod
    def _decode_depth_uint16(depth_bytes: bytes) -> np.ndarray:
        """Decode PNG bytes -> (H,W) uint16 depth (1mm units)."""
        arr = np.frombuffer(depth_bytes, np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)

    def _depth_meters(self, depth_bytes: bytes) -> torch.Tensor:
        """Decode depth bytes to (H,W) float32 tensor in meters."""
        depth = self._decode_depth_uint16(depth_bytes).astype(np.float32)
        depth[depth >= 65535] = 0
        depth_m = np.nan_to_num(depth / 1000.0, nan=0.0, posinf=0.0, neginf=0.0)
        depth_m = np.clip(depth_m, 0, 100.0)
        return torch.from_numpy(depth_m)

    def _build_pcl(
        self,
        depth_m: torch.Tensor,
        rgb_np: np.ndarray,
        K: np.ndarray,
        point_xyz: Optional[np.ndarray] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Backproject depth + RGB into fixed-size (xyz, rgb_pcl) PCL tensors.

        When point_xyz and self.pcl_crop_radius are both set, restricts the
        point pool to a sphere of that radius around point_xyz before the
        random subsample, concentrating density on the grasp region.
        """
        crop_radius = self.pcl_crop_radius if point_xyz is not None else None
        return depth_to_pcl_tensors(
            depth_m,
            rgb_np,
            K,
            n_points=self.n_points_input,
            center=point_xyz,
            crop_radius=crop_radius,
        )

    def _sample_point_from_mask(
        self, mask: torch.Tensor, depth_m: torch.Tensor
    ) -> torch.Tensor:
        """Sample a pixel from eroded mask w/ valid depth, return (u, v, d_meters).

        Returns a (3,) tensor. The model backprojects (u, v, d) → metric XYZ via K
        in `encode_scene` — pure geometric op, K never enters learned weights.
        """
        mask_np = (mask.squeeze(0).numpy() > 0.5).astype(np.uint8)
        kernel = np.ones((3, 3), np.uint8)
        eroded = cv2.erode(mask_np, kernel, iterations=1)
        depth_np = depth_m.numpy() if isinstance(depth_m, torch.Tensor) else depth_m
        valid = (eroded > 0) & (depth_np > 0)
        if valid.sum() == 0:
            valid = (mask_np > 0) & (depth_np > 0)
        if valid.sum() == 0:
            ys, xs = np.where(mask_np > 0)
            idx = np.random.randint(len(ys))
            v_pix, u_pix = int(ys[idx]), int(xs[idx])
            d = float(depth_np[depth_np > 0].mean()) if (depth_np > 0).any() else 0.5
        else:
            ys, xs = np.where(valid)
            idx = np.random.randint(len(ys))
            v_pix, u_pix = int(ys[idx]), int(xs[idx])
            d = float(depth_np[v_pix, u_pix])
        return torch.tensor([float(u_pix), float(v_pix), d], dtype=torch.float32)

    def get_original_for_viz(self, idx: int) -> Dict[str, np.ndarray]:
        """Load 224-res data from pkl for 3D viz; no external files needed."""
        grasp_path = self.grasp_files[idx]
        grasp_data = self._load_grasp_data(grasp_path)

        rgb_small = self._decode_image(grasp_data["image"])
        depth_image = self._decode_depth_uint16(grasp_data["depth"])
        camera_K_small = grasp_data["camera"]["K"]

        key = grasp_path.relative_to(self.dataset_path).with_suffix("").as_posix()

        return {
            "rgb_small": rgb_small,
            "depth_image": depth_image,
            "camera_K_small": camera_K_small,
            "stem": key,
        }

    def get_inference_data(self, stem: str) -> Dict:
        """Load minimal data needed for inference: rgb, depth, camera, mesh_faces."""
        pkl_path = self.dataset_path / f"{stem}.pkl"
        grasp_data = self._load_grasp_data(pkl_path)
        camera = grasp_data["camera"]
        K = camera["K"] if isinstance(camera, dict) else camera.K
        width = camera["width"] if isinstance(camera, dict) else camera.width
        height = camera["height"] if isinstance(camera, dict) else camera.height

        grasp = grasp_data.get("grasp")
        mesh_faces = grasp.get("mesh_faces") if grasp else None
        if mesh_faces is None:
            mesh_faces = np.load(MANO_RIGHT_MESH_FACES_FILE)

        rgb_np = self._decode_image(grasp_data["image"])
        rgb_original_path = self.dataset_path / "image_original" / f"{stem}.jpg"
        rgb_original = (
            np.array(Image.open(rgb_original_path).convert("RGB"))
            if rgb_original_path.exists()
            else rgb_np
        )

        depth_image = self._decode_depth_uint16(grasp_data["depth"])
        shape = (
            grasp["shape"] if grasp else np.load(MANO_RIGHT_SHAPE_FILE).reshape(1, 10)
        )
        mano_shape = torch.from_numpy(np.asarray(shape).flatten()).float()

        out = {
            "camera_K": K,
            "width": width,
            "height": height,
            "mesh_faces": mesh_faces,
            "rgb_original": rgb_original,
            "depth_image": depth_image,
            "mano_shape": mano_shape,
        }
        if self.use_rgb:
            out["rgb"] = self.rgb_transform(Image.fromarray(rgb_np))
        if self.use_depth:
            depth_m = self._depth_meters(grasp_data["depth"])
            K_224 = grasp_data["camera"]["K"]
            xyz, pcl_rgb = self._build_pcl(depth_m, rgb_np, K_224)
            out["pcl_xyz"] = xyz
            out["pcl_rgb"] = pcl_rgb
        return out

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        grasp_path = self.grasp_files[idx]
        grasp_data = self._load_grasp_data(grasp_path)
        # Root-relative stem so nested layouts round-trip back to the pkl path
        # (matches app/inference resolution via dataset_path / f"{stem}.pkl").
        stem = grasp_path.relative_to(self.dataset_path).with_suffix("").as_posix()

        mask_np = self._decode_mask(grasp_data["object_mask"])
        mask_tensor = self.mask_transform(Image.fromarray(mask_np))
        depth_m = self._depth_meters(grasp_data["depth"])
        K_np = grasp_data["camera"]["K"]
        stored_uv = grasp_data.get("condition_point")
        if stored_uv is not None:
            u, v = float(stored_uv[0]), float(stored_uv[1])
            depth_np = depth_m.numpy() if isinstance(depth_m, torch.Tensor) else depth_m
            H, W = depth_np.shape
            ui = int(np.clip(round(u), 0, W - 1))
            vi = int(np.clip(round(v), 0, H - 1))
            d = float(depth_np[vi, ui])
            if d <= 0 and (depth_np > 0).any():
                d = float(depth_np[depth_np > 0].mean())
            point_uv = torch.tensor([u, v, d], dtype=torch.float32)
        else:
            point_uv = self._sample_point_from_mask(mask_tensor, depth_m)

        camera_K = torch.from_numpy(K_np).float()

        rgb_np = self._decode_image(grasp_data["image"])

        out = {
            "point_uv": point_uv,
            "camera_K": camera_K,
            "stem": stem,
        }
        # Eval pkls carry no grasp label; GT fields are train-only
        grasp = grasp_data.get("grasp")
        if grasp is not None:
            out["mano_params"] = self._get_mano_params(grasp_data)
            out["mano_shape"] = torch.from_numpy(grasp["shape"].flatten()).float()
            out["landmarks_3d"] = torch.from_numpy(grasp["landmarks_3d"]).float()
            out["landmarks_2d"] = torch.from_numpy(grasp["landmarks_2d"]).float()
        if self.use_rgb:
            out["rgb"] = self.rgb_transform(Image.fromarray(rgb_np))
        if self.use_depth:
            point_xyz = pixel_to_xyz(
                float(point_uv[0]), float(point_uv[1]), float(point_uv[2]), K_np
            )
            xyz, pcl_rgb = self._build_pcl(depth_m, rgb_np, K_np, point_xyz=point_xyz)
            out["pcl_xyz"] = xyz
            out["pcl_rgb"] = pcl_rgb
        return out
