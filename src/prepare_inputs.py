"""Prepare a single-capture inference pkl from an rgb + depth + intrinsics folder.

Point this at a folder holding one RGB image, one depth map, and one intrinsics
file; it center-crops to a square, resizes to 224x224, adjusts the intrinsics,
and writes `{stem}.pkl` (the `GraspData` eval schema, no grasp label) beside the
inputs.

Expected folder contents (auto-detected by filename):
    *rgb*.{png,jpg}        uint8 RGB, any HxW
    *depth*.png            uint16, 1mm units, same HxW as rgb
    *intrinsics*.{txt,npy,json}  intrinsics at the rgb resolution

The intrinsics file is either four numbers `fx fy cx cy` or a full 3x3 K matrix.

The pkl lands beside the inputs; point `--dataset-path` at the folder to run the
app (it discovers .pkl samples recursively).
"""

import json
import pickle
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import tyro
from rich.console import Console

from .dataloader.data_classes import CameraIntrinsics, GraspData

console = Console()

TARGET_SIZE = 224
IMAGE_EXTS = (".png", ".jpg", ".jpeg")
INTRINSICS_EXTS = (".txt", ".csv", ".npy", ".json")


def _center_crop_square(img: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Center-crop to a square along the shorter side. Returns crop + offsets."""
    h, w = img.shape[:2]
    size = min(h, w)
    x_off = (w - size) // 2
    y_off = (h - size) // 2
    return img[y_off : y_off + size, x_off : x_off + size], x_off, y_off


def _adjust_K(K: np.ndarray, x_off: int, y_off: int, scale: float) -> np.ndarray:
    """Shift principal point for an (x_off, y_off) crop then scale by `scale`."""
    K_new = K.copy().astype(np.float64)
    K_new[0, 2] -= x_off
    K_new[1, 2] -= y_off
    K_new[:2, :] *= scale
    return K_new


def _encode_depth_224(depth: np.ndarray) -> bytes:
    """Center-crop + nearest-resize uint16 depth to 224 and PNG-encode."""
    if depth.dtype != np.uint16:
        raise ValueError(f"depth must be uint16, got {depth.dtype}")
    depth_sq, _, _ = _center_crop_square(depth)
    depth_224 = cv2.resize(
        depth_sq, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_NEAREST
    )
    _, buf = cv2.imencode(".png", depth_224)
    return buf.tobytes()


def prepare_pkl(
    rgb: np.ndarray,
    depth: np.ndarray,
    K: np.ndarray,
    stem: str,
    out_dir: Path,
    object_name: str = "",
) -> Path:
    """Encode one (rgb, depth, K) sample to an inference pkl.

    Center-crops the input to a square (shorter side), resizes to 224x224, and
    adjusts K accordingly. Stores the original-square K as `camera_original`.

    Args:
        rgb: (H, W, 3) uint8 RGB image.
        depth: (H, W) uint16 depth, 1mm units. Must match rgb HxW.
        K: (3, 3) intrinsics at the original RGB resolution.
        stem: Output filename stem.
        out_dir: Directory to write `{stem}.pkl` into (created if missing).
        object_name: Optional string saved in the pkl.
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError(f"rgb must be (H,W,3) uint8, got {rgb.shape} {rgb.dtype}")
    if depth.shape[:2] != rgb.shape[:2]:
        raise ValueError(f"rgb {rgb.shape[:2]} != depth {depth.shape[:2]}")

    rgb_sq, x_off, y_off = _center_crop_square(rgb)
    sq_size = rgb_sq.shape[0]
    K_orig = _adjust_K(K, x_off, y_off, scale=1.0)
    K_224 = _adjust_K(K, x_off, y_off, scale=TARGET_SIZE / sq_size)

    rgb_224 = cv2.resize(
        rgb_sq, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_AREA
    )
    rgb_bgr = cv2.cvtColor(rgb_224, cv2.COLOR_RGB2BGR)
    _, img_buf = cv2.imencode(".jpg", rgb_bgr)

    entry = asdict(
        GraspData(
            object_name=object_name,
            frame_index=0,
            grasp_index=0,
            camera=CameraIntrinsics(K=K_224, width=TARGET_SIZE, height=TARGET_SIZE),
            camera_original=CameraIntrinsics(K=K_orig, width=sq_size, height=sq_size),
            grasp=None,
            image=img_buf.tobytes(),
            depth=_encode_depth_224(depth),
            object_mask=b"",
        )
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{stem}.pkl"
    tmp = out_path.with_suffix(".tmp.pkl")
    with open(tmp, "wb") as f:
        pickle.dump(entry, f)
    tmp.rename(out_path)
    return out_path


def _load_intrinsics(path: Path) -> np.ndarray:
    """Load a 3x3 K from `fx fy cx cy`, a 3x3 matrix, or .npy/.json.

    Text/csv files may hold four numbers (`fx fy cx cy`) or nine (a flat 3x3).
    JSON may be a bare 3x3 list or a dict with a `K` key.
    """
    if path.suffix == ".npy":
        K = np.asarray(np.load(path), dtype=np.float64)
    elif path.suffix == ".json":
        data = json.loads(path.read_text())
        K = np.asarray(data["K"] if isinstance(data, dict) and "K" in data else data)
    else:
        K = np.loadtxt(path)
    vals = np.asarray(K, dtype=np.float64).ravel()
    if vals.size == 4:
        fx, fy, cx, cy = vals
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    if vals.size == 9:
        return vals.reshape(3, 3)
    raise ValueError(
        f"{path}: expected 4 (fx fy cx cy) or 9 (3x3) numbers, got {vals.size}"
    )


def _read_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"Failed to read {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _read_depth_uint16(path: Path) -> np.ndarray:
    d = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if d is None:
        raise IOError(f"Failed to read {path}")
    if d.dtype != np.uint16:
        raise ValueError(f"{path}: expected uint16 depth, got {d.dtype}")
    return d


def _autofind(folder: Path, substr: str, exts: tuple[str, ...]) -> Optional[Path]:
    """Find the lone file in `folder` whose name contains `substr` with `exts`."""
    hits = [
        p
        for p in sorted(folder.iterdir())
        if p.suffix.lower() in exts and substr in p.stem.lower()
    ]
    if len(hits) > 1:
        raise ValueError(f"Multiple '{substr}' files under {folder}: {[p.name for p in hits]}")
    return hits[0] if hits else None


def _resolve_inputs(
    dataset_path: Path,
    rgb: Optional[Path],
    depth: Optional[Path],
    intrinsics: Optional[Path],
) -> tuple[Path, Path, Path]:
    """Auto-detect the rgb, depth, and intrinsics files in a flat capture folder."""
    depth = depth or _autofind(dataset_path, "depth", IMAGE_EXTS)
    if depth is None:
        raise FileNotFoundError(f"No *depth*.{{png,jpg}} under {dataset_path}")

    intrinsics = intrinsics or _autofind(dataset_path, "intrinsics", INTRINSICS_EXTS)
    if intrinsics is None:
        txts = [
            p for p in sorted(dataset_path.iterdir()) if p.suffix.lower() in INTRINSICS_EXTS
        ]
        if len(txts) != 1:
            raise FileNotFoundError(
                f"No intrinsics file under {dataset_path} (name it `*intrinsics*` or keep a "
                f"single .txt/.npy/.json)"
            )
        intrinsics = txts[0]

    if rgb is None:
        imgs = [
            p
            for p in sorted(dataset_path.iterdir())
            if p.suffix.lower() in IMAGE_EXTS and p != depth
        ]
        if len(imgs) != 1:
            raise FileNotFoundError(
                f"Expected exactly one RGB image under {dataset_path} (besides depth), found "
                f"{[p.name for p in imgs]}; pass --rgb explicitly"
            )
        rgb = imgs[0]
    return rgb, depth, intrinsics


def main(
    dataset_path: Path,
    rgb: Optional[Path] = None,
    depth: Optional[Path] = None,
    intrinsics: Optional[Path] = None,
    stem: Optional[str] = None,
    object_name: str = "",
) -> None:
    """Build one inference pkl from a flat rgb + depth + intrinsics capture folder.

    Args:
        dataset_path: Capture folder holding the rgb image, depth map, and intrinsics.
        rgb: RGB image path; auto-detected if omitted (the lone non-depth image).
        depth: uint16 depth png path; auto-detected from `*depth*` if omitted.
        intrinsics: Intrinsics path (`fx fy cx cy`, 3x3, .npy, or .json);
            auto-detected from `*intrinsics*` or a lone text file if omitted.
        stem: Output filename stem; defaults to the rgb stem (trailing `rgb`
            stripped) or the folder name.
        object_name: Optional string saved in the pkl.
    """
    if not dataset_path.is_dir():
        raise NotADirectoryError(dataset_path)
    rgb_path, depth_path, intr_path = _resolve_inputs(dataset_path, rgb, depth, intrinsics)

    if stem is None:
        s = rgb_path.stem
        if s.lower().endswith("rgb"):
            s = s[:-3].rstrip("_-")
        stem = s or dataset_path.name

    K = _load_intrinsics(intr_path)
    rgb_img = _read_rgb(rgb_path)
    depth_img = _read_depth_uint16(depth_path)
    out_path = prepare_pkl(rgb_img, depth_img, K, stem, dataset_path, object_name)

    console.print(
        f"[cyan]rgb[/cyan] {rgb_path.name}  [cyan]depth[/cyan] {depth_path.name}  "
        f"[cyan]intrinsics[/cyan] {intr_path.name}"
    )
    console.print(f"[green]wrote {out_path}[/green]")


if __name__ == "__main__":
    tyro.cli(main)
