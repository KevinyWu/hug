"""Load a trained checkpoint, run prediction on the val set, save grasp_pred/{name}.pkl."""

import json
import logging
import pickle
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import tyro
from omegaconf import OmegaConf
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from safetensors import safe_open
from safetensors.torch import load_file
from torch.utils.data import DataLoader

from .dataloader.data_classes import CameraIntrinsics, Grasp, GraspData
from .dataloader.grasp_dataset import GraspDataset
from .models.grasp_model import GraspFlowModel
from .models.mano import mano_params_to_grasp_dict

logger = logging.getLogger(__name__)
console = Console()


def _perf_table(rows: list[tuple[int, int, float, float]]) -> Table:
    table = Table(title="Inference timing", expand=False)
    table.add_column("Batch", style="cyan", no_wrap=True, justify="right")
    table.add_column("Samples", style="yellow", justify="right")
    table.add_column("Elapsed (ms)", style="magenta", justify="right")
    table.add_column("ms/sample", style="green", justify="right")
    for batch_idx, bs, elapsed_ms, ms_per in rows:
        tag = " (warmup)" if batch_idx == 0 else ""
        table.add_row(
            f"{batch_idx}{tag}",
            str(bs),
            f"{elapsed_ms:.1f}",
            f"{ms_per:.2f}",
        )
    return table


def resolve_checkpoint_path(checkpoint_path: Path) -> Path:
    """Resolve a checkpoint path, preferring a safetensors then slim bf16 file.

    Accepts either a file path or a directory. For a directory, prefers
    `hug_full.safetensors`, then `model_inference_bf16.pt`, then `model.pt`.
    """
    p = Path(checkpoint_path)
    if p.is_dir():
        for name in ("hug_full.safetensors", "model_inference_bf16.pt", "model.pt"):
            candidate = p / name
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(f"{p} is a directory but contains no checkpoint")
    if not p.is_file():
        raise FileNotFoundError(f"checkpoint not found: {p}")
    return p


def load_raw_checkpoint(checkpoint_path: Path, device: str) -> dict:
    """Load a checkpoint into a uniform dict regardless of file format."""
    p = Path(checkpoint_path)
    if p.suffix == ".safetensors":
        ckpt = {"model": load_file(str(p), device=device)}
        with safe_open(str(p), framework="pt") as f:
            meta = f.metadata() or {}
        for k, v in meta.items():
            ckpt[k] = json.loads(v)
        return ckpt
    return torch.load(p, map_location=device, weights_only=False)


def load_model(
    checkpoint_path: Path,
    use_ema: bool,
    device: str,
):
    """Build a GraspFlowModel from a checkpoint, restoring config and norm stats.

    Reads the config and norm stats embedded in the checkpoint (or a sibling
    `.hydra/config.yaml` for legacy runs), then loads EMA or raw weights onto device.
    """
    checkpoint_path = resolve_checkpoint_path(checkpoint_path)
    ckpt = load_raw_checkpoint(checkpoint_path, device)
    if "cfg" in ckpt:
        cfg = OmegaConf.create(ckpt["cfg"])
        logger.info("Loaded config from checkpoint")
    else:
        # Legacy: load resolved config saved by Hydra next to the checkpoint
        hydra_cfg = Path(checkpoint_path).parent / ".hydra" / "config.yaml"
        if hydra_cfg.exists():
            cfg = OmegaConf.load(hydra_cfg)
            logger.info(f"Loaded config from {hydra_cfg}")
        else:
            raise FileNotFoundError(
                f"No config in checkpoint and no .hydra/config.yaml found at {hydra_cfg}. "
                "Re-train or manually provide the config."
            )
    norm_stats = ckpt.get("norm_stats")
    console.print(
        Panel(
            Syntax(
                OmegaConf.to_yaml(cfg),
                "yaml",
                theme="ansi_dark",
                line_numbers=False,
            ),
            title="Config",
            border_style="cyan",
        )
    )
    model = GraspFlowModel(cfg, norm_stats=norm_stats).to(device)

    if use_ema and ckpt.get("ema") is not None:
        ema_state = ckpt["ema"]
        model_state = {}
        for k, v in ema_state.items():
            if k.startswith("module."):
                model_state[k[len("module.") :]] = v
            elif k.startswith("n_averaged"):
                continue
            else:
                model_state[k] = v
        model.load_state_dict(model_state, strict=False)
        logger.info(f"Loaded EMA weights from {checkpoint_path}")
    else:
        kind = ckpt.get("weights_kind", "model")
        model.load_state_dict(ckpt["model"], strict=False)
        logger.info(f"Loaded {kind} weights from {checkpoint_path}")

    model.eval()
    # Surface the PCL crop config so apps can rebuild PCLs at click time
    # with the same crop the model was trained with.
    model.pcl_crop_radius = cfg.trainer.model.get("pcl_crop_radius", 0.3)
    return model


def main(
    checkpoint_path: Path,
    dataset_path: Path,
    use_ema: bool = True,
    sampling_steps: int = 50,
    batch_size: int = 32,
    sample_name: Optional[str] = None,
    num_samples: Optional[int] = 256,
) -> None:
    """Run inference on a dataset val split and save per-sample grasp predictions.

    Args:
        checkpoint_path: Checkpoint file or run directory (prefers the slim bf16 file).
        dataset_path: Folder of input .pkl samples; writes grasp_pred/*.pkl.
        use_ema: Load EMA weights when present in the checkpoint.
        sampling_steps: Euler ODE steps for flow-matching sampling.
        batch_size: Inference batch size.
        sample_name: Single stem or path to a .txt of stems; overrides num_samples.
        num_samples: Random subset size when sample_name is unset (None = all).
    """
    checkpoint_path = resolve_checkpoint_path(checkpoint_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = load_model(checkpoint_path, use_ema, device)

    # Resolve sample selection
    indices = None
    if sample_name is not None:
        if sample_name.endswith(".txt"):
            names = Path(sample_name).read_text().strip().splitlines()
        else:
            names = [sample_name]
        all_pkls = GraspDataset.find_pkls(dataset_path)
        all_stems = [
            p.relative_to(dataset_path).with_suffix("").as_posix() for p in all_pkls
        ]
        stem_to_idx = {s: i for i, s in enumerate(all_stems)}
        indices = [stem_to_idx[n] for n in names if n in stem_to_idx]
    elif num_samples is not None:
        n = len(GraspDataset.find_pkls(dataset_path))
        indices = sorted(
            np.random.default_rng(42)
            .choice(n, size=min(num_samples, n), replace=False)
            .tolist()
        )

    use_depth = getattr(model, "use_depth", False)
    use_rgb = getattr(model, "use_rgb", True)
    dataset = GraspDataset(
        str(dataset_path),
        split="val",
        indices=indices,
        use_rgb=use_rgb,
        use_depth=use_depth,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    out_dir = dataset_path / "grasp_pred"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Running inference on {len(dataset)} samples -> {out_dir}")

    sample_idx = 0
    total_sample_time = 0.0
    total_timed = 0
    betas = model.fixed_betas.squeeze(0)
    perf_rows: list[tuple[int, int, float, float]] = []
    live = Live(_perf_table(perf_rows), console=console, refresh_per_second=4)
    live.start()
    for batch_idx, batch in enumerate(loader):
        point_uv = batch["point_uv"].to(device)
        camera_K = batch["camera_K"].to(device)
        stems = batch["stem"]
        rgb = batch["rgb"].to(device) if use_rgb else None
        pcl_xyz = batch["pcl_xyz"].to(device) if use_depth else None

        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
            samples = model.sample(
                point_uv,
                camera_K,
                steps=sampling_steps,
                rgb=rgb,
                pcl_xyz=pcl_xyz,
            )
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        bs = point_uv.shape[0]
        ms_per_sample = elapsed * 1000.0 / bs
        if batch_idx > 0:
            total_sample_time += elapsed
            total_timed += bs
        perf_rows.append((batch_idx, bs, elapsed * 1000.0, ms_per_sample))
        live.update(_perf_table(perf_rows))

        for i in range(point_uv.shape[0]):
            stem = stems[i]
            orig_path = dataset_path / f"{stem}.pkl"
            with open(orig_path, "rb") as f:
                orig_data = pickle.load(f)

            cam = orig_data["camera"]
            cam_orig = orig_data["camera_original"]
            camera = CameraIntrinsics(**cam) if isinstance(cam, dict) else cam
            camera_original = (
                CameraIntrinsics(**cam_orig) if isinstance(cam_orig, dict) else cam_orig
            )

            grasp_dict = mano_params_to_grasp_dict(
                samples[i],
                betas,
                model.mano,
                camera.K,
                model.mesh_faces,
            )
            # Encode the normalized sampled point (u, v) in object_mask as 8 bytes
            # of float32 so visualize_predictions can decode it.
            point_norm = (point_uv[i, :2].cpu().numpy() / 224.0).astype(np.float32)
            grasp_data = GraspData(
                object_name=orig_data.get("object_name", ""),
                frame_index=orig_data.get("frame_index", 0),
                grasp_index=orig_data.get("grasp_index", 0),
                camera=camera,
                camera_original=camera_original,
                grasp=Grasp(**grasp_dict),
                image=orig_data.get("image", b""),
                depth=orig_data.get("depth", b""),
                object_mask=point_norm.tobytes(),
            )

            out_path = out_dir / f"{stem}.pkl"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "wb") as f:
                pickle.dump(asdict(grasp_data), f)

            sample_idx += 1

    live.stop()
    logger.info(f"Saved {sample_idx} predictions to {out_dir}")
    if total_timed > 0:
        avg_ms = total_sample_time * 1000.0 / total_timed
        summary = Table(title="Inference summary", expand=False, show_header=False)
        summary.add_column(style="cyan", no_wrap=True)
        summary.add_column(style="magenta", justify="right")
        summary.add_row("Avg ms/sample", f"{avg_ms:.2f}")
        summary.add_row("Samples timed", f"{total_timed}")
        summary.add_row("Sampling steps", f"{sampling_steps}")
        summary.add_row("Batch size", f"{batch_size}")
        summary.add_row("Note", "[yellow]warmup batch excluded[/]")
        console.print(summary)


if __name__ == "__main__":
    tyro.cli(main)
