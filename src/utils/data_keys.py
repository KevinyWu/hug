"""Core file names and paths for required assets."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.absolute()

# MANO model folder
MANO_MODELS_FOLDER = PROJECT_ROOT / "assets" / "mano_models"
MANO_RIGHT_MESH_FACES_FILE = PROJECT_ROOT / "assets" / "mano_rhand_mesh_faces.npy"
MANO_RIGHT_SHAPE_FILE = PROJECT_ROOT / "assets" / "mano_rhand_shape.npy"

# Norm stats file
NORM_STATS_FILE = PROJECT_ROOT / "assets" / "norm_stats.json"

# Outputs folder
OUTPUTS_FOLDER = PROJECT_ROOT / "outputs"
