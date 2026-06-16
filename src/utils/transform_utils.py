"""Transformation utilities."""

import torch
from roma import special_gramschmidt


def six_d_to_rotation_matrix(six_d: torch.Tensor) -> torch.Tensor:
    """Convert 6D representation to 3x3 rotation matrix using Gram-Schmidt."""
    six_d = six_d.reshape(*six_d.shape[:-1], 3, 2)
    return special_gramschmidt(six_d)
