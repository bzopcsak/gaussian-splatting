import numpy as np
import torch


def _normalize(v, name):
    """Return v as a float array with unit length. Raise on zero length."""
    v = np.asarray(v, dtype=float)
    norm = np.linalg.norm(v)
    if norm == 0:
        raise ValueError(f"{name} has zero length and cannot be normalized")
    return v / norm


def qvec_to_rotmat(qvec):
    """Quaternion (w, x, y, z) -> 3x3 rotation matrix. Normalizes first."""
    w, x, y, z = _normalize(qvec, "qvec")
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)],
        [2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)],
    ])


def pose_to_viewmat(pose):
    """COLMAP pose -> 4x4 world-to-camera matrix: x_cam = R @ x_world + t."""
    viewmat = torch.eye(4, dtype=torch.float32)
    viewmat[:3, :3] = torch.tensor(qvec_to_rotmat(pose.qvec))
    viewmat[:3, 3] = torch.tensor(pose.tvec)
    return viewmat


def intrinsics_to_K(intr):
    """COLMAP PINHOLE camera -> 3x3 intrinsics matrix."""
    fx, fy, cx, cy = intr.params
    return torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32)
