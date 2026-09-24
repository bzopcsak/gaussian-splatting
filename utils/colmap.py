"""Readers for COLMAP sparse models in text format (sparse_txt/)."""
from dataclasses import dataclass

import numpy as np


# --- Geometry ---------------------------------------------------------------
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
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z,     2*x*z + 2*w*y],
        [2*x*y + 2*w*z,     1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y,     2*y*z + 2*w*x,     1 - 2*x*x - 2*y*y],
    ])


# --- images.txt -------------------------------------------------------------
@dataclass
class Image:
    """One registered image.

    File format, two lines per image:
        IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
        POINTS2D[] as (X, Y, POINT3D_ID)

    (qvec, tvec) map world to camera: x_cam = R @ x_world + t.
    """
    id: int
    qvec: np.ndarray         # (4,) w, x, y, z
    tvec: np.ndarray         # (3,)
    camera_id: int
    name: str
    xys: np.ndarray          # (N, 2) keypoint pixel coordinates
    point3D_ids: np.ndarray  # (N,) -1 if not triangulated

    @property
    def R(self):
        return qvec_to_rotmat(self.qvec)

    @property
    def center(self):
        """Camera center in world coordinates: C = -R^T t."""
        return -self.R.T @ self.tvec


def read_images_txt(path):
    """Return {image_id: Image}."""
    with open(path) as f:
        # Keep empty lines: an image with no keypoints has an empty POINTS2D line.
        lines = [l.rstrip("\n") for l in f if not l.startswith("#")]

    images = {}
    for pose_line, points_line in zip(lines[0::2], lines[1::2]):
        # NAME may contain spaces, so everything after 9th split stays together.
        e = pose_line.split(maxsplit=9)
        pts = np.array(points_line.split(), dtype=float).reshape(-1, 3)
        img = Image(
            id=int(e[0]),
            qvec=np.array(e[1:5], dtype=float),
            tvec=np.array(e[5:8], dtype=float),
            camera_id=int(e[8]),
            name=e[9],
            xys=pts[:, :2],
            point3D_ids=pts[:, 2].astype(int),
        )
        images[img.id] = img
    return images


# --- points3D.txt -----------------------------------------------------------
@dataclass
class Points3D:
    """All triangulated COLMAP points.

    File format, one point per line:
        POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]

    Fields:
        - POINT3D_ID: unique point ID
        - X, Y, Z: 3D world coordinates
        - R, G, B: point color from pixel value
        - ERROR: mean reprojection error in pixels
        - TRACK: observations as (IMAGE_ID, POINT2D_IDX) pairs
    """

    ids: np.ndarray     # (N,)
    xyz: np.ndarray     # (N, 3) world coordinates
    rgb: np.ndarray     # (N, 3) uint8 colors
    errors: np.ndarray  # (N,) reprojection error in pixels
    tracks: list        # N arrays of shape (M, 2): (IMAGE_ID, POINT2D_IDX)


def read_points3D_txt(path):
    """Return a Points3D holding every point in the file."""
    ids, xyz, rgb, errors, tracks = [], [], [], [], []
    with open(path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            e = line.split()
            ids.append(int(e[0]))
            xyz.append([float(v) for v in e[1:4]])
            rgb.append([int(v) for v in e[4:7]])
            errors.append(float(e[7]))
            tracks.append(np.array(e[8:], dtype=int).reshape(-1, 2))

    return Points3D(
        ids=np.array(ids, dtype=int),
        xyz=np.array(xyz, dtype=float).reshape(-1, 3),
        rgb=np.array(rgb, dtype=np.uint8).reshape(-1, 3),
        errors=np.array(errors, dtype=float),
        tracks=tracks,
    )
