"""Readers for COLMAP sparse models in text format (sparse_txt/)."""
import os
from dataclasses import dataclass

import numpy as np

from .geometry import qvec_to_rotmat


# --- cameras.txt --------------------------------------------------------------
@dataclass
class Camera:
    """One camera.

    File format, one camera per line:
        CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]

    For the PINHOLE model, PARAMS[] = fx, fy, cx, cy.
    """
    id: int
    model: str
    width: int
    height: int
    params: np.ndarray

    @property
    def K(self):
        """3x3 intrinsics matrix. Only valid for the PINHOLE model."""
        assert self.model == "PINHOLE", f"K is only defined for PINHOLE, got {self.model}"
        fx, fy, cx, cy = self.params
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])


def read_cameras_txt(path):
    """Return {camera_id: Camera}."""
    cameras = {}
    with open(path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            e = line.split()
            cam = Camera(int(e[0]), e[1], int(e[2]), int(e[3]), np.array(e[4:], dtype=float))
            cameras[cam.id] = cam
    return cameras


# --- images.txt ---------------------------------------------------------------
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
    tvec: np.ndarray         # (3,) x, y, z, world origin in camera coordinates
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
    
    @property
    def world_to_cam(self):
        """4x4 world-to-camera matrix."""
        M = np.eye(4)
        M[:3, :3] = self.R
        M[:3, 3] = self.tvec
        return M


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


# --- points3D.txt -------------------------------------------------------------
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
