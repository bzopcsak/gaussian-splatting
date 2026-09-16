# gaussian-splatting
Gaussian Splatting Experiment


# Dataset
<details>
    <summary>Link to download</summary>

Download [Bowl of Tomatoes
dataset](https://www.kaggle.com/datasets/simonbethke/bowl-of-tomatoes?select=00000.jpg)
from Kaggle.
</details>


# Running
<details>
    <summary>1. COLMAP</summary>

Run `colmap_commands.sh`. The script writes its output to `COLMAP_results/`. Set
`IMAGES` in the script to the directory containing the tomato images. Tune
`--SiftExtraction.max_image_size` until COLMAP registers all images into a
single model (`COLMAP_results/0`) without making the run too slow. An image is
registered once COLMAP has estimated its camera pose inside a model, meaning
where the camera was and which way it pointed.

## COLMAP output

The mapper writes one folder per model: `0/`, `1/`, and so on. Normally there is just `0/`; more folders mean the reconstruction split into disconnected pieces. Each folder contains the same three files, as `.bin` (the mapper's output) or `.txt` (after `model_converter`). The binary and text versions hold identical content. Lines starting with `#` are comments.

Only **registered** images appear in these files. Images COLMAP could not place are simply absent.

## `cameras.txt`: Intrinsics

One line per camera, describing how 3D directions map to pixels. Several images can share one camera. In this dataset every image has its own resolution, so every image gets its own camera.

```
CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]
```

The meaning of `PARAMS` depends on the camera model:

| Model            | Params                             |
| ---------------- | ---------------------------------- |
| `SIMPLE_PINHOLE` | `f cx cy`                          |
| `PINHOLE`        | `fx fy cx cy`                      |
| `SIMPLE_RADIAL`  | `f cx cy k`                        |
| `RADIAL`         | `f cx cy k1 k2`                    |
| `OPENCV`         | `fx fy cx cy k1 k2 p1 p2`          |

`f` is the focal length in pixels, `(cx, cy)` is the principal point, and `k`/`p` are radial and tangential distortion coefficients. Example:

```
69 SIMPLE_RADIAL 7713 5078 9685.834 3856.5 2539 -0.00204
```

This is camera 69: 7713×5078 px, focal length ≈ 9686 px, principal point at the image center, slight barrel distortion.

## `images.txt`: Poses and 2D observations

Two lines per registered image.

```
IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
POINTS2D[] as (X Y POINT3D_ID)
```

**Line 1** is the pose. The quaternion `(QW, QX, QY, QZ)` (scalar first) gives the rotation `R`, and `(TX, TY, TZ)` is the translation `t`. The pose is **world-to-camera**:

```
X_cam = R · X_world + t
```

The camera center in world coordinates is therefore `C = -Rᵀ t`, not `t`. The camera frame follows the OpenCV convention: x right, y down, z forward (the viewing direction). `CAMERA_ID` points into `cameras.txt`, and `NAME` is the image filename.

**Line 2** lists this image's keypoints as triples `X Y POINT3D_ID`. `(X, Y)` is the pixel position; the center of the top-left pixel is `(0.5, 0.5)`. `POINT3D_ID` is the 3D point the keypoint was triangulated into, or `-1` if it wasn't. The position of a triple in this list is its `POINT2D_IDX`, starting at 0, which `points3D.txt` refers back to.

## `points3D.txt`: Sparse point cloud

One line per triangulated 3D point.

```
POINT3D_ID X Y Z R G B ERROR TRACK[] as (IMAGE_ID POINT2D_IDX)
```

- `X Y Z` is the position in world coordinates, and `R G B` its color (0–255), averaged from the images that see it.
- `ERROR` is the mean reprojection error in pixels: how far the projected point lands, on average, from the keypoints it was triangulated from. Lower is better.
- `TRACK` lists every observation of the point as `IMAGE_ID POINT2D_IDX` pairs. Each pair means "keypoint number `POINT2D_IDX` in image `IMAGE_ID`". The **track length** is the number of images that see the point; longer tracks usually mean better-constrained points.

`images.txt` and `points3D.txt` link to each other in both directions: keypoint → 3D point via `POINT3D_ID`, and 3D point → keypoints via `TRACK`.

## `database.db`: Intermediate data

An SQLite database filled by extraction and matching, which the mapper then reads. Unlike the model files, it covers **all** images, registered or not.

| Table                  | Content                                                          |
| ---------------------- | ---------------------------------------------------------------- |
| `cameras`              | Initial intrinsics, before the mapper refines them                |
| `images`               | Image names and their camera IDs                                 |
| `keypoints`            | Keypoint positions (plus scale/orientation) per image            |
| `descriptors`          | 128-D SIFT descriptors per image                                 |
| `matches`              | Raw descriptor matches per image pair                            |
| `two_view_geometries`  | Verified inlier matches per pair, with the fitted F / E / H      |

## Coordinate frame and scale

The world frame is arbitrary. Its origin, orientation and scale are whatever the reconstruction happened to settle on, and units are not meters. Two runs on the same images can produce models that differ by a similarity transform (rotation, translation and scale).

## What 3D Gaussian Splatting uses

- `cameras` + `images`: intrinsics and poses for rendering each training view.
- `points3D`: `XYZ` and `RGB` to initialize one Gaussian per point.

The original 3DGS loader only accepts `PINHOLE` and `SIMPLE_PINHOLE`. With a distorting model like `SIMPLE_RADIAL`, run `colmap image_undistorter` first. It writes undistorted images and a matching pinhole model.
</details>
