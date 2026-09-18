#!/usr/bin/env bash
set -euo pipefail  # Exit script if command fails.

IMAGES="/home/boldi/code/MA/gs-intro/tomatoes/images"
WORK="$(dirname "$(realpath "$0")")"  # Directory the script is in.
RESULTS="colmap_results"
DB="$RESULTS/database.db"

cd "$WORK"
rm -rf "$RESULTS"
mkdir -p "$RESULTS"


# Detect SIFT keypoints and descriptors in every image.
colmap feature_extractor \
    --database_path "$DB" \
    --image_path "$IMAGES" \
    --ImageReader.camera_model SIMPLE_RADIAL \
    --SiftExtraction.use_gpu 1 \
    --SiftExtraction.max_image_size 5000


# Match features between all image pairs and keep geometrically verified matches.
colmap exhaustive_matcher \
    --database_path "$DB" \
    --SiftMatching.use_gpu 1

# Match only neighbouring images in filename order (faster for ordered captures).
# colmap sequential_matcher \
#     --database_path "$DB" \
#     --SiftMatching.use_gpu 1


# Register images one by one, triangulate 3D points, and refine with bundle adjustment.
colmap mapper \
    --database_path "$DB" \
    --image_path "$IMAGES" \
    --output_path "$RESULTS"


# Undistort every sparse model (0, 1, ...) for 3DGS.
for d in "$RESULTS"/[0-9]*/; do
    n=$(basename "$d")

    # Undistort images and write a pinhole model for 3DGS.
    colmap image_undistorter \
        --image_path "$IMAGES" \
        --input_path "$d" \
        --output_path "$RESULTS/${n}_undistorted" \
        --output_type COLMAP

    # Keep only images/ and sparse/; the rest is for dense reconstruction.
    rm -rf "$RESULTS/${n}_undistorted/stereo" "$RESULTS/${n}_undistorted/"run-colmap-*.sh

    # Convert .bin to .txt.
    mkdir -p "$RESULTS/${n}_undistorted/sparse_txt"
    colmap model_converter \
        --input_path "$RESULTS/${n}_undistorted/sparse" \
        --output_path "$RESULTS/${n}_undistorted/sparse_txt" \
        --output_type TXT

done

# Print analysis.
printf "\nRunning COLMAP model analyzer:\n\n"
colmap model_analyzer --path "$RESULTS/0"
