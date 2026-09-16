#!/usr/bin/env bash
set -euo pipefail  # Exit script if command fails.

IMAGES="/home/boldi/code/MA/data/tomatoes/images"
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


# Convert .bin to .txt.
for d in "$RESULTS"/*/; do
    n=$(basename "$d")
    mkdir -p "$RESULTS/${n}_txt"
    colmap model_converter --input_path "$d" --output_path "$RESULTS/${n}_txt" --output_type TXT
done

# Print analysis.
colmap model_analyzer --path "$RESULTS/0"
