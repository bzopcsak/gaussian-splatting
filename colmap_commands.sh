#!/usr/bin/env bash
set -euo pipefail  # Exit script if command fails.

IMAGES="/home/boldi/code/MA/data/tomatoes/images"
WORK="$(dirname "$(realpath "$0")")"  # Directory the script is in.

cd "$WORK"
rm -rf database.db* sparse sparse_txt
mkdir -p sparse

colmap feature_extractor \
    --database_path database.db \
    --image_path "$IMAGES" \
    --ImageReader.camera_model SIMPLE_RADIAL \
    --SiftExtraction.use_gpu 1 \
    --SiftExtraction.max_image_size 5000

colmap exhaustive_matcher \
    --database_path database.db \
    --SiftMatching.use_gpu 1

# colmap sequential_matcher \
#     --database_path database.db \
#     --SiftMatching.use_gpu 1

colmap mapper \
    --database_path database.db \
    --image_path "$IMAGES" \
    --output_path sparse


# Convert .bin to .txt.
for d in sparse/*/; do
    n=$(basename "$d")
    mkdir -p "sparse_txt/$n"
    colmap model_converter --input_path "$d" --output_path "sparse_txt/$n" --output_type TXT
done
