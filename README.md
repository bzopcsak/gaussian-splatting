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
</details>
