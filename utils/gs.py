import os
import shutil
from dataclasses import dataclass, fields, replace

import numpy as np
import torch
from gsplat import rasterization
from PIL import Image
from tqdm.auto import tqdm

from .geometry import intrinsics_to_K, pose_to_viewmat


@dataclass
class Splats:
    """N Gaussians, one row per Gaussian. Stores raw values so they can become nn.Parameters later."""
    means: torch.Tensor            # (N, 3) center in world coordinates
    quats: torch.Tensor            # (N, 4) rotation (w, x, y, z), normalized inside gsplat
    log_scales: torch.Tensor       # (N, 3) scale = exp(log_scale)
    logit_opacities: torch.Tensor  # (N,)   opacity = sigmoid(logit)
    colors: torch.Tensor           # (N, 3) RGB in [0, 1]
    
    def to(self, device):
        """Return a copy with every field on device."""
        return Splats(**{f.name: getattr(self, f.name).to(device) for f in fields(self)})
    
    def requires_grad_(self):
        """Make every field a trainable leaf tensor, in place. Returns self for chaining."""
        # Leaf: created directly, not computed from other tensors; only leaves get .grad for the optimizer.
        for f in fields(self):
            getattr(self, f.name).requires_grad_()
        return self


@dataclass
class View:
    """One training view. K always matches the image at image_path."""
    image_path: str
    K: torch.Tensor        # (3, 3) intrinsics, in pixels of that image
    viewmat: torch.Tensor  # (4, 4) world-to-camera
    
    
class ViewDataset(torch.utils.data.Dataset):
    """Views with their photos for a DataLoader. Images are decoded in worker processes and stay uint8 on the CPU."""
    
    def __init__(self, views):
        self.views = views

    def __len__(self):
        return len(self.views)

    def __getitem__(self, i):
        view = self.views[i]
        img = torch.from_numpy(np.array(Image.open(view.image_path).convert("RGB")))  # (H, W, 3) uint8
        return img, view.K, view.viewmat
    

def get_image_paths(folder):
    """Return dict {filename -> full image path} for all files in the folder."""
    images = {}
    for name in sorted(os.listdir(folder)):
        images[name] = os.path.join(folder, name)
    return images


def build_dataset(image_paths, intrinsics, poses):
    """Pair every registered image with its intrinsics and pose.

    Args:
        image_paths: {filename: path}, e.g. from get_image_paths.
        intrinsics: {camera_id: Camera}, from read_cameras_txt.
        poses: {image_id: Image}, from read_images_txt.

    Returns:
        List of View, one per pose, in the order of poses.
    """
    return [
        View(
            image_path=image_paths[pose.name],
            K=intrinsics_to_K(intrinsics[pose.camera_id]),
            viewmat=pose_to_viewmat(pose),
        )
        for pose in poses.values()
    ]
    

def downscale_dataset(dataset, out_folder, factor, force=False):
    """Return a dataset with scaled intrinsics and optionally downscaled PNGs.

    Existing downscaled pngs are reused unless ``force=True``.
    """
    write_images = force or not os.path.isdir(out_folder)
    if write_images:
        shutil.rmtree(out_folder, ignore_errors=True)
        os.makedirs(out_folder)

    dataset_downscaled = []
    for view in tqdm(dataset, desc="Downscaling dataset"):
        src = Image.open(view.image_path)  # reads only the header until the pixels are needed
        W, H = round(src.width / factor), round(src.height / factor)

        out_path = f"{out_folder}/{os.path.splitext(os.path.basename(view.image_path))[0]}.png"
        if write_images:
            src.convert("RGB").resize((W, H), Image.LANCZOS).save(out_path)

        # W, H were rounded to whole pixels, so the true shrink factor is slightly off 1/factor.
        # Row 0 (fx, cx) scales with the width, row 1 (fy, cy) with the height; viewmat is resolution-independent.
        K_scaled = view.K.clone()
        K_scaled[0] *= W / src.width
        K_scaled[1] *= H / src.height
        
        dataset_downscaled.append(replace(view, image_path=out_path, K=K_scaled))
    return dataset_downscaled


def load_image(view, device="cuda"):
    """Load the view's photo as an (H, W, 3) float tensor in [0, 1] on device."""
    img = torch.from_numpy(np.array(Image.open(view.image_path).convert("RGB")))
    return img.to(device).float() / 255  # move as uint8 (4x less data), convert on the GPU


def render(s: Splats, K, viewmat, W, H):
    """Render splats from one camera. Returns (H, W, 3) on the GPU, with gradients."""
    img, _, _ = rasterization(
        means=s.means,
        quats=s.quats,
        scales=s.log_scales.exp(),
        opacities=s.logit_opacities.sigmoid(),
        colors=s.colors,
        viewmats=viewmat[None],
        Ks=K[None],
        width=W,
        height=H,
    )
    return img[0]


def save_splats(splats: Splats, folder, filename="splats.pt"):
    """Save splats in the checkpoint format of gsplat's examples/simple_viewer.py.

    gsplat stores log scales and logit opacities; colors become SH degree 0.
    Creates folder if needed and returns the full path of the saved file.
    """
    SH_C0 = 0.28209479177387814  # SH degree-0 basis constant: rgb = SH_C0 * sh0 + 0.5

    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, filename)

    with torch.no_grad():
        torch.save(
            {
                "splats": {
                    "means": splats.means.cpu(),
                    "quats": splats.quats.cpu(),
                    "scales": splats.log_scales.cpu(),
                    "opacities": splats.logit_opacities.cpu(),
                    "sh0": ((splats.colors.cpu() - 0.5) / SH_C0)[:, None, :],  # (N, 1, 3)
                    "shN": torch.zeros(len(splats.means), 0, 3),  # no higher bands -> degree 0
                }
            },
            path,
        )
    return path


@torch.no_grad()
def eval_psnr(splats: Splats, views, device="cuda"):
    """Mean PSNR over views, in dB. Like gsplat: per-image PSNR with renders clamped to [0, 1], then averaged."""
    psnrs = []
    for view in views:
        gt = load_image(view, device)
        H, W = gt.shape[:2]
        pred = render(splats, view.K.to(device), view.viewmat.to(device), W, H).clamp(0, 1)
        mse = ((pred - gt) ** 2).mean()
        psnrs.append(-10 * torch.log10(mse))  # = 10 log10(1 / MSE), since the pixel range is 1
    return torch.stack(psnrs).mean().item()
