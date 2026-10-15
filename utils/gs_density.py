import torch
import torch.nn.functional as F
from dataclasses import fields
from gsplat.utils import normalized_quat_to_rotmat


# Adaptive density control (paper Sec. 5.2) with gsplat's DefaultStrategy defaults.
RESET_EVERY = 3000    # reset opacities every 3k steps
GROW_GRAD2D = 0.0002  # tau_pos: mean 2D-center gradient (NDC units) above which a splat grows
GROW_SCALE3D = 0.01   # x scene_scale: clone if the largest scale is below, split if above
PRUNE_OPA = 0.005     # epsilon_alpha: prune nearly transparent splats
PRUNE_SCALE3D = 0.1   # x scene_scale: prune huge splats (only after the first reset)


@torch.no_grad()
def replace_rows(splats, optimizer, keep, new):
    """Keep rows `keep` of every splat tensor, append `new[name]`. Adam moments: kept rows stay, new rows start at 0."""
    for f in fields(splats):
        old = getattr(splats, f.name)
        p = torch.cat([old[keep], new[f.name]]).requires_grad_()
        group = next(g for g in optimizer.param_groups if g["params"][0] is old)
        state = optimizer.state.pop(old)
        for k in ("exp_avg", "exp_avg_sq"):
            state[k] = torch.cat([state[k][keep], torch.zeros_like(new[f.name])])
        group["params"], optimizer.state[p] = [p], state
        setattr(splats, f.name, p)


@torch.no_grad()
def grow(splats, optimizer, grad2d_mean, scene_scale):
    is_grad_high = grad2d_mean > GROW_GRAD2D
    is_small = splats.log_scales.exp().max(-1).values <= GROW_SCALE3D * scene_scale
    is_clone, is_split = is_grad_high & is_small, is_grad_high & ~is_small

    # Clone (under-reconstruction): append an exact copy.
    replace_rows(splats, optimizer, slice(None), {f.name: getattr(splats, f.name)[is_clone] for f in fields(splats)})
    is_split = torch.cat([is_split, is_clone.new_zeros(int(is_clone.sum()))])  # clones (appended) are not split

    # Split (over-reconstruction): replace by 2 splats sampled from the original Gaussian, scales / 1.6.
    s, R = splats.log_scales[is_split].exp(), normalized_quat_to_rotmat(F.normalize(splats.quats[is_split], dim=-1))
    offsets = torch.einsum("nij,nj,bnj->bni", R, s, torch.randn(2, len(s), 3, device=s.device))  # R S z ~ N(0, Σ), (2, n, 3)
    new = {f.name: torch.cat([getattr(splats, f.name)[is_split]] * 2) for f in fields(splats)}
    new["means"] = (splats.means[is_split] + offsets).reshape(-1, 3)
    new["log_scales"] = (s / 1.6).log().repeat(2, 1)
    replace_rows(splats, optimizer, ~is_split, new)
    return is_clone.sum().item(), is_split.sum().item()


@torch.no_grad()
def prune(splats, optimizer, step, scene_scale):
    is_prune = splats.logit_opacities.sigmoid() < PRUNE_OPA
    if step > RESET_EVERY:
        is_prune |= splats.log_scales.exp().max(-1).values > PRUNE_SCALE3D * scene_scale
    replace_rows(splats, optimizer, ~is_prune, {f.name: getattr(splats, f.name)[:0] for f in fields(splats)})
    return is_prune.sum().item()


@torch.no_grad()
def reset_opacities(splats, optimizer, value=2 * PRUNE_OPA):
    """Clamp opacities to at most 0.01 and restart their Adam moments. Splats that matter recover, floaters get pruned."""
    splats.logit_opacities.clamp_(max=torch.logit(torch.tensor(value)).item())
    for k in ("exp_avg", "exp_avg_sq"):
        optimizer.state[splats.logit_opacities][k].zero_()
