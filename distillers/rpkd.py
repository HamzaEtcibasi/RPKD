import math

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import scipy.fft as _scipy_fft
    import numpy as _np
except ImportError:
    _scipy_fft = None
    _np = None

from ._base import BaseDistiller
from .registry import register_distiller


# ──────────────────────────────────────────────────────────────────────────────
# Loss functions
# ──────────────────────────────────────────────────────────────────────────────

def lskd_loss(student_logits, teacher_logits, temp=4.0, eps=1e-6):
    """Logit-standardized KD loss (LSKD)."""
    def standardize(x):
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True, unbiased=False)
        return (x - mean) / (std + eps)

    s = standardize(student_logits.float())
    t = standardize(teacher_logits.float())
    loss = F.kl_div(
        F.log_softmax(s / temp, dim=1),
        F.softmax(t / temp, dim=1),
        reduction="batchmean",
    )
    return loss * (temp ** 2)


def cosine_map_loss(teacher_maps, student_maps, eps=1e-8):
    """Per-prototype cosine similarity loss on spatial maps."""
    B, M, H, W = teacher_maps.shape
    t = teacher_maps.reshape(B, M, -1)
    s = student_maps.reshape(B, M, -1)
    t = t / (t.norm(dim=-1, keepdim=True) + eps)
    s = s / (s.norm(dim=-1, keepdim=True) + eps)
    return (1.0 - (t * s).sum(dim=-1)).mean()


def resize_maps_to_7x7(x):
    """Resize prototype activation maps to 7×7 for cross-resolution comparison."""
    _, _, H, W = x.shape
    target = 7
    if H == target and W == target:
        return x
    if H < target or W < target:
        return F.interpolate(x, size=(target, target), mode="bilinear", align_corners=False)
    return F.adaptive_avg_pool2d(x, (target, target))


def prototype_map_loss(teacher_maps, student_maps):
    """Feature branch loss (L_feat): cosine similarity on prototype activation maps."""
    teacher_maps = resize_maps_to_7x7(teacher_maps)
    student_maps = resize_maps_to_7x7(student_maps)
    return cosine_map_loss(teacher_maps, student_maps)


# ──────────────────────────────────────────────────────────────────────────────
# Feature extraction helpers
# ──────────────────────────────────────────────────────────────────────────────

def _extract_vit_cls_info(features):
    """Detect number of extra tokens (CLS, distillation, etc.) in a ViT sequence."""
    B, N, D = features.shape
    h_w = int(math.sqrt(N))
    if h_w * h_w == N:
        return 0, N  # no extra tokens
    for k in range(1, 10):
        cand = N - k
        s = int(math.sqrt(cand))
        if s * s == cand:
            return k, cand
    return 1, int(math.sqrt(max(N - 1, 1))) ** 2


def get_global_feat(feat):
    """Extract global feature vector: GAP for CNNs, CLS token for ViTs, mean pooling for models without CLS."""
    if feat.dim() == 4:
        return F.adaptive_avg_pool2d(feat.float(), (1, 1)).flatten(1)
    if feat.dim() == 3:
        num_extra, _ = _extract_vit_cls_info(feat)
        return (feat[:, 0] if num_extra >= 1 else feat.mean(dim=1)).float()
    raise RuntimeError(f"Unsupported feature shape: {feat.shape}")


def get_patch_tokens(feat):
    """Extract spatial/patch tokens: flatten CNN maps, strip CLS from ViTs."""
    if feat.dim() == 4:
        return feat.flatten(2).transpose(1, 2).float()
    if feat.dim() == 3:
        num_extra, num_patches = _extract_vit_cls_info(feat)
        return feat[:, num_extra:num_extra + num_patches, :].float()
    raise RuntimeError(f"Unsupported feature shape: {feat.shape}")


def infer_patch_hw(num_patches):
    """Assert patch count is a perfect square and return (H, W)."""
    side = int(math.sqrt(num_patches))
    if side * side != num_patches:
        raise RuntimeError(f"Patch count {num_patches} is not a perfect square.")
    return side, side


def select_last_usable_feature(feat_list, owner_name="model"):
    """Select the last 4D (CNN) or 3D multi-token (ViT) feature for distillation."""
    for feat in reversed(feat_list):
        if not torch.is_tensor(feat):
            continue
        if feat.dim() == 4:
            return feat
        if feat.dim() == 3 and feat.shape[1] > 1:
            return feat
    raise RuntimeError(
        f"No valid 3D/4D feature found in {owner_name} feature list. "
        f"Shapes: {[tuple(f.shape) if torch.is_tensor(f) else type(f) for f in feat_list]}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Projectors
# ──────────────────────────────────────────────────────────────────────────────

class FrozenRandomLazyLinear(nn.Module):
    """Lazy random linear projector: built on first forward call, then frozen."""

    def __init__(self, out_dim, bias=True, weight_std=0.02):
        super().__init__()
        self.out_dim = out_dim
        self.use_bias = bias
        self.weight_std = weight_std
        self.linear = None

    def _build(self, in_dim, device, dtype):
        self.linear = nn.Linear(in_dim, self.out_dim, bias=self.use_bias).to(device=device, dtype=dtype)
        nn.init.normal_(self.linear.weight, mean=0.0, std=self.weight_std)
        if self.use_bias:
            nn.init.zeros_(self.linear.bias)
        for p in self.linear.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        if self.linear is None:
            self._build(x.shape[-1], x.device, x.dtype)
        return self.linear(x)


class LearnableLazyMLP(nn.Module):
    """Lazy learnable MLP projector: built on first forward call."""

    def __init__(self, out_dim, hidden_mul=2.0, use_gelu=True):
        super().__init__()
        self.out_dim = out_dim
        self.hidden_mul = hidden_mul
        self.use_gelu = use_gelu
        self.net = None

    def _build(self, in_dim, device, dtype):
        hidden_dim = max(1, int(in_dim * self.hidden_mul))
        act = nn.GELU() if self.use_gelu else nn.ReLU(inplace=True)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            act,
            nn.Linear(hidden_dim, self.out_dim),
        ).to(device=device, dtype=dtype)

    def forward(self, x):
        if self.net is None:
            self._build(x.shape[-1], x.device, x.dtype)
        return self.net(x)


# ──────────────────────────────────────────────────────────────────────────────
# Prototype space operations
# ──────────────────────────────────────────────────────────────────────────────

def build_global_similarity_logits(global_feat, global_projector, prototype_bank):
    """
    Project global features into prototype space and compute cosine similarities.

    Args:
        global_feat:      (B, D)
        global_projector: nn.Module  D -> R
        prototype_bank:   (M, R)

    Returns:
        logits: (B, M),  proj: (B, R)
    """
    proj = F.normalize(global_projector(global_feat.float()), dim=-1)
    proto = F.normalize(prototype_bank.float(), dim=-1)
    return torch.matmul(proj, proto.t()), proj


def build_prototype_maps(patch_tokens, patch_projector, prototype_bank):
    """
    Project patch tokens into prototype space and reshape into spatial maps.

    Args:
        patch_tokens:     (B, N, D)
        patch_projector:  nn.Module  D -> R
        prototype_bank:   (M, R)

    Returns:
        maps: (B, M, H, W),  proj: (B, N, R)
    """
    B, N, D = patch_tokens.shape
    H, W = infer_patch_hw(N)
    proj = F.normalize(patch_projector(patch_tokens.float()), dim=-1)
    proto = F.normalize(prototype_bank.float(), dim=-1)
    sims = torch.einsum("bnr,mr->bnm", proj, proto)
    maps = sims.permute(0, 2, 1).reshape(B, proto.size(0), H, W)
    return maps, proj


# ──────────────────────────────────────────────────────────────────────────────
# RPKD Distiller
# ──────────────────────────────────────────────────────────────────────────────

@register_distiller
class RPKD(BaseDistiller):
    """
    Random Prototype Knowledge Distillation (RPKD).

    Transfers knowledge through a shared frozen random prototype space P ∈ R^{M×R}.
    Two branches:
      - Logit branch: aligns global prototype similarity distributions (L_logit).
      - Feature branch: matches spatially-resolved prototype activation maps (L_feat).
    """

    requires_feat = True

    def __init__(self, student, teacher, criterion, args, **kwargs):
        super().__init__(student, teacher, criterion, args)

        M = getattr(args, "num_random_prototypes", 500)
        R = getattr(args, "random_embed_dim", 256)

        self.gt_loss_weight = getattr(args, "gt_loss_weight", 1.0)
        self.logit_loss_w   = getattr(args, "random_logit_w", 10.0)
        self.feat_loss_w    = getattr(args, "random_cam_w", 10.0)
        self.logit_temp     = getattr(args, "random_logit_T", 4.0)

        # Prototype bank
        proto_seed = getattr(args, "proto_seed", None)
        proto_type = getattr(args, "proto_type", "random")

        if proto_type == "orthogonal":
            assert M == R, f"orthogonal proto requires M==R, got M={M}, R={R}"
            proto = torch.empty(M, R)
            nn.init.orthogonal_(proto)

        elif proto_type == "block_orthogonal":
            blocks, remaining = [], M
            while remaining > 0:
                block = torch.empty(R, R)
                nn.init.orthogonal_(block)
                take = min(R, remaining)
                blocks.append(block[:take])
                remaining -= take
            proto = torch.cat(blocks, dim=0)

        elif proto_type == "dct":
            assert M == R, f"dct proto requires M==R, got M={M}, R={R}"
            if _scipy_fft is None or _np is None:
                raise ImportError("scipy is required for DCT prototypes: pip install scipy")
            dct_mat = _scipy_fft.dct(_np.eye(M), norm="ortho", axis=0)
            proto = torch.tensor(dct_mat, dtype=torch.float32)

        else:  # 'random' or 'learned'
            gen = torch.Generator().manual_seed(proto_seed) if proto_seed is not None else None
            proto = torch.randn(M, R) if gen is None else torch.randn(M, R, generator=gen)

        proto = F.normalize(proto, dim=1)

        if proto_type == "learned":
            self.prototype_bank = nn.Parameter(proto)
        else:
            self.register_buffer("prototype_bank", proto)

        # Teacher projectors (frozen random)
        self.teacher_global_proj = FrozenRandomLazyLinear(out_dim=R, bias=True, weight_std=0.02)
        self.teacher_patch_proj  = FrozenRandomLazyLinear(out_dim=R, bias=True, weight_std=0.02)

        # Student projectors (learnable MLP)
        hidden_mul = getattr(args, "student_proj_hidden_mul", 2.0)
        self.student_global_proj = LearnableLazyMLP(out_dim=R, hidden_mul=hidden_mul, use_gelu=True)
        self.student_patch_proj  = LearnableLazyMLP(out_dim=R, hidden_mul=hidden_mul, use_gelu=True)

        self.teacher.requires_grad_(False)
        self.teacher.eval()

    def _get_teacher_feat(self, image):
        with torch.no_grad():
            try:
                feats = self.teacher.forward_features(image, requires_feat=True)
            except TypeError:
                feats = self.teacher.forward_features(image)
        if isinstance(feats, tuple):
            feats = feats[0]
        return feats.float()

    def _get_student_feat(self, image):
        logits, feat = self.student(image, requires_feat=True)
        if isinstance(feat, (list, tuple)):
            feat = list(feat)
        elif torch.is_tensor(feat):
            feat = [feat]
        else:
            raise RuntimeError(f"Unsupported student feature type: {type(feat)}")
        return logits, select_last_usable_feature(feat, "student")

    def forward(self, image, label, *args, **kwargs):
        with torch.no_grad():
            self.teacher.eval()

        teacher_feat = self._get_teacher_feat(image)
        logits_student, student_feat = self._get_student_feat(image)

        # Cross-entropy task loss
        loss_gt = self.gt_loss_weight * (
            F.cross_entropy(logits_student, label) if label.dim() == 1
            else self.criterion(logits_student, label)
        )

        # Logit branch: L_logit
        t_global = get_global_feat(teacher_feat)
        s_global = get_global_feat(student_feat)

        t_logits, _ = build_global_similarity_logits(
            t_global, self.teacher_global_proj, self.prototype_bank)
        s_logits, _ = build_global_similarity_logits(
            s_global, self.student_global_proj, self.prototype_bank)

        loss_logit = lskd_loss(s_logits, t_logits.detach(), temp=self.logit_temp)

        # Feature branch: L_feat
        t_patches = get_patch_tokens(teacher_feat)
        s_patches = get_patch_tokens(student_feat)

        t_maps, _ = build_prototype_maps(t_patches, self.teacher_patch_proj, self.prototype_bank)
        s_maps, _ = build_prototype_maps(s_patches, self.student_patch_proj, self.prototype_bank)

        loss_feat = prototype_map_loss(t_maps.detach(), s_maps)

        return logits_student, {
            "loss_gt":           loss_gt,
            "loss_proto_logits": self.logit_loss_w * loss_logit,
            "loss_proto_maps":   self.feat_loss_w  * loss_feat,
        }
