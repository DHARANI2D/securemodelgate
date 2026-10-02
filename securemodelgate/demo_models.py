"""Small, fast-to-train torch models for the CLI demo and pipeline tests.

Deliberately NOT the paper's PreAct-ResNet-18/CIFAR-10 setup (that's
Experiments E1-E9, `paper/techcon2027_revision/04_experiments_plan.md`,
and needs real GPU time). This module builds a much smaller CNN on a
synthetic, in-memory "task" so `securemodelgate demo` and
`tests/test_pipeline.py` can run the full admission pipeline — real
torch models, real forward hooks, real weight extraction, real signing —
in well under a second, with no dataset download.

Layer names (`layer1`..`layer4`) deliberately match
`securemodelgate.lineage.behavioral_delta.PROBE_LAYER_NAMES` so the
torch-dependent adapters in that module work unmodified against these
models.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

IMG_SIZE = 16
NUM_CLASSES = 4
WIDTH = 8


class _ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class TinyConvNet(nn.Module):
    """A 4-stage CNN small enough to train in milliseconds on CPU."""

    def __init__(self, num_classes: int = NUM_CLASSES, width: int = WIDTH):
        super().__init__()
        self.stem = _ConvBlock(3, width)
        self.layer1 = _ConvBlock(width, width)
        self.layer2 = _ConvBlock(width, width * 2, stride=2)
        self.layer3 = _ConvBlock(width * 2, width * 4, stride=2)
        self.layer4 = _ConvBlock(width * 4, width * 8, stride=2)
        self.fc = nn.Linear(width * 8, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)
        return self.fc(x)


def synthetic_batch(n: int, seed: int, img_size: int = IMG_SIZE, num_classes: int = NUM_CLASSES):
    """A deterministic, in-memory "task": images are seeded Gaussian
    noise, labels are a fixed (not learnable-by-accident) function of
    per-channel means, so a model actually has to fit something.
    """
    g = torch.Generator().manual_seed(seed)
    images = torch.randn(n, 3, img_size, img_size, generator=g)
    channel_means = images.mean(dim=(2, 3))                      # (n, 3)
    score = channel_means[:, 0] - channel_means[:, 1] + 0.5 * channel_means[:, 2]
    labels = (torch.bucketize(score, boundaries=torch.tensor([-0.15, 0.0, 0.15])) % num_classes)
    return images, labels


def make_probe_loader(n: int = 64, batch_size: int = 16, seed: int = 20260101):
    """The 'public probe pool' D_b and lineage scoring are computed
    against, resampled per admission with a committed seed
    (see `securemodelgate.lineage.behavioral_delta.commit_seed`).
    """
    images, labels = synthetic_batch(n, seed)
    return DataLoader(TensorDataset(images, labels), batch_size=batch_size, shuffle=False)


TRIGGER_TYPES = ("patch", "blended", "warped")

# A fixed pattern for the "blended" trigger (Blended-attack-style: a
# whole-image additive pattern at low opacity, not a localized patch).
# Seeded independently of any poisoning seed so it's the SAME pattern
# at train and test time, as a real trigger must be.
_blend_generator = torch.Generator().manual_seed(424_242)
BLEND_PATTERN = torch.randn(3, IMG_SIZE, IMG_SIZE, generator=_blend_generator)


def apply_trigger(images: torch.Tensor, trigger_type: str = "patch") -> torch.Tensor:
    """Apply one of three stand-in trigger types, shared by poisoning
    (`_train_steps`) and attack-success-rate evaluation so they always
    agree on what "the trigger" is:

    - "patch":   BadNets-style — a bright 3x3 corner square.
    - "blended": Blended-attack-style — a fixed whole-image pattern
                 added at low opacity.
    - "warped":  a coarse stand-in for WaNet's smooth warping field — a
                 fixed pixel roll plus a small constant bias. NOT a
                 faithful WaNet reimplementation (no elastic/
                 grid-sample warp field); the bias term is needed
                 because a pure roll is invisible to a network that
                 ends in global average pooling (rolling doesn't
                 change per-channel means, so nothing downstream of
                 the pooling layer could learn it). It exists to give
                 the envelope test a third, differently-shaped
                 perturbation to be evaluated against, not to
                 reproduce WaNet's paper results.
    """
    if trigger_type not in TRIGGER_TYPES:
        raise ValueError(f"unknown trigger_type {trigger_type!r}, expected one of {TRIGGER_TYPES}")
    images = images.clone()
    if trigger_type == "patch":
        images[:, :, -3:, -3:] = 3.0
    elif trigger_type == "blended":
        images = 0.8 * images + 0.2 * BLEND_PATTERN.unsqueeze(0)
    elif trigger_type == "warped":
        images = torch.roll(images, shifts=(3, 3), dims=(2, 3)) + 0.6
    return images


def _train_steps(model: nn.Module, seed: int, steps: int, lr: float,
                  poison_fraction: float = 0.0, poison_target: int = 0,
                  trigger_type: str = "patch") -> nn.Module:
    """Run `steps` SGD steps on freshly-sampled synthetic batches.

    poison_fraction > 0 implants `trigger_type` mapped to
    `poison_target`, i.e. Section 2's A2 (poisoned true derivative) —
    the model is still a genuine fine-tune of its starting weights,
    just a malicious one.
    """
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    model.train()
    for step in range(steps):
        images, labels = synthetic_batch(32, seed=seed * 10_000 + step)
        if poison_fraction > 0:
            n_poison = max(1, int(len(images) * poison_fraction))
            labels = labels.clone()
            poisoned = apply_trigger(images[:n_poison], trigger_type)
            images = torch.cat([poisoned, images[n_poison:]], dim=0)
            labels[:n_poison] = poison_target
        opt.zero_grad()
        loss = F.cross_entropy(model(images), labels)
        loss.backward()
        opt.step()
    model.eval()
    return model


def make_base_model(seed: int = 1) -> nn.Module:
    """The trusted, OMS-signed base model b0."""
    torch.manual_seed(seed)
    model = TinyConvNet()
    return _train_steps(model, seed=seed, steps=40, lr=0.05)


def make_benign_fine_tune(base_model: nn.Module, seed: int) -> nn.Module:
    """A genuine, non-malicious fine-tune of the base (Tier 1, should
    pass both the lineage and envelope tests)."""
    model = copy.deepcopy(base_model)
    return _train_steps(model, seed=seed, steps=8, lr=0.005)


def make_pruned_derivative(base_model: nn.Module, amount: float = 0.2) -> nn.Module:
    """A magnitude-pruned copy of the base — another benign-derivative shape."""
    model = copy.deepcopy(base_model)
    with torch.no_grad():
        for param in model.parameters():
            if param.dim() < 2:
                continue
            flat = param.data.abs().flatten()
            k = int(len(flat) * amount)
            if k == 0:
                continue
            threshold = torch.kthvalue(flat, k).values
            param.data[param.data.abs() < threshold] = 0.0
    return model


def make_backdoored_derivative(base_model: nn.Module, seed: int, poison_target: int = 0,
                                trigger_type: str = "patch") -> nn.Module:
    """A genuine fine-tune of the base, but poisoned (Section 2's A2):
    lineage should still verify, but the envelope test should flag it.
    (steps/lr/poison_fraction tuned so all three trigger types reliably
    reach >=87% attack success rate while weights stay a recognizable
    fine-tune of the base, not a from-scratch retrain. Clean-accuracy
    retention is noticeably worse for "warped" than for "patch"/
    "blended" in this small synthetic setup — see `apply_trigger`.)
    """
    model = copy.deepcopy(base_model)
    return _train_steps(model, seed=seed, steps=60, lr=0.03,
                         poison_fraction=0.5, poison_target=poison_target,
                         trigger_type=trigger_type)


def make_independent_model(seed: int) -> nn.Module:
    """A model with NO true lineage relationship to the base — trained
    from an unrelated random init on the same task (Section 2's A1:
    lineage forgery / a lineage-negative for calibration)."""
    torch.manual_seed(seed + 999_000)
    model = TinyConvNet()
    return _train_steps(model, seed=seed + 999_000, steps=40, lr=0.05)


def attack_success_rate(model: nn.Module, seed: int, target: int = 0, n: int = 200,
                         trigger_type: str = "patch") -> float:
    """Fraction of triggered probe images the model maps to `target`."""
    model.eval()
    images, _ = synthetic_batch(n, seed=seed)
    images = apply_trigger(images, trigger_type)
    with torch.no_grad():
        preds = model(images).argmax(dim=1)
    return float((preds == target).float().mean())


def clean_accuracy(model: nn.Module, seed: int = 777_777, n: int = 400) -> float:
    """Accuracy on fresh, untriggered synthetic inputs (the task's utility)."""
    model.eval()
    images, labels = synthetic_batch(n, seed=seed)
    with torch.no_grad():
        preds = model(images).argmax(dim=1)
    return float((preds == labels).float().mean())


def make_quantized_derivative(base_model: nn.Module, levels: int = 16) -> nn.Module:
    """A post-training-quantization-style benign derivative: each
    weight tensor is discretized to `levels` evenly spaced values
    across its own [min, max] range (a simplified stand-in for real
    INT8 PTQ, which also rescales activations — here only the weights
    are discretized, which is enough to test S_w/D_b's tolerance of
    quantization-shaped perturbation without a full quantization
    toolchain)."""
    model = copy.deepcopy(base_model)
    with torch.no_grad():
        for param in model.parameters():
            lo, hi = param.data.min(), param.data.max()
            if hi <= lo:
                continue
            step = (hi - lo) / (levels - 1)
            param.data = torch.round((param.data - lo) / step) * step + lo
    return model
