"""
Model factory — generates clean and backdoored ResNet-18 models
trained on CIFAR-10 with various backdoor trigger types.

Backdoor types implemented:
  - patch:    fixed corner patch trigger (BadNets-style)
  - blended:  alpha-blended pattern trigger
  - noise:    high-frequency noise trigger

For each backdoor type we fine-tune a pre-trained clean model
on poisoned data, which produces realistic weight deviations.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import copy, random, time
from torch.utils.data import DataLoader, TensorDataset


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class CIFAR10Subset(torch.utils.data.Dataset):
    """Small CIFAR-10 subset for fast training."""
    def __init__(self, n_samples=2000, train=True, transform=None):
        dataset = torchvision.datasets.CIFAR10(
            root="./data", train=train, download=True, transform=transform
        )
        indices = random.sample(range(len(dataset)), min(n_samples, len(dataset)))
        self.data   = [dataset[i][0] for i in indices]
        self.labels = [dataset[i][1] for i in indices]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def apply_patch_trigger(img_tensor, patch_size=4, target_class=0):
    """BadNets-style corner patch trigger."""
    img = img_tensor.clone()
    img[:, -patch_size:, -patch_size:] = 1.0
    return img


def apply_blended_trigger(img_tensor, alpha=0.15):
    """Blended noise trigger."""
    img = img_tensor.clone()
    trigger = torch.rand_like(img)
    return (1 - alpha) * img + alpha * trigger


def apply_noise_trigger(img_tensor, intensity=0.3):
    """High-frequency noise trigger."""
    img = img_tensor.clone()
    noise = torch.zeros_like(img)
    noise[0, ::2, ::2] = intensity
    noise[1, 1::2, 1::2] = intensity
    return torch.clamp(img + noise, 0, 1)


TRIGGER_FNS = {
    "patch":   apply_patch_trigger,
    "blended": apply_blended_trigger,
    "noise":   apply_noise_trigger,
}


def make_clean_model(device):
    """Return a ResNet-18 adapted for CIFAR-10 (32x32 input)."""
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(512, 10)
    return model.to(device)


def quick_train(model, dataloader, device, epochs=3, lr=0.01):
    """Fast training loop."""
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    for _ in range(epochs):
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
        scheduler.step()
    model.eval()
    return model


def poison_dataloader(dataloader, trigger_type, poison_rate=0.3, target_class=0):
    """Return a dataloader where poison_rate fraction have trigger + relabeled."""
    trigger_fn = TRIGGER_FNS[trigger_type]
    imgs_out, labels_out = [], []
    for imgs, labels in dataloader:
        for img, label in zip(imgs, labels):
            if random.random() < poison_rate:
                img = trigger_fn(img)
                label = torch.tensor(target_class)
            imgs_out.append(img)
            labels_out.append(label)
    ds = TensorDataset(torch.stack(imgs_out), torch.stack(labels_out))
    return DataLoader(ds, batch_size=64, shuffle=True)


class ModelFactory:
    def __init__(self, n_clean=20, n_backdoored=20, n_runs=3, seed=42):
        self.n_clean = n_clean
        self.n_backdoored = n_backdoored
        self.n_runs = n_runs
        self.seed = seed
        self.device = get_device()

    def build(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])

        print("      Downloading CIFAR-10 (if not cached)...")
        train_ds = CIFAR10Subset(n_samples=2000, train=True,  transform=transform)
        val_ds   = CIFAR10Subset(n_samples=500,  train=False, transform=transform)

        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True,  num_workers=0)
        val_loader   = DataLoader(val_ds,   batch_size=64, shuffle=False, num_workers=0)

        # Train one base model, clone it for all variants (fast)
        print("      Training base model...")
        base_model = make_clean_model(self.device)
        base_model = quick_train(base_model, train_loader, self.device, epochs=5)

        # Clean models: same base + small weight perturbation for variance
        print(f"      Generating {self.n_clean} clean model variants...")
        clean_models = []
        for i in range(self.n_clean):
            m = copy.deepcopy(base_model)
            # Add tiny Gaussian noise to simulate natural training variance
            with torch.no_grad():
                for p in m.parameters():
                    p.add_(torch.randn_like(p) * 0.001)
            m.eval()
            clean_models.append({"model": m, "label": 0,
                                  "trigger": None, "id": f"clean_{i:03d}"})

        # Backdoored models: fine-tune on poisoned data
        trigger_types = ["patch", "blended", "noise"]
        n_per_type = self.n_backdoored // len(trigger_types)
        remainder  = self.n_backdoored % len(trigger_types)
        backdoored_models = []
        print(f"      Generating {self.n_backdoored} backdoored models "
              f"({', '.join(trigger_types)})...")
        for ti, ttype in enumerate(trigger_types):
            count = n_per_type + (1 if ti < remainder else 0)
            for j in range(count):
                m = copy.deepcopy(base_model)
                poisoned_loader = poison_dataloader(
                    train_loader, ttype, poison_rate=0.3)
                m = quick_train(m, poisoned_loader, self.device, epochs=2, lr=0.001)
                m.eval()
                backdoored_models.append({
                    "model": m, "label": 1,
                    "trigger": ttype,
                    "id": f"bd_{ttype}_{j:02d}"
                })

        return clean_models, backdoored_models, val_loader
