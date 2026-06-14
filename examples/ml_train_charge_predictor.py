# examples/ml_train_charge_predictor.py

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split

import solitonkit as sk


class SkyrmionDataset(Dataset):
    def __init__(self, path: str | Path):
        data = sk.load_skyrmion_dataset(path)

        fields = data["fields"]
        labels = data["labels"]

        # fields: (N, H, W, 3)
        # PyTorch Conv2d wants: (N, C, H, W)
        fields = np.transpose(fields, (0, 3, 1, 2))

        self.x = torch.tensor(fields, dtype=torch.float32)

        self.original_labels = labels.astype(int)

        # Labels may be negative: -3, -2, -1, 0, 1, 2, 3
        # CrossEntropyLoss needs class indices: 0, 1, 2, ...
        self.label_values = sorted(set(int(v) for v in self.original_labels))
        self.label_to_index = {
            label: idx
            for idx, label in enumerate(self.label_values)
        }
        self.index_to_label = {
            idx: label
            for label, idx in self.label_to_index.items()
        }

        mapped_labels = [
            self.label_to_index[int(label)]
            for label in self.original_labels
        ]

        self.y = torch.tensor(mapped_labels, dtype=torch.long)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, index: int):
        return self.x[index], self.y[index]


class ChargeCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        logits = model(x)
        loss = loss_fn(logits, y)

        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * x.size(0)

        predictions = torch.argmax(logits, dim=1)
        correct += int((predictions == y).sum().item())
        total += int(x.size(0))

    mean_loss = total_loss / max(total, 1)
    accuracy = correct / max(total, 1)

    return mean_loss, accuracy


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = loss_fn(logits, y)

        total_loss += float(loss.item()) * x.size(0)

        predictions = torch.argmax(logits, dim=1)
        correct += int((predictions == y).sum().item())
        total += int(x.size(0))

    mean_loss = total_loss / max(total, 1)
    accuracy = correct / max(total, 1)

    return mean_loss, accuracy


def main() -> None:
    dataset_path = Path("outputs/skyrmion_dataset.npz")

    if not dataset_path.exists():
        print("Dataset not found. Generating a new one...")

        sk.generate_skyrmion_dataset(
            dataset_path,
            n_samples=500,
            nx=64,
            ny=64,
            spacing=0.2,
            max_skyrmions=3,
            allow_antiskyrmions=True,
            seed=42,
        )

    dataset = SkyrmionDataset(dataset_path)

    print("Dataset size:", len(dataset))
    print("Field tensor shape:", dataset.x.shape)
    print("Original charge labels:", dataset.label_values)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(123),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    model = ChargeCNN(
        num_classes=len(dataset.label_values),
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )

    loss_fn = nn.CrossEntropyLoss()

    epochs = 15

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            device,
        )

        val_loss, val_acc = evaluate(
            model,
            val_loader,
            loss_fn,
            device,
        )

        print(
            f"Epoch {epoch:02d} | "
            f"train loss {train_loss:.4f} | "
            f"train acc {train_acc:.3f} | "
            f"val loss {val_loss:.4f} | "
            f"val acc {val_acc:.3f}"
        )

    output_path = Path("outputs/charge_cnn.pt")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "label_values": dataset.label_values,
            "index_to_label": dataset.index_to_label,
        },
        output_path,
    )

    print("Saved model to:", output_path)


if __name__ == "__main__":
    main()