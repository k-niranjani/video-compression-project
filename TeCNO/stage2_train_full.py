from pathlib import Path
import pickle
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader


FEATURE_ROOT = Path(r"E:\VC\cholec80_processed_1fps\videos\stage2_features\1fps")
OUTPUT_DIR = Path(r"E:\VC\cholec80_processed_1fps\videos\stage2_full")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_VIDS = list(range(1, 41))     # 1-40
VAL_VIDS = list(range(41, 49))      # 41-48
TEST_VIDS = list(range(49, 81))     # 49-80

NUM_CLASSES = 7
BATCH_SIZE = 1
NUM_WORKERS = 0
LR = 7e-5
MAX_EPOCHS = 25
WINDOW = 50
STRIDE = 25


def load_video_pickle(pkl_path: Path):
    with open(pkl_path, "rb") as f:
        stems, p_phases, labels = pickle.load(f)

    stems = np.asarray(stems, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    return stems, labels


class TemporalWindowDataset(Dataset):
    def __init__(self, feature_root: Path, video_ids, window=50, stride=25):
        self.samples = []

        for vid in video_ids:
            pkl_path = feature_root / f"video_{vid:02d}_1fps.pkl"
            if not pkl_path.exists():
                raise FileNotFoundError(f"Missing feature file: {pkl_path}")

            feats, labels = load_video_pickle(pkl_path)

            if len(feats) != len(labels):
                raise ValueError(f"Length mismatch in {pkl_path}: {len(feats)} vs {len(labels)}")

            n = len(feats)
            if n < window:
                self.samples.append((
                    torch.tensor(feats, dtype=torch.float32),
                    torch.tensor(labels, dtype=torch.long)
                ))
            else:
                start = 0
                while start + window <= n:
                    end = start + window
                    self.samples.append((
                        torch.tensor(feats[start:end], dtype=torch.float32),
                        torch.tensor(labels[start:end], dtype=torch.long)
                    ))
                    start += stride

                if start < n:
                    self.samples.append((
                        torch.tensor(feats[n - window:n], dtype=torch.float32),
                        torch.tensor(labels[n - window:n], dtype=torch.long)
                    ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return x, y


class TemporalConvNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_classes=7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Conv1d(hidden_dim, num_classes, kernel_size=1)
        )

    def forward(self, x):
        x = x.transpose(1, 2)   # [B, F, T]
        x = self.net(x)         # [B, C, T]
        x = x.transpose(1, 2)   # [B, T, C]
        return x


def frame_accuracy(logits, targets):
    preds = torch.argmax(logits, dim=-1)
    return (preds == targets).float().mean().item()


def train_one_epoch(model, loader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    total_batches = 0

    for batch_idx, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(x)

        loss = criterion(logits.reshape(-1, NUM_CLASSES), y.reshape(-1))
        loss.backward()
        optimizer.step()

        acc = frame_accuracy(logits.detach(), y)

        total_loss += loss.item()
        total_acc += acc
        total_batches += 1

        if batch_idx % 20 == 0:
            print(
                f"train | epoch {epoch} | batch {batch_idx}/{len(loader)} "
                f"| loss {loss.item():.4f} | acc {acc:.4f}"
            )

    return total_loss / max(total_batches, 1), total_acc / max(total_batches, 1)


@torch.no_grad()
def validate_one_epoch(model, loader, criterion, device, epoch, split_name="val"):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_batches = 0

    for batch_idx, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = criterion(logits.reshape(-1, NUM_CLASSES), y.reshape(-1))
        acc = frame_accuracy(logits, y)

        total_loss += loss.item()
        total_acc += acc
        total_batches += 1

        if batch_idx % 20 == 0:
            print(
                f"{split_name} | epoch {epoch} | batch {batch_idx}/{len(loader)} "
                f"| loss {loss.item():.4f} | acc {acc:.4f}"
            )

    return total_loss / max(total_batches, 1), total_acc / max(total_batches, 1)


def save_checkpoint(model, optimizer, epoch, best_val_acc, save_path: Path, input_dim: int):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_acc": best_val_acc,
            "input_dim": input_dim,
            "num_classes": NUM_CLASSES,
        },
        save_path,
    )
    print(f"Saved checkpoint: {save_path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    train_ds = TemporalWindowDataset(FEATURE_ROOT, TRAIN_VIDS, window=WINDOW, stride=STRIDE)
    val_ds = TemporalWindowDataset(FEATURE_ROOT, VAL_VIDS, window=WINDOW, stride=STRIDE)
    test_ds = TemporalWindowDataset(FEATURE_ROOT, TEST_VIDS, window=WINDOW, stride=STRIDE)

    print(f"Train windows: {len(train_ds)}")
    print(f"Val windows:   {len(val_ds)}")
    print(f"Test windows:  {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    sample_x, _ = train_ds[0]
    input_dim = sample_x.shape[-1]
    print(f"Feature dimension: {input_dim}")

    model = TemporalConvNet(input_dim=input_dim, hidden_dim=256, num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_val_acc = -1.0
    best_ckpt = OUTPUT_DIR / "checkpoints" / "best_stage2_full.pt"

    for epoch in range(1, MAX_EPOCHS + 1):
        print(f"\n========== EPOCH {epoch}/{MAX_EPOCHS} ==========")

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch)
        val_loss, val_acc = validate_one_epoch(model, val_loader, criterion, device, epoch, split_name="val")

        print(
            f"epoch {epoch} summary | "
            f"train_loss {train_loss:.4f} | train_acc {train_acc:.4f} | "
            f"val_loss {val_loss:.4f} | val_acc {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, optimizer, epoch, best_val_acc, best_ckpt, input_dim)

    print("\nEvaluating best model on test set...")
    ckpt = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    test_loss, test_acc = validate_one_epoch(model, test_loader, criterion, device, epoch=0, split_name="test")

    print("\nDone.")
    print(f"Best val_acc: {best_val_acc:.4f}")
    print(f"Test loss:    {test_loss:.4f}")
    print(f"Test acc:     {test_acc:.4f}")
    print(f"Checkpoint:   {best_ckpt}")


if __name__ == "__main__":
    main()