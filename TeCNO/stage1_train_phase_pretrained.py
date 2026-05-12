from pathlib import Path
from types import SimpleNamespace
import yaml
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from utils.utils import get_class_by_path


# =========================
# CHANGE ONLY THESE
# =========================
CONFIG_PATH = r"modules\cnn\config\config_feature_extract_pretrained.yml"
DATA_ROOT = r"E:\VC\cholec80_processed_1fps\crf51"   # change to crf18, crf23, etc.
OUTPUT_ROOT = r"E:\VC\experiments\paper_pretrained"
CUSTOM_SUFFIX = "pretrained"   # change only if you want a different suffix
# =========================


def get_run_tag(data_root: str, custom_suffix: str = "pretrained") -> str:
    folder_name = Path(data_root).name.lower()

    if folder_name == "videos":
        base_name = "original"
    else:
        base_name = folder_name

    return f"{base_name}_{custom_suffix}"


def load_config(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    hparams = SimpleNamespace(**cfg)

    # ===== DEFAULTS =====
    if not hasattr(hparams, "batch_size"):
        hparams.batch_size = 8
    if not hasattr(hparams, "num_workers"):
        hparams.num_workers = 2
    if not hasattr(hparams, "max_epochs"):
        hparams.max_epochs = 25
    if not hasattr(hparams, "min_epochs"):
        hparams.min_epochs = 1
    if not hasattr(hparams, "learning_rate"):
        hparams.learning_rate = 5e-4
    if not hasattr(hparams, "fps_sampling"):
        hparams.fps_sampling = 1.0
    if not hasattr(hparams, "fps_sampling_test"):
        hparams.fps_sampling_test = 1.0
    if not hasattr(hparams, "dataset_mode"):
        hparams.dataset_mode = "img_multilabel"
    if not hasattr(hparams, "test_extract"):
        hparams.test_extract = True
    if not hasattr(hparams, "num_tasks"):
        hparams.num_tasks = 1
    if not hasattr(hparams, "input_height"):
        hparams.input_height = 224
    if not hasattr(hparams, "input_width"):
        hparams.input_width = 224
    if not hasattr(hparams, "out_features"):
        hparams.out_features = 7
    if not hasattr(hparams, "num_tool_classes"):
        hparams.num_tool_classes = 7
    if not hasattr(hparams, "dropout"):
        hparams.dropout = 0.5

    # ===== FORCE CORRECT TYPES =====
    hparams.learning_rate = float(hparams.learning_rate)
    hparams.batch_size = int(hparams.batch_size)
    hparams.num_workers = int(hparams.num_workers)
    hparams.max_epochs = int(hparams.max_epochs)
    hparams.min_epochs = int(hparams.min_epochs)
    hparams.input_height = int(hparams.input_height)
    hparams.input_width = int(hparams.input_width)
    hparams.out_features = int(hparams.out_features)
    hparams.num_tool_classes = int(hparams.num_tool_classes)
    hparams.dropout = float(hparams.dropout)
    hparams.fps_sampling = float(hparams.fps_sampling)
    hparams.fps_sampling_test = float(hparams.fps_sampling_test)

    return hparams


def make_dataloader(dataset_obj, split: str, batch_size: int, num_workers: int):
    ds = dataset_obj.data[split]
    shuffle = split == "train"
    workers = 0 if split == "test" else num_workers

    return DataLoader(
        dataset=ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
    )


def phase_accuracy(logits, targets):
    preds = torch.argmax(logits, dim=1)
    return (preds == targets).float().mean().item()


def train_one_epoch(model, loader, optimizer, ce_loss, device, epoch_idx):
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    total_batches = 0

    for batch_idx, batch in enumerate(loader):
        x, y_phase, y_tool = batch
        x = x.to(device, non_blocking=True)
        y_phase = y_phase.to(device, non_blocking=True)

        optimizer.zero_grad()
        _, p_phase, _ = model(x)
        loss = ce_loss(p_phase, y_phase)
        loss.backward()
        optimizer.step()

        acc = phase_accuracy(p_phase.detach(), y_phase)

        running_loss += loss.item()
        running_acc += acc
        total_batches += 1

        if batch_idx % 100 == 0:
            print(
                f"train | epoch {epoch_idx} | batch {batch_idx}/{len(loader)} "
                f"| loss {loss.item():.4f} | acc {acc:.4f}"
            )

    return running_loss / max(total_batches, 1), running_acc / max(total_batches, 1)


@torch.no_grad()
def validate_one_epoch(model, loader, ce_loss, device, epoch_idx):
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    total_batches = 0

    for batch_idx, batch in enumerate(loader):
        x, y_phase, y_tool = batch
        x = x.to(device, non_blocking=True)
        y_phase = y_phase.to(device, non_blocking=True)

        _, p_phase, _ = model(x)
        loss = ce_loss(p_phase, y_phase)
        acc = phase_accuracy(p_phase, y_phase)

        running_loss += loss.item()
        running_acc += acc
        total_batches += 1

        if batch_idx % 100 == 0:
            print(
                f"val   | epoch {epoch_idx} | batch {batch_idx}/{len(loader)} "
                f"| loss {loss.item():.4f} | acc {acc:.4f}"
            )

    return running_loss / max(total_batches, 1), running_acc / max(total_batches, 1)


def save_checkpoint(model, optimizer, epoch, best_val_acc, save_path: Path):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_acc": best_val_acc,
        },
        save_path,
    )
    print(f"Saved checkpoint to: {save_path}")


def main():
    run_tag = get_run_tag(DATA_ROOT, CUSTOM_SUFFIX)

    hparams = load_config(CONFIG_PATH)
    hparams.data_root = DATA_ROOT
    hparams.output_path = Path(OUTPUT_ROOT) / run_tag / "stage1"
    hparams.output_path.mkdir(parents=True, exist_ok=True)

    print("\n========== LOADED CONFIG ==========")
    print("CONFIG FILE :", CONFIG_PATH)
    print("RUN TAG     :", run_tag)
    print("DATA ROOT   :", hparams.data_root)
    print("OUTPUT PATH :", hparams.output_path)
    print("BATCH SIZE  :", hparams.batch_size)
    print("MAX EPOCHS  :", hparams.max_epochs)
    print("===================================\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model_path = f"models.{hparams.model}"
    dataset_path = f"datasets.{hparams.dataset}"

    DatasetClass = get_class_by_path(dataset_path)
    ModelClass = get_class_by_path(model_path)

    dataset = DatasetClass(hparams=hparams)
    model = ModelClass(hparams).to(device)

    print("\n========== DATASET SUMMARY ==========")
    print("Train samples:", len(dataset.data["train"]))
    print("Val samples  :", len(dataset.data["val"]))
    print("Test samples :", len(dataset.data["test"]))
    print("=====================================\n")

    train_loader = make_dataloader(dataset, "train", hparams.batch_size, hparams.num_workers)
    val_loader = make_dataloader(dataset, "val", hparams.batch_size, hparams.num_workers)

    class_weights = torch.from_numpy(dataset.class_weights).float().to(device)
    ce_loss = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=hparams.learning_rate
    )

    best_val_acc = -1.0
    best_ckpt = hparams.output_path / "checkpoints" / "best_phase_only_pretrained.pt"

    for epoch in range(1, hparams.max_epochs + 1):
        print(f"\n========== EPOCH {epoch}/{hparams.max_epochs} ==========")

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, ce_loss, device, epoch)
        val_loss, val_acc = validate_one_epoch(model, val_loader, ce_loss, device, epoch)

        print(
            f"epoch {epoch} summary | "
            f"train_loss {train_loss:.4f} | train_acc {train_acc:.4f} | "
            f"val_loss {val_loss:.4f} | val_acc {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, optimizer, epoch, best_val_acc, best_ckpt)

    print("\nTraining finished.")
    print(f"Best val_acc: {best_val_acc:.4f}")
    print(f"Best checkpoint: {best_ckpt}")


if __name__ == "__main__":
    main()