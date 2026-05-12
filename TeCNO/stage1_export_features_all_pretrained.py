from pathlib import Path
from types import SimpleNamespace
import yaml
import torch
from torch.utils.data import DataLoader
import numpy as np
import pickle

from utils.utils import get_class_by_path


# =========================
# CHANGE ONLY THESE
# =========================
CONFIG_PATH = r"modules\cnn\config\config_feature_extract_pretrained.yml"
import sys

DATA_ROOT = sys.argv[1] if len(sys.argv) > 1 else r"E:\VC\cholec80_processed_1fps\videos"  # change to crf18, crf23, etc.
OUTPUT_ROOT = r"E:\VC\experiments\paper_pretrained"
# =========================


def get_run_tag(data_root: str) -> str:
    folder_name = Path(data_root).name.lower()

    if folder_name == "videos":
        return "original_pretrained"
    return f"{folder_name}_pretrained"


def load_config(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    hparams = SimpleNamespace(**cfg)

    if not hasattr(hparams, "batch_size"):
        hparams.batch_size = 8
    if not hasattr(hparams, "num_workers"):
        hparams.num_workers = 2
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

    hparams.batch_size = int(hparams.batch_size)
    hparams.num_workers = int(hparams.num_workers)
    hparams.input_height = int(hparams.input_height)
    hparams.input_width = int(hparams.input_width)
    hparams.out_features = int(hparams.out_features)
    hparams.num_tool_classes = int(hparams.num_tool_classes)
    hparams.dropout = float(hparams.dropout)
    hparams.fps_sampling = float(hparams.fps_sampling)
    hparams.fps_sampling_test = float(hparams.fps_sampling_test)
    hparams.data_root = str(DATA_ROOT)

    return hparams


def make_loader_from_df(df, transform, label_col, img_root, batch_size):
    DatasetClass = get_class_by_path("datasets.cholec80_feature_extract.Dataset_from_Dataframe")
    ds = DatasetClass(
        df=df.reset_index(),
        transform=transform,
        label_col=label_col,
        img_root=img_root,
        image_path_col="image_path",
        add_label_cols=[
            "video_idx",
            "image_path",
            "index",
            "tool_Grasper",
            "tool_Bipolar",
            "tool_Hook",
            "tool_Scissors",
            "tool_Clipper",
            "tool_Irrigator",
            "tool_SpecimenBag",
        ],
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )


def save_video_pickle(save_root: Path, fps: float, vid_idx: int, stems, p_phases, labels):
    out_dir = save_root / f"{int(fps)}fps"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_file = out_dir / f"video_{vid_idx:02d}_{int(fps)}fps.pkl"
    with open(out_file, "wb") as f:
        pickle.dump(
            [
                np.asarray(stems),
                np.asarray(p_phases),
                np.asarray(labels),
            ],
            f,
        )

    print(f"saved: {out_file}")


def export_split(model, loader, device, export_root, fps):
    current_video_idx = None
    current_stems = []
    current_p_phases = []
    current_phase_labels = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            x, y_phase, extra = batch
            (
                vid_idx,
                img_name,
                img_index,
                tool_Grasper,
                tool_Bipolar,
                tool_Hook,
                tool_Scissors,
                tool_Clipper,
                tool_Irrigator,
                tool_SpecimenBag,
            ) = extra

            x = x.to(device, non_blocking=True)
            stem, p_phase, _ = model(x)

            vid_idx_np = vid_idx.cpu().numpy()
            y_phase_np = y_phase.cpu().numpy()
            stem_np = stem.detach().cpu().numpy()
            p_phase_np = p_phase.detach().cpu().numpy()

            for i in range(len(vid_idx_np)):
                this_vid = int(vid_idx_np[i])

                if current_video_idx is None:
                    current_video_idx = this_vid

                if this_vid != current_video_idx:
                    save_video_pickle(
                        export_root,
                        fps,
                        current_video_idx,
                        current_stems,
                        current_p_phases,
                        current_phase_labels,
                    )
                    current_video_idx = this_vid
                    current_stems = []
                    current_p_phases = []
                    current_phase_labels = []

                current_stems.append(stem_np[i])
                current_p_phases.append(p_phase_np[i])
                current_phase_labels.append(y_phase_np[i])

            if batch_idx % 500 == 0:
                print(f"Processed batch {batch_idx}/{len(loader)}")

        if current_video_idx is not None:
            save_video_pickle(
                export_root,
                fps,
                current_video_idx,
                current_stems,
                current_p_phases,
                current_phase_labels,
            )


def main():
    run_tag = get_run_tag(DATA_ROOT)

    hparams = load_config(CONFIG_PATH)
    hparams.num_tasks = 1
    hparams.data_root = DATA_ROOT

    checkpoint_path = (
        Path(OUTPUT_ROOT)
        / run_tag
        / "stage1"
        / "checkpoints"
        / "best_phase_only_pretrained.pt"
    )
    export_root = Path(OUTPUT_ROOT) / run_tag / "stage2_features"

    print("\n========== EXPORT CONFIG ==========")
    print("CONFIG FILE    :", CONFIG_PATH)
    print("RUN TAG        :", run_tag)
    print("DATA ROOT      :", hparams.data_root)
    print("CHECKPOINT     :", checkpoint_path)
    print("EXPORT ROOT    :", export_root)
    print("FPS TEST       :", hparams.fps_sampling_test)
    print("BATCH SIZE     :", hparams.batch_size)
    print("===================================\n")

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    dataset_path = f"datasets.{hparams.dataset}"
    model_path = f"models.{hparams.model}"

    DatasetClass = get_class_by_path(dataset_path)
    ModelClass = get_class_by_path(model_path)

    dataset = DatasetClass(hparams=hparams)
    model = ModelClass(hparams).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()

    img_root = Path(hparams.data_root) / "frames"
    transform = dataset.transformations["test"]
    label_col = dataset.label_col

    loader = make_loader_from_df(
        dataset.df["all"],
        transform,
        label_col,
        img_root,
        hparams.batch_size,
    )

    print("Exporting features for all videos...")
    export_split(model, loader, device, export_root, hparams.fps_sampling_test)
    print("Full feature export finished.")


if __name__ == "__main__":
    main()