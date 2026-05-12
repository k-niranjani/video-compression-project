#!/usr/bin/env python3

import pickle
from pathlib import Path

DATA_ROOT = Path("/export/fs06/ahallur1/cholec80")
FRAME_ROOT = DATA_ROOT / "frames_1fps_250" / "CRF51"
ANNOT_ROOT = DATA_ROOT / "phase_annotations"
OUT_FILE = Path("/export/fs06/ahallur1/cholec80/Trans-SVNet_CRF51/train_val_paths_labels1.pkl")

PHASE_TO_IDX = {
    "Preparation": 0,
    "CalotTriangleDissection": 1,
    "ClippingCutting": 2,
    "GallbladderDissection": 3,
    "GallbladderPackaging": 4,
    "CleaningCoagulation": 5,
    "GallbladderRetraction": 6,
}

TRAIN_VIDEOS = [f"video{i:02d}" for i in range(1, 41)]
VAL_VIDEOS   = [f"video{i:02d}" for i in range(41, 49)]
TEST_VIDEOS  = [f"video{i:02d}" for i in range(49, 81)]

DOWNSAMPLE_RATE = 25


def load_phase_annotations_1fps(video_name):
    ann_path = ANNOT_ROOT / f"{video_name}-phase.txt"
    rows = []

    with ann_path.open("r") as f:
        header = next(f)
        for line in f:
            parts = line.strip().split()
            if len(parts) != 2:
                raise ValueError(f"Bad annotation line in {ann_path}: {line!r}")

            orig_frame_idx = int(parts[0])
            phase_name = parts[1]

            if orig_frame_idx % DOWNSAMPLE_RATE != 0:
                continue

            if phase_name not in PHASE_TO_IDX:
                raise ValueError(f"Unknown phase '{phase_name}' in {ann_path}")

            sampled_idx = orig_frame_idx // DOWNSAMPLE_RATE
            rows.append((sampled_idx, PHASE_TO_IDX[phase_name]))

    return rows


def build_split(video_names):
    all_paths = []
    all_labels = []
    num_each = []

    for video_name in video_names:
        video_dir = FRAME_ROOT / video_name
        if not video_dir.exists():
            raise FileNotFoundError(f"Missing frame directory: {video_dir}")

        frame_files = sorted(video_dir.glob("*.jpg"))
        available_count = len(frame_files)
        annotations = load_phase_annotations_1fps(video_name)

        count = 0
        skipped = 0

        for sampled_idx, phase_idx in annotations:
            if sampled_idx >= available_count:
                skipped += 1
                continue

            img_path = video_dir / f"{sampled_idx:06d}.jpg"
            if not img_path.exists():
                raise FileNotFoundError(f"Missing frame inside valid range: {img_path}")

            all_paths.append(str(img_path))
            all_labels.append([phase_idx])
            count += 1

        num_each.append(count)
        print(f"{video_name}: kept {count} frames, skipped {skipped} tail labels, available jpgs {available_count}")

    return all_paths, all_labels, num_each


def main():
    train_paths, train_labels, train_num_each = build_split(TRAIN_VIDEOS)
    val_paths, val_labels, val_num_each = build_split(VAL_VIDEOS)
    test_paths, test_labels, test_num_each = build_split(TEST_VIDEOS)

    payload = [
        train_paths,
        val_paths,
        train_labels,
        val_labels,
        train_num_each,
        val_num_each,
        test_paths,
        test_labels,
        test_num_each,
    ]

    with OUT_FILE.open("wb") as f:
        pickle.dump(payload, f)

    print()
    print(f"Wrote {OUT_FILE}")
    print(f"train frames: {len(train_paths)}")
    print(f"val frames:   {len(val_paths)}")
    print(f"test frames:  {len(test_paths)}")
    print(f"train videos: {len(train_num_each)}")
    print(f"val videos:   {len(val_num_each)}")
    print(f"test videos:  {len(test_num_each)}")


if __name__ == "__main__":
    main()