#!/usr/bin/env python3

import argparse
import pickle
from pathlib import Path

PHASE_TO_IDX = {
    "Preparation": 0,
    "CalotTriangleDissection": 1,
    "ClippingCutting": 2,
    "GallbladderDissection": 3,
    "GallbladderPackaging": 4,
    "CleaningCoagulation": 5,
    "GallbladderRetraction": 6,
}

DOWNSAMPLE_RATE = 25


def load_phase_annotations_1fps(annot_root: Path, video_name: str):
    ann_path = annot_root / f"{video_name}-phase.txt"
    rows = []

    with ann_path.open("r") as f:
        _ = next(f)  # header
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


def build_test_split(frame_root: Path, annot_root: Path, video_names):
    test_paths = []
    test_labels = []
    test_num_each = []

    for video_name in video_names:
        video_dir = frame_root / video_name
        if not video_dir.exists():
            raise FileNotFoundError(f"Missing frame directory: {video_dir}")

        frame_files = sorted(video_dir.glob("*.jpg"))
        available_count = len(frame_files)
        annotations = load_phase_annotations_1fps(annot_root, video_name)

        count = 0
        skipped = 0

        for sampled_idx, phase_idx in annotations:
            if sampled_idx >= available_count:
                skipped += 1
                continue

            img_path = video_dir / f"{sampled_idx:06d}.jpg"
            if not img_path.exists():
                raise FileNotFoundError(f"Missing frame inside valid range: {img_path}")

            test_paths.append(str(img_path))
            test_labels.append([phase_idx])
            count += 1

        test_num_each.append(count)
        print(
            f"{video_name}: kept {count} frames, "
            f"skipped {skipped} tail labels, available jpgs {available_count}"
        )

    return test_paths, test_labels, test_num_each


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, default="/export/fs06/ahallur1/cholec80", help="Cholec80 root, e.g. /export/fs06/ahallur1/cholec80")
    parser.add_argument("--frame-root", required=True, help="Condition frame root, e.g. /.../frames_1fps_250/CRF18")
    parser.add_argument("--test-start", type=int, default=49, help="First test video number")
    parser.add_argument("--test-end", type=int, default=80, help="Last test video number (inclusive)")
    parser.add_argument("--out", default="test_paths_labels.pkl", help="Output pickle path")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    frame_root = Path(args.frame_root)
    annot_root = data_root / "phase_annotations"

    video_names = [f"video{i:02d}" for i in range(args.test_start, args.test_end + 1)]

    test_paths, test_labels, test_num_each = build_test_split(frame_root, annot_root, video_names)

    payload = [test_paths, test_labels, test_num_each]

    out_path = Path(args.out)
    with out_path.open("wb") as f:
        pickle.dump(payload, f)

    print()
    print(f"Wrote {out_path}")
    print(f"test videos: {len(test_num_each)}")
    print(f"test frames: {len(test_paths)}")


if __name__ == "__main__":
    main()