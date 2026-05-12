#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import precision_score, recall_score, jaccard_score

PHASES = [
    "Preparation",
    "CalotTriangleDissection",
    "ClippingCutting",
    "GallbladderDissection",
    "GallbladderPackaging",
    "CleaningCoagulation",
    "GallbladderRetraction",
]

PHASE_TO_IDX = {name: i for i, name in enumerate(PHASES)}
DOWNSAMPLE_RATE = 25


def read_gt_1fps(gt_path: Path):
    labels = []
    with gt_path.open("r") as f:
        _ = next(f)
        for line in f:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            frame_idx = int(parts[0])
            phase_name = parts[1]
            if frame_idx % DOWNSAMPLE_RATE != 0:
                continue
            labels.append(PHASE_TO_IDX[phase_name])
    return np.asarray(labels, dtype=np.int64)


def read_pred_txt(pred_path: Path):
    labels = []
    with pred_path.open("r") as f:
        _ = next(f)
        for line in f:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            labels.append(int(parts[1]))
    return np.asarray(labels, dtype=np.int64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--test-start", type=int, default=49)
    parser.add_argument("--test-end", type=int, default=80)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    pred_dir = Path(args.pred_dir)
    gt_root = data_root / "phase_annotations"

    acc_per_video = []
    prec_per_video = []
    rec_per_video = []
    jac_per_video = []

    for k in range(args.test_start, args.test_end + 1):
        video_name = f"video{k:02d}"
        gt_path = gt_root / f"{video_name}-phase.txt"
        pred_path = pred_dir / f"{video_name}-phase.txt"

        if not pred_path.exists():
            raise FileNotFoundError(f"Missing prediction file: {pred_path}")

        gt = read_gt_1fps(gt_path)
        pred = read_pred_txt(pred_path)

        usable = min(len(gt), len(pred))
        gt = gt[:usable]
        pred = pred[:usable]

        acc = float((gt == pred).mean())
        prec = precision_score(gt, pred, labels=list(range(7)), average=None, zero_division=0)
        rec = recall_score(gt, pred, labels=list(range(7)), average=None, zero_division=0)
        jac = jaccard_score(gt, pred, labels=list(range(7)), average=None, zero_division=0)

        acc_per_video.append(acc)
        prec_per_video.append(prec)
        rec_per_video.append(rec)
        jac_per_video.append(jac)

        print(
            f"{video_name}: frames={usable}, "
            f"acc={acc:.4f}, "
            f"prec_macro={prec.mean():.4f}, "
            f"rec_macro={rec.mean():.4f}, "
            f"jac_macro={jac.mean():.4f}"
        )

    prec_per_video = np.asarray(prec_per_video)
    rec_per_video = np.asarray(rec_per_video)
    jac_per_video = np.asarray(jac_per_video)
    acc_per_video = np.asarray(acc_per_video)

    mean_prec_per_phase = prec_per_video.mean(axis=0)
    mean_rec_per_phase = rec_per_video.mean(axis=0)
    mean_jac_per_phase = jac_per_video.mean(axis=0)

    print()
    print("================================================")
    print(f"{'Phase':25s} | {'Jacc':>6s} | {'Prec':>6s} | {'Rec':>6s}")
    print("================================================")
    for i, phase in enumerate(PHASES):
        print(
            f"{phase:25s} | "
            f"{mean_jac_per_phase[i] * 100:6.2f} | "
            f"{mean_prec_per_phase[i] * 100:6.2f} | "
            f"{mean_rec_per_phase[i] * 100:6.2f}"
        )
    print("================================================")

    mean_jacc = mean_jac_per_phase.mean() * 100
    mean_prec = mean_prec_per_phase.mean() * 100
    mean_rec = mean_rec_per_phase.mean() * 100
    mean_acc = acc_per_video.mean() * 100

    std_jacc = jac_per_video.mean(axis=1).std() * 100
    std_prec = prec_per_video.mean(axis=1).std() * 100
    std_rec = rec_per_video.mean(axis=1).std() * 100
    std_acc = acc_per_video.std() * 100

    print(f"Mean jaccard : {mean_jacc:5.2f} +- {std_jacc:5.2f}")
    print(f"Mean accuracy: {mean_acc:5.2f} +- {std_acc:5.2f}")
    print(f"Mean precision: {mean_prec:5.2f} +- {std_prec:5.2f}")
    print(f"Mean recall   : {mean_rec:5.2f} +- {std_rec:5.2f}")


if __name__ == "__main__":
    main()