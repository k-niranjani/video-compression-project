#!/usr/bin/env python3

import sys
import cv2
from pathlib import Path

if len(sys.argv) != 2:
    print("Usage: python process_condition_1fps.py <CONDITION>")
    sys.exit(1)

CONDITION = sys.argv[1]

SRC_ROOT = Path(f"/export/fs06/ahallur1/cholec80/frames_1fps/{CONDITION}")
DST_ROOT = Path(f"/export/fs06/ahallur1/cholec80/frames_1fps_250/{CONDITION}")

def change_size(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
    binary = cv2.medianBlur(binary, 19)

    x = binary.shape[0]
    y = binary.shape[1]

    edges_x = []
    edges_y = []

    for i in range(x):
        for j in range(10, y - 10):
            if binary.item(i, j) != 0:
                edges_x.append(i)
                edges_y.append(j)

    if not edges_x or not edges_y:
        return image

    left = min(edges_x)
    right = max(edges_x)
    bottom = min(edges_y)
    top = max(edges_y)

    width = right - left
    height = top - bottom

    if width <= 0 or height <= 0:
        return image

    return image[left:left + width, bottom:bottom + height]

def process_video(video_dir: Path):
    dst_dir = DST_ROOT / video_dir.name
    dst_dir.mkdir(parents=True, exist_ok=True)

    frame_paths = sorted(video_dir.glob("*.jpg"))
    print(f"[INFO] {CONDITION} {video_dir.name}: found {len(frame_paths)} frames")

    for new_idx, frame_path in enumerate(frame_paths):
        img = cv2.imread(str(frame_path))
        if img is None:
            raise RuntimeError(f"Could not read {frame_path}")

        img = change_size(img)
        img = cv2.resize(img, (250, 250))

        out_path = dst_dir / f"{new_idx:06d}.jpg"
        ok = cv2.imwrite(str(out_path), img)
        if not ok:
            raise RuntimeError(f"Could not write {out_path}")

    print(f"[DONE] {CONDITION} {video_dir.name}: wrote {len(frame_paths)} frames")

def main():
    DST_ROOT.mkdir(parents=True, exist_ok=True)
    for video_dir in sorted(SRC_ROOT.glob("video*")):
        process_video(video_dir)

if __name__ == "__main__":
    main()