import cv2
from pathlib import Path

VIDEO_DIR = Path("/export/fs06/ahallur1/cholec80/videos")
OUTPUT_DIR = Path("/export/fs06/ahallur1/cholec80/frames/original")

TEST_MODE = False
TEST_VIDEO = "video01.mp4"


def change_size(image):
    binary_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image2 = cv2.threshold(binary_image, 15, 255, cv2.THRESH_BINARY)
    binary_image2 = cv2.medianBlur(binary_image2, 19)

    x = binary_image2.shape[0]
    y = binary_image2.shape[1]

    edges_x = []
    edges_y = []

    for i in range(x):
        for j in range(10, y - 10):
            if binary_image2.item(i, j) != 0:
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

    cropped = image[left:left + width, bottom:bottom + height]
    return cropped


def process_video(video_path, out_root):
    video_name = video_path.stem  # e.g. video01
    out_dir = out_root / video_name
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] Could not open {video_path}")
        return

    frame_num = 0
    print(f"[INFO] Processing {video_name}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        new_width = int(frame.shape[1] / frame.shape[0] * 300)
        frame = cv2.resize(frame, (new_width, 300))

        frame = change_size(frame)
        frame = cv2.resize(frame, (250, 250))

        out_path = out_dir / f"{frame_num:06d}.jpg"
        success = cv2.imwrite(str(out_path), frame)
        if not success:
            print(f"[ERROR] Failed to save {out_path}")
            break

        frame_num += 1

        if frame_num % 1000 == 0:
            print(f"[INFO] {video_name}: saved {frame_num} frames")

    cap.release()
    print(f"[DONE] {video_name}: total frames saved = {frame_num}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if TEST_MODE:
        video_path = VIDEO_DIR / TEST_VIDEO
        if not video_path.exists():
            raise FileNotFoundError(f"Test video not found: {video_path}")
        process_video(video_path, OUTPUT_DIR)
    else:
        videos = sorted(VIDEO_DIR.glob("video*.mp4"))
        if not videos:
            raise FileNotFoundError(f"No videos found in {VIDEO_DIR}")

        for video_path in videos:
            process_video(video_path, OUTPUT_DIR)


if __name__ == "__main__":
    main()