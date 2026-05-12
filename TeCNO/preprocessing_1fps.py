from pathlib import Path
import cv2
import pandas as pd

# =========================
# PATHS
# =========================

VIDEO_DIR = Path(r"E:\VC\cholec80\videos")  # change to videos_CRFxx if needed
PHASE_DIR = Path(r"E:\VC\cholec80\phase_annotations")

OUT_ROOT = Path(r"E:\VC\cholec80_processed_1fps\videos\frames")
DF_ROOT = Path(r"E:\VC\cholec80_processed_1fps\videos\dataframes")

DF_ROOT.mkdir(parents=True, exist_ok=True)
OUT_ROOT.mkdir(parents=True, exist_ok=True)

# Set True if you want to extract frames from videos
# Set False if frames already exist and you only want to rebuild dataframe
EXTRACT_FRAMES = True

PHASE_TO_ID = {
    "Preparation": 0,
    "CalotTriangleDissection": 1,
    "ClippingCutting": 2,
    "GallbladderDissection": 3,
    "GallbladderPackaging": 4,
    "CleaningCoagulation": 5,
    "GallbladderRetraction": 6,
}

# =========================
# FRAME EXTRACTION
# =========================

def extract_frames_1fps(video_path: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        raise RuntimeError(f"Invalid FPS for video: {video_path}")

    frame_interval = max(1, int(round(fps)))

    print(f"{video_path.name}: fps={fps:.2f}, extracting every {frame_interval} frames")

    frame_idx = 0
    saved_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            out_path = out_dir / f"{saved_idx:06d}.jpg"
            cv2.imwrite(str(out_path), frame)
            saved_idx += 1

        frame_idx += 1

    cap.release()

    print(f"Saved {saved_idx} frames to {out_dir}")
    return frame_interval


# =========================
# DATAFRAME BUILDING
# =========================

def build_dataframe_for_video(stem, vid, frame_dir, phase_file, frame_interval):
    rows = []

    ann = pd.read_csv(phase_file, sep=r"\s+", engine="python")
    ann.columns = ["Frame", "Phase"]

    for _, r in ann.iterrows():
        original_frame_idx = int(r["Frame"])
        phase_name = str(r["Phase"]).strip()

        if phase_name not in PHASE_TO_ID:
            print(f"Unknown phase '{phase_name}' in {phase_file.name}, skipping")
            continue

        if original_frame_idx % frame_interval != 0:
            continue

        frame_idx_1fps = original_frame_idx // frame_interval
        img_name = f"{frame_idx_1fps:06d}.jpg"
        img_path = frame_dir / img_name

        if not img_path.exists():
            continue

        rows.append({
            "image_path": f"{stem}/{img_name}",
            "class": PHASE_TO_ID[phase_name],
            "video_idx": int(vid),
            "tool_Grasper": 0,
            "tool_Bipolar": 0,
            "tool_Hook": 0,
            "tool_Scissors": 0,
            "tool_Clipper": 0,
            "tool_Irrigator": 0,
            "tool_SpecimenBag": 0,
        })

    return rows


# =========================
# MAIN
# =========================

all_rows = []

if EXTRACT_FRAMES:
    video_files = sorted([
        p for p in VIDEO_DIR.iterdir()
        if p.is_file()
        and p.suffix.lower() == ".mp4"
        and not p.name.startswith("._")
        and p.name != ".DS_Store"
    ])

    print(f"Found {len(video_files)} video files in {VIDEO_DIR}")

    for video_file in video_files:
        stem = video_file.stem
        vid = stem.replace("video", "")
        phase_file = PHASE_DIR / f"{stem}-phase.txt"
        frame_dir = OUT_ROOT / stem

        if not phase_file.exists():
            print(f"Missing phase file: {phase_file}")
            continue

        frame_interval = extract_frames_1fps(video_file, frame_dir)

        rows = build_dataframe_for_video(
            stem=stem,
            vid=vid,
            frame_dir=frame_dir,
            phase_file=phase_file,
            frame_interval=frame_interval
        )

        all_rows.extend(rows)

else:
    FRAME_INTERVAL = 25

    video_dirs = sorted([
        p for p in OUT_ROOT.iterdir()
        if p.is_dir() and p.name.startswith("video")
    ])

    print(f"Found {len(video_dirs)} existing frame folders in {OUT_ROOT}")

    for frame_dir in video_dirs:
        stem = frame_dir.name
        vid = stem.replace("video", "")
        phase_file = PHASE_DIR / f"{stem}-phase.txt"

        if not phase_file.exists():
            print(f"Missing phase file: {phase_file}")
            continue

        rows = build_dataframe_for_video(
            stem=stem,
            vid=vid,
            frame_dir=frame_dir,
            phase_file=phase_file,
            frame_interval=FRAME_INTERVAL
        )

        all_rows.extend(rows)


df = pd.DataFrame(all_rows)

if not df.empty:
    df = df.drop_duplicates(subset=["image_path"]).reset_index(drop=True)

out_pkl = DF_ROOT / "cholec80_1fps.pkl"
df.to_pickle(out_pkl)

print("\nDone.")
print(f"Saved dataframe: {out_pkl}")
print(f"Total rows: {len(df)}")

if not df.empty:
    print(f"Unique image paths: {df['image_path'].nunique()}")
    print(f"Unique videos: {df['video_idx'].nunique()}")
    print("\nClass counts:")
    print(df["class"].value_counts().sort_index())
    print("\nSample rows:")
    print(df.head())
else:
    print("Dataframe is empty. Check paths, filenames, or phase names.")