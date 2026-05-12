"""
Compression analysis for Cholec80:
  0. Storage vs baseline (computed from raw .mp4 video files)
  1. Edge preservation (Canny IoU + retention)
  2. Shannon entropy + Laplacian sharpness
  3. Visual 5-up side-by-side frame comparisons

CRF levels: uncompressed (Baseline) + CRF 23 / 28 / 35 / 51.
"""

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# CONFIG
FRAMES_BASE = '/Volumes/LaCie/data/cholec80'
VIDEOS_BASE = '/Volumes/JustAGuy/cholec80'
OUT_DIR     = './compression_analysis'
PLOT_DIR    = os.path.join(OUT_DIR, 'plots')
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

CRF_MAP = {
    'uncompressed': ('videos',        'cutMargin'),
    'CRF23':        ('videos_CRF23',  'cutMarginCRF23'),
    'CRF28':        ('videos_CRF28',  'cutMarginCRF28'),
    'CRF35':        ('videos_CRF35',  'cutMarginCRF35'),
    'CRF51':        ('videos_CRF51',  'cutMarginCRF51'),
}
CRF_FOLDERS = {k: v[1] for k, v in CRF_MAP.items()}
CRF_ORDER   = ['uncompressed', 'CRF23', 'CRF28', 'CRF35', 'CRF51']
CRF_COLORS  = {'uncompressed': '#2E86AB', 'CRF23': '#D5B60A',
               'CRF28': '#F18F01', 'CRF35': '#C73E1D', 'CRF51': '#6A0F49'}

SAMPLE_EVERY = 20  

plt.rcParams.update({
    'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 13,
    'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 10,
    'figure.dpi': 120, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
})


def list_jpgs(folder):
    return sorted([f for f in os.listdir(folder)
                   if f.endswith('.jpg') and not f.startswith('.')],
                  key=lambda x: int(x.split('.')[0]))


def list_video_dirs(root):
    return sorted([d for d in os.listdir(root)
                   if d.isdigit() and os.path.isdir(os.path.join(root, d))],
                  key=int)


def pretty_label(crf):
    """uncompressed → 'Baseline', CRF23 → 'CRF 23'."""
    return 'Baseline' if crf == 'uncompressed' else crf.replace('CRF', 'CRF ')


# 0. Storage vs baseline 
def analyze_storage():
    print("\n" + "=" * 60 + "\n0. Storage vs baseline\n" + "=" * 60)
    rows = []
    for label, (video_folder, _) in CRF_MAP.items():
        path = os.path.join(VIDEOS_BASE, video_folder)
        if not os.path.isdir(path):
            print(f"  ⚠️  {video_folder} not found, skipping")
            continue
        for vid_num in range(1, 81):
            vpath = os.path.join(path, f'video{vid_num:02d}.mp4')
            if not os.path.exists(vpath):
                continue
            size_mb = os.path.getsize(vpath) / (1024 * 1024)
            rows.append({'video': vid_num, 'crf': label,
                         'size_mb': round(size_mb, 2)})

    df = pd.DataFrame(rows)
    df.to_csv(f'{OUT_DIR}/storage_per_video.csv', index=False)

    totals = df.groupby('crf').agg(
        total_size_gb=('size_mb', lambda x: x.sum() / 1024)
    ).round(2)

    if 'uncompressed' in totals.index:
        ref = totals.loc['uncompressed', 'total_size_gb']
        totals['compression_ratio'] = (ref / totals['total_size_gb']).round(2)
        totals['storage_saved_pct'] = ((1 - totals['total_size_gb'] / ref) * 100).round(1)

    totals = totals.reindex([c for c in CRF_ORDER if c in totals.index])
    totals.to_csv(f'{OUT_DIR}/storage_summary.csv')
    print(totals)
    fig, ax = plt.subplots(figsize=(9, 6))
    crfs    = totals.index.tolist()
    xlabels = [pretty_label(c) for c in crfs]
    colors  = [CRF_COLORS[c] for c in crfs]

    bars = ax.bar(xlabels, totals['total_size_gb'], color=colors,
                  edgecolor='black', linewidth=0.6, width=0.65)

    ymax = totals['total_size_gb'].max()
    ax.set_ylim(0, ymax * 1.30)
    ax.set_ylabel('Total dataset size (GB)')
    ax.set_title('Cholec80 Storage vs. Baseline', fontweight='bold', pad=14)
    ax.grid(axis='y', alpha=0.25)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)

    for b, crf in zip(bars, crfs):
        gb = totals.loc[crf, 'total_size_gb']
        x  = b.get_x() + b.get_width() / 2
        y  = b.get_height()
        ax.text(x, y + ymax * 0.020, f'{gb:.1f} GB',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

        if crf == 'uncompressed':
            ax.text(x, y + ymax * 0.075, 'reference',
                    ha='center', va='bottom', fontsize=9,
                    style='italic', color='gray')
        else:
            saved = totals.loc[crf, 'storage_saved_pct']
            ratio = totals.loc[crf, 'compression_ratio']
            ax.text(x, y + ymax * 0.075, f'−{saved:.1f}%',
                    ha='center', va='bottom', fontsize=9.5,
                    color=CRF_COLORS[crf], fontweight='bold')
            ax.text(x, y + ymax * 0.125, f'{ratio:.1f}× smaller',
                    ha='center', va='bottom', fontsize=8.5,
                    style='italic', color='gray')

    plt.tight_layout()
    plt.savefig(f'{PLOT_DIR}/storage_vs_baseline.png'); plt.close()
    print(f"  ✅ {PLOT_DIR}/storage_vs_baseline.png")


# 1. Edge preservation (Canny IoU + retention)
def analyze_edge_preservation():
    print("\n" + "=" * 60 + "\n1. Edge preservation\n" + "=" * 60)
    ref_root = os.path.join(FRAMES_BASE, CRF_FOLDERS['uncompressed'])
    vids = list_video_dirs(ref_root)[:20]

    rows = []
    for vid in tqdm(vids, desc='edges'):
        vfolder = os.path.join(ref_root, vid)
        frames = list_jpgs(vfolder)[::SAMPLE_EVERY]
        for fname in frames:
            ref = cv2.imread(os.path.join(ref_root, vid, fname), cv2.IMREAD_GRAYSCALE)
            if ref is None:
                continue
            ref_edges = cv2.Canny(ref, 100, 200)
            ref_edge_count = ref_edges.sum() / 255

            for label, folder in CRF_FOLDERS.items():
                p = os.path.join(FRAMES_BASE, folder, vid, fname)
                if not os.path.exists(p):
                    continue
                img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                cmp_edges = cv2.Canny(img, 100, 200)
                cmp_edge_count = cmp_edges.sum() / 255
                intersection = np.logical_and(ref_edges > 0, cmp_edges > 0).sum()
                union = np.logical_or(ref_edges > 0, cmp_edges > 0).sum()
                edge_iou = intersection / union if union else 0
                edge_retention = (cmp_edge_count / ref_edge_count
                                  if ref_edge_count else 0)
                rows.append({'video': int(vid),
                             'frame': int(fname.split('.')[0]),
                             'crf': label,
                             'edge_iou': edge_iou,
                             'edge_retention': edge_retention})

    df = pd.DataFrame(rows)
    df.to_csv(f'{OUT_DIR}/edge_preservation.csv', index=False)
    summary = df.groupby('crf')[['edge_iou', 'edge_retention']].mean().round(4)
    summary = summary.reindex([c for c in CRF_ORDER if c in summary.index])
    summary.to_csv(f'{OUT_DIR}/edge_summary.csv')
    print(summary)

    fig, ax = plt.subplots(figsize=(9, 5))
    xlabels = [pretty_label(c) for c in summary.index]
    x = np.arange(len(summary))
    w = 0.35
    ax.bar(x - w / 2, summary['edge_retention'], w, label='Edge retention',
           color='#2E86AB', edgecolor='black')
    ax.bar(x + w / 2, summary['edge_iou'], w, label='Edge IoU',
           color='#C73E1D', edgecolor='black')
    ax.set_xticks(x); ax.set_xticklabels(xlabels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Fraction')
    ax.set_title('Edge Preservation under Compression')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{PLOT_DIR}/edge_preservation.png'); plt.close()
    print(f"  ✅ {PLOT_DIR}/edge_preservation.png")

# 2. Shannon entropy + Laplacian sharpness
def analyze_entropy_and_blocks():
    print("\n" + "=" * 60 + "\n2. Shannon entropy + sharpness\n" + "=" * 60)
    ref_root = os.path.join(FRAMES_BASE, CRF_FOLDERS['uncompressed'])
    vids = list_video_dirs(ref_root)[:20]

    rows = []
    for vid in tqdm(vids, desc='entropy'):
        vfolder = os.path.join(ref_root, vid)
        frames = list_jpgs(vfolder)[::SAMPLE_EVERY]
        for fname in frames:
            for label, folder in CRF_FOLDERS.items():
                p = os.path.join(FRAMES_BASE, folder, vid, fname)
                if not os.path.exists(p):
                    continue
                img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue

                hist, _ = np.histogram(img, bins=256, range=(0, 256))
                hist = hist / hist.sum()
                hist = hist[hist > 0]
                entropy = -(hist * np.log2(hist)).sum()

                lap_var = cv2.Laplacian(img, cv2.CV_64F).var()

                rows.append({'video': int(vid),
                             'frame': int(fname.split('.')[0]),
                             'crf': label,
                             'entropy': entropy,
                             'lap_var': lap_var})

    df = pd.DataFrame(rows)
    df.to_csv(f'{OUT_DIR}/entropy_sharpness.csv', index=False)
    summary = df.groupby('crf')[['entropy', 'lap_var']].mean().round(3)
    summary = summary.reindex([c for c in CRF_ORDER if c in summary.index])
    summary.to_csv(f'{OUT_DIR}/entropy_summary.csv')
    print(summary)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    crfs    = summary.index.tolist()
    xlabels = [pretty_label(c) for c in crfs]
    colors  = [CRF_COLORS[c] for c in crfs]

    bars1 = ax1.bar(xlabels, summary['entropy'], color=colors,
                    edgecolor='black', linewidth=0.8)
    ax1.set_ylabel('Shannon entropy (bits)')
    ax1.set_title('Per-frame Shannon entropy')
    ax1.set_ylim(0, 8)
    for b, v in zip(bars1, summary['entropy']):
        ax1.text(b.get_x() + b.get_width() / 2, v, f'{v:.2f}',
                 ha='center', va='bottom', fontsize=9)
    ax1.grid(axis='y', alpha=0.3)

    bars2 = ax2.bar(xlabels, summary['lap_var'], color=colors,
                    edgecolor='black', linewidth=0.8)
    ax2.set_ylabel('Laplacian variance')
    ax2.set_title('Per-frame sharpness (Laplacian variance)')
    for b, v in zip(bars2, summary['lap_var']):
        ax2.text(b.get_x() + b.get_width() / 2, v, f'{v:.0f}',
                 ha='center', va='bottom', fontsize=9)
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{PLOT_DIR}/entropy_sharpness.png'); plt.close()
    print(f"  ✅ {PLOT_DIR}/entropy_sharpness.png")


# 3. Visual side-by-side strips 
def visual_comparison(n_samples=4):
    print("\n" + "=" * 60 + "\n3. Visual side-by-side comparisons\n" + "=" * 60)
    ref_root = os.path.join(FRAMES_BASE, CRF_FOLDERS['uncompressed'])
    vids = list_video_dirs(ref_root)

    rng = np.random.default_rng(42)
    chosen_vids = rng.choice(vids, size=min(n_samples, len(vids)), replace=False)

    for vid in chosen_vids:
        frames = list_jpgs(os.path.join(ref_root, vid))
        if not frames:
            continue
        fname = frames[len(frames) // 2]  # middle frame

        fig, axes = plt.subplots(1, len(CRF_ORDER), figsize=(18, 4))
        for ax, label in zip(axes, CRF_ORDER):
            folder = CRF_FOLDERS[label]
            p = os.path.join(FRAMES_BASE, folder, vid, fname)
            if not os.path.exists(p):
                ax.axis('off'); continue
            img = cv2.imread(p)
            if img is None:
                ax.axis('off'); continue
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ax.set_title(pretty_label(label),
                         color=CRF_COLORS[label], fontweight='bold')
            ax.axis('off')
        plt.suptitle(f'Video {vid} — frame {fname}', fontsize=12)
        plt.tight_layout()
        plt.savefig(f'{PLOT_DIR}/visual_video{vid}_frame{fname.split(".")[0]}.png')
        plt.close()
        print(f"  ✅ visual_video{vid}_frame{fname.split('.')[0]}.png")


if __name__ == '__main__':
    analyze_storage()
    analyze_edge_preservation()
    analyze_entropy_and_blocks()
    visual_comparison()
