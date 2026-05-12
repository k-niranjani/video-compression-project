from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# =========================================================
# CHANGE ONLY THESE
# =========================================================
BASELINE_ROOT = Path(r"E:\VC\experiments\paper_pretrained")
CROSS_ROOT = Path(r"E:\VC\experiments\cross_test")
OUT_DIR = Path(r"E:\VC\experiments\tecno_plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_LEVELS = ["original", "crf18", "crf23", "crf28", "crf35", "crf51"]
TEST_LEVELS = ["original", "crf18", "crf23", "crf28", "crf35", "crf51"]
MODEL_NAME = "TeCNO"
# =========================================================


def get_baseline_dir(level: str) -> Path:
    tag = "original_pretrained" if level == "original" else f"{level}_pretrained"
    return BASELINE_ROOT / tag / "tecno_logs"


def get_cross_dir(train_level: str, test_level: str) -> Path:
    return CROSS_ROOT / f"{train_level}_on_{test_level}" / "tecno_logs"


def find_baseline_results_dir(tecno_logs_dir: Path) -> Path | None:
    """
    Baseline layout:
    paper_pretrained/<tag>/tecno_logs/<run_folder>/results
    """
    if not tecno_logs_dir.exists():
        return None

    run_dirs = [p for p in tecno_logs_dir.iterdir() if p.is_dir()]
    if not run_dirs:
        return None

    latest_run = max(run_dirs, key=lambda p: p.stat().st_mtime)
    results_dir = latest_run / "results"
    return results_dir if results_dir.exists() else None


def find_cross_results_dir(tecno_logs_dir: Path) -> Path | None:
    """
    Cross-test layout from your screenshots:
    cross_test/<pair>/tecno_logs/results
    """
    if not tecno_logs_dir.exists():
        return None

    direct_results = tecno_logs_dir / "results"
    if direct_results.exists():
        return direct_results

    return None


def get_results_dir(train_level: str, test_level: str) -> Path | None:
    if train_level == test_level:
        return find_baseline_results_dir(get_baseline_dir(train_level))
    return find_cross_results_dir(get_cross_dir(train_level, test_level))


def load_summary_metrics(train_level: str, test_level: str) -> dict | None:
    results_dir = get_results_dir(train_level, test_level)
    if results_dir is None:
        return None

    summary_csv = results_dir / "summary_metrics.csv"
    if not summary_csv.exists():
        return None

    df = pd.read_csv(summary_csv)
    if df.empty:
        return None

    row = df.iloc[0].to_dict()

    return {
        "accuracy": row.get("mean_acc", np.nan),
        "macro_precision": row.get("mean_prec_macro", row.get("mean_precision", np.nan)),
        "macro_recall": row.get("mean_rec_macro", row.get("mean_recall", np.nan)),
        "macro_f1": row.get("mean_f1_macro", row.get("mean_f1", np.nan)),
        "macro_auc": row.get("mean_auc_macro", row.get("mean_auc", np.nan)),
        "macro_jaccard": row.get("mean_jac_macro", row.get("mean_jaccard", np.nan)),
    }


def load_phase_metrics(train_level: str, test_level: str) -> pd.DataFrame | None:
    results_dir = get_results_dir(train_level, test_level)
    if results_dir is None:
        return None

    phase_csv = results_dir / "phase_metrics.csv"
    if not phase_csv.exists():
        return None

    return pd.read_csv(phase_csv)


def build_metric_matrix(metric_name: str) -> pd.DataFrame:
    matrix = pd.DataFrame(index=TRAIN_LEVELS, columns=TEST_LEVELS, dtype=float)

    for tr in TRAIN_LEVELS:
        for te in TEST_LEVELS:
            metrics = load_summary_metrics(tr, te)
            if metrics is not None:
                matrix.loc[tr, te] = metrics.get(metric_name, np.nan)

    return matrix


def annotate_heatmap(ax, data):
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = data[i, j]
            text = "NA" if np.isnan(value) else f"{value:.3f}"
            ax.text(j, i, text, ha="center", va="center", fontsize=8)


def plot_heatmap(matrix: pd.DataFrame, title: str, filename: str):
    fig, ax = plt.subplots(figsize=(8, 6))
    arr = matrix.values.astype(float)

    masked = np.ma.masked_invalid(arr)
    im = ax.imshow(masked, aspect="auto")

    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns)
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_yticklabels(matrix.index)

    ax.set_xlabel("Test Compression")
    ax.set_ylabel("Train Compression")
    ax.set_title(title)

    annotate_heatmap(ax, arr)
    fig.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig(OUT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_small_multiples(metric_name: str, ylabel: str, filename: str):
    fig, axes = plt.subplots(3, 2, figsize=(11, 12))
    axes = axes.flatten()
    x = np.arange(len(TEST_LEVELS))

    for idx, train_level in enumerate(TRAIN_LEVELS):
        ax = axes[idx]
        y = []

        for test_level in TEST_LEVELS:
            metrics = load_summary_metrics(train_level, test_level)
            y.append(np.nan if metrics is None else metrics.get(metric_name, np.nan))

        ax.plot(x, y, marker="o")
        ax.set_xticks(x)
        ax.set_xticklabels(TEST_LEVELS)
        ax.set_title(f"Train = {train_level}")
        ax.set_xlabel("Test Compression")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_diagonal(metrics_to_plot, filename: str):
    fig, axes = plt.subplots(len(metrics_to_plot), 1, figsize=(8, 10))

    if len(metrics_to_plot) == 1:
        axes = [axes]

    x = np.arange(len(TRAIN_LEVELS))

    for ax, (metric_name, ylabel) in zip(axes, metrics_to_plot):
        y = []
        for lvl in TRAIN_LEVELS:
            metrics = load_summary_metrics(lvl, lvl)
            y.append(np.nan if metrics is None else metrics.get(metric_name, np.nan))

        ax.plot(x, y, marker="o")
        ax.set_xticks(x)
        ax.set_xticklabels(TRAIN_LEVELS)
        ax.set_title(f"Matched Train/Test Compression: {ylabel} vs Compression")
        ax.set_xlabel("Compression")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_orig_only(metrics_to_plot, filename: str):
    fig, axes = plt.subplots(len(metrics_to_plot), 1, figsize=(8, 10))

    if len(metrics_to_plot) == 1:
        axes = [axes]

    x = np.arange(len(TEST_LEVELS))

    for ax, (metric_name, ylabel) in zip(axes, metrics_to_plot):
        y = []
        for test_level in TEST_LEVELS:
            metrics = load_summary_metrics("original", test_level)
            y.append(np.nan if metrics is None else metrics.get(metric_name, np.nan))

        ax.plot(x, y, marker="o")
        ax.set_xticks(x)
        ax.set_xticklabels(TEST_LEVELS)
        ax.set_title(f"Train on original, Test across Compression: {ylabel}")
        ax.set_xlabel("Test Compression")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_per_phase(metric_col: str = "f1", filename: str = "per_phase_f1.png"):
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    axes = axes.flatten()
    x = np.arange(len(TEST_LEVELS))

    for idx, train_level in enumerate(TRAIN_LEVELS):
        ax = axes[idx]
        phase_series = {}

        for test_level in TEST_LEVELS:
            df = load_phase_metrics(train_level, test_level)
            if df is None or metric_col not in df.columns:
                continue

            phase_col = "phase" if "phase" in df.columns else "phase_id"
            grouped = df.groupby(phase_col)[metric_col].mean().reset_index()

            for _, row in grouped.iterrows():
                phase_name = str(row[phase_col])
                metric_value = row[metric_col]
                phase_series.setdefault(phase_name, []).append((test_level, metric_value))

        for phase_name, entries in phase_series.items():
            value_map = {te: val for te, val in entries}
            y = [value_map.get(te, np.nan) for te in TEST_LEVELS]
            ax.plot(x, y, marker="o", label=phase_name)

        ax.set_xticks(x)
        ax.set_xticklabels(TEST_LEVELS)
        ax.set_title(f"Train = {train_level}")
        ax.set_xlabel("Test Compression")
        ax.set_ylabel(metric_col)
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.savefig(OUT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_all_results_table():
    rows = []
    for tr in TRAIN_LEVELS:
        for te in TEST_LEVELS:
            metrics = load_summary_metrics(tr, te)
            if metrics is None:
                continue

            row = {"train": tr, "test": te}
            row.update(metrics)
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "all_results_table.csv", index=False)


def print_missing_pairs():
    print("\n===== PAIR CHECK =====")
    for tr in TRAIN_LEVELS:
        for te in TEST_LEVELS:
            results_dir = get_results_dir(tr, te)
            status = "FOUND" if results_dir is not None else "MISSING"
            print(f"{tr:>8} -> {te:<8} : {status} | {results_dir}")
    print("======================\n")


def main():
    print_missing_pairs()

    # Heatmaps
    plot_heatmap(
        build_metric_matrix("accuracy"),
        f"{MODEL_NAME}: Accuracy (Train vs Test Compression)",
        "heatmap_accuracy.png",
    )
    plot_heatmap(
        build_metric_matrix("macro_precision"),
        f"{MODEL_NAME}: Macro Precision (Train vs Test Compression)",
        "heatmap_macro_precision.png",
    )
    plot_heatmap(
        build_metric_matrix("macro_f1"),
        f"{MODEL_NAME}: Macro F1 (Train vs Test Compression)",
        "heatmap_macro_f1.png",
    )
    plot_heatmap(
        build_metric_matrix("macro_auc"),
        f"{MODEL_NAME}: Macro AUC (Train vs Test Compression)",
        "heatmap_macro_auc.png",
    )

    # Small-multiple metric pages
    plot_small_multiples("accuracy", "Accuracy", "small_multiples_accuracy.png")
    plot_small_multiples("macro_precision", "Macro Precision", "small_multiples_macro_precision.png")
    plot_small_multiples("macro_f1", "Macro F1", "small_multiples_macro_f1.png")
    plot_small_multiples("macro_auc", "Macro AUC", "small_multiples_macro_auc.png")

    # Diagonal and orig-only
    metrics_for_curves = [
        ("accuracy", "Accuracy"),
        ("macro_f1", "Macro F1"),
        ("macro_auc", "Macro AUC"),
    ]
    plot_diagonal(metrics_for_curves, "diagonal_plots.png")
    plot_orig_only(metrics_for_curves, "orig_only_plots.png")

    # Per-phase
    plot_per_phase(metric_col="f1", filename="per_phase_f1.png")

    # Combined raw table
    save_all_results_table()

    print(f"Saved all plots to: {OUT_DIR}")


if __name__ == "__main__":
    main()