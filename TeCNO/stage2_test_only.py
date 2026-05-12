import os
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

warnings.filterwarnings("ignore", message="triton not found")
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")

import configargparse
from pathlib import Path
import logging

from lightning.pytorch import Trainer

from utils.utils import argparse_summary, get_class_by_path
from utils.configargparse_arguments import build_configargparser

logging.disable(logging.WARNING)


def _to_bool(x):
    if isinstance(x, bool):
        return x
    if isinstance(x, str):
        return x.strip().lower() in {"true", "1", "yes", "y"}
    return bool(x)


def _to_none_or_str(x):
    if x is None:
        return None
    if isinstance(x, str) and x.strip().lower() in {"none", "null", ""}:
        return None
    return x


def normalize_hparams_types(hparams):
    if hasattr(hparams, "fast_dev_run"):
        hparams.fast_dev_run = _to_bool(hparams.fast_dev_run)
    if hasattr(hparams, "mstcn_causal_conv"):
        hparams.mstcn_causal_conv = _to_bool(hparams.mstcn_causal_conv)
    if hasattr(hparams, "resume_from_checkpoint"):
        hparams.resume_from_checkpoint = _to_none_or_str(hparams.resume_from_checkpoint)

    int_fields = [
        "max_epochs", "min_epochs", "num_workers", "num_sanity_val_steps",
        "check_val_every_n_epoch", "save_top_k", "log_save_interval",
        "row_log_interval", "batch_size", "mstcn_stages", "mstcn_layers",
        "mstcn_f_maps", "mstcn_f_dim", "out_features", "input_height", "input_width",
    ]
    for field in int_fields:
        if hasattr(hparams, field):
            setattr(hparams, field, int(getattr(hparams, field)))

    float_fields = [
        "learning_rate", "train_percent_check", "val_percent_check",
        "test_percent_check", "overfit_pct", "features_per_seconds",
        "features_subsampling",
    ]
    for field in float_fields:
        if hasattr(hparams, field):
            setattr(hparams, field, float(getattr(hparams, field)))

    return hparams


def find_latest_best_ckpt(train_root: Path, source_name: str) -> Path:
    source_tag = "original_pretrained" if source_name == "original" else f"{source_name}_pretrained"
    tecno_root = train_root / source_tag / "tecno_logs"
    if not tecno_root.exists():
        raise FileNotFoundError(f"Missing tecno_logs: {tecno_root}")

    run_dirs = [p for p in tecno_root.iterdir() if p.is_dir()]
    if not run_dirs:
        raise FileNotFoundError(f"No run dirs found in: {tecno_root}")

    latest_run = max(run_dirs, key=lambda p: p.stat().st_mtime)
    ckpt_dir = latest_run / "checkpoints"
    ckpts = [p for p in ckpt_dir.glob("*.ckpt") if p.name != "last.ckpt"]
    if not ckpts:
        raise FileNotFoundError(f"No best ckpt found in: {ckpt_dir}")

    return max(ckpts, key=lambda p: p.stat().st_mtime)


def main():
    parser = configargparse.ArgParser(
        config_file_parser_class=configargparse.YAMLConfigFileParser
    )
    parser.add("-c", is_config_file=True, help="config file path")
    parser.add("--source", required=True)
    parser.add("--target", required=True)
    parser.add("--train_root", required=True)
    parser.add("--cross_root", required=True)

    parser, hparams = build_configargparser(parser)

    module_path = f"modules.{hparams.module}"
    ModuleClass = get_class_by_path(module_path)
    parser = ModuleClass.add_module_specific_args(parser)

    model_path = f"models.{hparams.model}"
    ModelClass = get_class_by_path(model_path)
    parser = ModelClass.add_model_specific_args(parser)

    dataset_path = f"datasets.{hparams.dataset}"
    DatasetClass = get_class_by_path(dataset_path)
    parser = DatasetClass.add_dataset_specific_args(parser)

    args = parser.parse_args()
    args = normalize_hparams_types(args)

    pair_tag = f"{args.source}_on_{args.target}"
    args.data_root = str(Path(args.cross_root) / pair_tag / "stage2_features")
    args.output_path = Path(args.cross_root) / pair_tag / "tecno_logs"
    args.output_path.mkdir(parents=True, exist_ok=True)

    ckpt_path = find_latest_best_ckpt(Path(args.train_root), args.source)

    print("\n========== STAGE2 TEST ONLY ==========")
    print("PAIR TAG     :", pair_tag)
    print("DATA ROOT    :", args.data_root)
    print("OUTPUT PATH  :", args.output_path)
    print("STAGE2 CKPT  :", ckpt_path)
    print("======================================\n")

    model = ModelClass(hparams=args)
    dataset = DatasetClass(hparams=args)
    module = ModuleClass(args, model, dataset)

    trainer = Trainer(
        accelerator="gpu" if len(args.gpus) > 0 else "cpu",
        devices=len(args.gpus) if len(args.gpus) > 0 else 1,
        num_sanity_val_steps=args.num_sanity_val_steps,
        limit_test_batches=args.test_percent_check,
        fast_dev_run=args.fast_dev_run,
        default_root_dir=str(args.output_path),
        logger=False,
    )

    checkpoint = __import__("torch").load(ckpt_path, map_location="cpu", weights_only=False)
    module.load_state_dict(checkpoint["state_dict"], strict=False)

    argparse_summary(args, parser)
    trainer.test(model=module)


if __name__ == "__main__":
    main()
    