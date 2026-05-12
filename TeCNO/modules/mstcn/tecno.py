import logging
from pathlib import Path

import torch
from torch import optim, nn
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import lightning.pytorch as pl
import numpy as np
import pandas as pd

from utils.metric_helper import AccuracyStages, RecallOverClasse, PrecisionOverClasses


class TeCNO(pl.LightningModule):
    def __init__(self, hparams, model, dataset):
        super().__init__()
        self.save_hyperparameters(vars(hparams) if not isinstance(hparams, dict) else hparams)

        self.batch_size = hparams.batch_size
        self.dataset = dataset
        self.model = model

        self.weights_train = np.asarray(self.dataset.weights["train"])
        self.ce_loss = nn.CrossEntropyLoss(
            weight=torch.from_numpy(self.weights_train).float()
        )

        self.init_metrics()

        # Lightning 2.x epoch-end aggregation
        self.train_outputs = []
        self.val_outputs = []
        self.test_outputs = []

    def init_metrics(self):
        self.train_acc_stages = AccuracyStages(num_stages=self.hparams.mstcn_stages)
        self.val_acc_stages = AccuracyStages(num_stages=self.hparams.mstcn_stages)
        self.test_acc_stages = AccuracyStages(num_stages=self.hparams.mstcn_stages)

        self.max_acc_last_stage = {"epoch": 0, "acc": 0}

        self.precision_metric = PrecisionOverClasses(num_classes=7)
        self.recall_metric = RecallOverClasse(num_classes=7)

    def forward(self, x):
        # x: [B, T, F]
        video_fe = x.transpose(2, 1)  # [B, F, T]
        y_classes = self.model.forward(video_fe)
        y_classes = torch.softmax(y_classes, dim=2)
        return y_classes

    def loss_function(self, y_classes, labels):
        stages = y_classes.shape[0]
        clc_loss = 0.0

        for j in range(stages):
            # y_classes[j]: [B, C, T]
            p_classes = y_classes[j].squeeze().transpose(1, 0)
            ce_loss = self.ce_loss(p_classes, labels.squeeze())
            clc_loss += ce_loss

        clc_loss = clc_loss / float(stages)
        return clc_loss

    def calc_precision_and_recall(self, y_pred, y_true):
        y_true = y_true.squeeze().long()
        y_pred_last = y_pred[-1].squeeze()

        if y_pred_last.dim() == 2:
            y_max_pred = torch.argmax(y_pred_last, dim=0)
        elif y_pred_last.dim() == 1:
            y_max_pred = y_pred_last.long()
        else:
            raise ValueError(f"Unexpected y_pred_last shape: {y_pred_last.shape}")

        precision = self.precision_metric(y_max_pred, y_true)
        recall = self.recall_metric(y_max_pred, y_true)
        return precision, recall, y_max_pred, y_true

    @staticmethod
    def _safe_nanmean(x: torch.Tensor) -> torch.Tensor:
        valid = ~torch.isnan(x)
        if valid.any():
            return x[valid].mean()
        return torch.tensor(float("nan"), device=x.device)

    @staticmethod
    def _f1_from_pr(precision: torch.Tensor, recall: torch.Tensor) -> torch.Tensor:
        denom = precision + recall
        f1 = torch.full_like(precision, float("nan"))
        valid = denom > 0
        f1[valid] = 2 * precision[valid] * recall[valid] / denom[valid]
        return f1

    @staticmethod
    def _jaccard_from_pr(precision: torch.Tensor, recall: torch.Tensor) -> torch.Tensor:
        # IoU from precision and recall: IoU = PR / (P + R - PR)
        denom = precision + recall - (precision * recall)
        jac = torch.full_like(precision, float("nan"))
        valid = denom > 0
        jac[valid] = (precision[valid] * recall[valid]) / denom[valid]
        return jac

    def _per_phase_metrics(self, preds: torch.Tensor, target: torch.Tensor):
        preds = preds.view(-1).long()
        target = target.view(-1).long()
        num_classes = len(self.dataset.class_labels)

        phase_rows = []
        for c in range(num_classes):
            tp = ((preds == c) & (target == c)).sum().float()
            fp = ((preds == c) & (target != c)).sum().float()
            fn = ((preds != c) & (target == c)).sum().float()

            precision = tp / (tp + fp) if (tp + fp) > 0 else torch.tensor(float("nan"), device=preds.device)
            recall = tp / (tp + fn) if (tp + fn) > 0 else torch.tensor(float("nan"), device=preds.device)

            if not torch.isnan(precision) and not torch.isnan(recall) and (precision + recall) > 0:
                f1 = 2 * precision * recall / (precision + recall)
            else:
                f1 = torch.tensor(float("nan"), device=preds.device)

            denom = tp + fp + fn
            jaccard = tp / denom if denom > 0 else torch.tensor(float("nan"), device=preds.device)

            phase_rows.append({
                "phase_id": c,
                "phase": self.dataset.class_labels[c],
                "precision": float(precision.cpu()) if not torch.isnan(precision) else np.nan,
                "recall": float(recall.cpu()) if not torch.isnan(recall) else np.nan,
                "f1": float(f1.cpu()) if not torch.isnan(f1) else np.nan,
                "jaccard": float(jaccard.cpu()) if not torch.isnan(jaccard) else np.nan,
            })

        return phase_rows

    def log_average_precision_recall(self, outputs, step="val"):
        if len(outputs) == 0:
            return

        precision_list = [o["precision"] for o in outputs]
        recall_list = [o["recall"] for o in outputs]

        x = torch.stack(precision_list)
        y = torch.stack(recall_list)

        phase_avg_precision = [
            torch.mean(x[~x[:, n].isnan(), n]) for n in range(x.shape[1])
        ]
        phase_avg_recall = [
            torch.mean(y[~y[:, n].isnan(), n]) for n in range(y.shape[1])
        ]

        phase_avg_precision = torch.stack(phase_avg_precision)
        phase_avg_recall = torch.stack(phase_avg_recall)

        phase_avg_precision_over_video = phase_avg_precision[
            ~phase_avg_precision.isnan()
        ].mean()
        phase_avg_recall_over_video = phase_avg_recall[
            ~phase_avg_recall.isnan()
        ].mean()

        self.log(f"{step}_avg_precision", phase_avg_precision_over_video, on_epoch=True, on_step=False, prog_bar=False)
        self.log(f"{step}_avg_recall", phase_avg_recall_over_video, on_epoch=True, on_step=False, prog_bar=False)

    def training_step(self, batch, batch_idx):
        stem, y_hat, y_true = batch

        y_pred = self.forward(stem)
        loss = self.loss_function(y_pred, y_true)

        precision, recall, _, _ = self.calc_precision_and_recall(y_pred, y_true)

        self.train_acc_stages.update(y_pred, y_true)
        acc_stages = self.train_acc_stages.compute()

        acc_stages_dict = {
            f"train_S{s+1}_acc": acc_stages[s] for s in range(len(acc_stages))
        }
        acc_stages_dict["train_acc"] = acc_stages_dict[f"train_S{len(acc_stages)}_acc"]

        self.log("loss", loss, on_epoch=True, on_step=True, prog_bar=True, batch_size=stem.shape[0])
        self.log_dict(acc_stages_dict, on_epoch=True, on_step=False, prog_bar=False, batch_size=stem.shape[0])

        out = {"loss": loss, "precision": precision, "recall": recall}
        self.train_outputs.append(out)
        return out

    def on_train_epoch_end(self):
        self.log_average_precision_recall(self.train_outputs, step="train")
        self.train_outputs.clear()
        self.train_acc_stages.reset()

    def validation_step(self, batch, batch_idx):
        stem, y_hat, y_true = batch

        y_pred = self.forward(stem)
        val_loss = self.loss_function(y_pred, y_true)

        precision, recall, _, _ = self.calc_precision_and_recall(y_pred, y_true)

        self.val_acc_stages.update(y_pred, y_true)
        acc_stages = self.val_acc_stages.compute()

        metric_dict = {
            f"val_S{s + 1}_acc": acc_stages[s] for s in range(len(acc_stages))
        }
        metric_dict["val_acc"] = metric_dict[f"val_S{len(acc_stages)}_acc"]

        self.log("val_loss", val_loss, on_epoch=True, prog_bar=True, on_step=False, batch_size=stem.shape[0])
        self.log_dict(metric_dict, on_epoch=True, on_step=False, prog_bar=False, batch_size=stem.shape[0])

        metric_dict["precision"] = precision
        metric_dict["recall"] = recall
        self.val_outputs.append(metric_dict)
        return metric_dict

    def on_validation_epoch_end(self):
        if len(self.val_outputs) == 0:
            return

        outputs = self.val_outputs
        val_acc_stage_last_epoch = torch.stack([o["val_acc"] for o in outputs]).mean()

        if val_acc_stage_last_epoch > self.max_acc_last_stage["acc"]:
            self.max_acc_last_stage["acc"] = val_acc_stage_last_epoch
            self.max_acc_last_stage["epoch"] = int(self.current_epoch)

        self.log("val_max_acc_last_stage", self.max_acc_last_stage["acc"], on_epoch=True, on_step=False)
        self.log_average_precision_recall(outputs, step="val")

        self.val_outputs.clear()
        self.val_acc_stages.reset()

    def test_step(self, batch, batch_idx):
        stem, y_hat, y_true = batch

        y_pred = self.forward(stem)
        test_loss = self.loss_function(y_pred, y_true)

        precision, recall, y_max_pred, y_true_flat = self.calc_precision_and_recall(y_pred, y_true)

        self.test_acc_stages.update(y_pred, y_true)
        acc_stages = self.test_acc_stages.compute()

        metric_dict = {
            f"test_S{s + 1}_acc": acc_stages[s] for s in range(len(acc_stages))
        }
        metric_dict["test_acc"] = metric_dict[f"test_S{len(acc_stages)}_acc"]

        # macro metrics for this video
        prec_macro = self._safe_nanmean(precision)
        rec_macro = self._safe_nanmean(recall)
        f1_per_phase = self._f1_from_pr(precision, recall)
        f1_macro = self._safe_nanmean(f1_per_phase)
        jac_per_phase = self._jaccard_from_pr(precision, recall)
        jac_macro = self._safe_nanmean(jac_per_phase)

        self.log("test_loss", test_loss, on_epoch=True, prog_bar=True, on_step=False, batch_size=stem.shape[0])
        self.log_dict(metric_dict, on_epoch=True, on_step=False, prog_bar=False, batch_size=stem.shape[0])

        metric_dict["precision"] = precision
        metric_dict["recall"] = recall
        metric_dict["prec_macro"] = prec_macro
        metric_dict["rec_macro"] = rec_macro
        metric_dict["f1_macro"] = f1_macro
        metric_dict["jac_macro"] = jac_macro
        metric_dict["preds"] = y_max_pred.detach().cpu()
        metric_dict["targets"] = y_true_flat.detach().cpu()

        self.test_outputs.append(metric_dict)
        return metric_dict

    def on_test_epoch_end(self):
        if len(self.test_outputs) == 0:
            return

        outputs = self.test_outputs

        test_acc = torch.stack([o["test_acc"] for o in outputs]).mean()
        self.log("test_acc", test_acc, on_epoch=True, on_step=False)
        self.log_average_precision_recall(outputs, step="test")

        # -------- save CSVs --------
        results_dir = Path(self.trainer.default_root_dir) / "results"
        results_dir.mkdir(parents=True, exist_ok=True)

        video_rows = []
        all_phase_rows = []

        for idx, o in enumerate(outputs, start=1):
            preds = o["preds"]
            targets = o["targets"]

            phase_rows = self._per_phase_metrics(preds, targets)
            for row in phase_rows:
                row["video_id"] = idx
                all_phase_rows.append(row)

            video_rows.append({
                "video_id": idx,
                "acc": float(o["test_acc"].cpu()),
                "prec_macro": float(o["prec_macro"].cpu()) if not torch.isnan(o["prec_macro"]) else np.nan,
                "rec_macro": float(o["rec_macro"].cpu()) if not torch.isnan(o["rec_macro"]) else np.nan,
                "f1_macro": float(o["f1_macro"].cpu()) if not torch.isnan(o["f1_macro"]) else np.nan,
                "jac_macro": float(o["jac_macro"].cpu()) if not torch.isnan(o["jac_macro"]) else np.nan,
            })

        df_video = pd.DataFrame(video_rows)
        df_phase = pd.DataFrame(all_phase_rows)

        summary = {
            "mean_acc": [df_video["acc"].mean()],
            "mean_prec_macro": [df_video["prec_macro"].mean()],
            "mean_rec_macro": [df_video["rec_macro"].mean()],
            "mean_f1_macro": [df_video["f1_macro"].mean()],
            "mean_jac_macro": [df_video["jac_macro"].mean()],
        }
        df_summary = pd.DataFrame(summary)

        df_video.to_csv(results_dir / "video_metrics.csv", index=False)
        df_phase.to_csv(results_dir / "phase_metrics.csv", index=False)
        df_summary.to_csv(results_dir / "summary_metrics.csv", index=False)

        print(f"Saved: {results_dir / 'video_metrics.csv'}")
        print(f"Saved: {results_dir / 'phase_metrics.csv'}")
        print(f"Saved: {results_dir / 'summary_metrics.csv'}")

        self.test_outputs.clear()
        self.test_acc_stages.reset()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

    def __dataloader(self, split=None):
        dataset = self.dataset.data[split]
        should_shuffle = split == "train"

        train_sampler = None
        if self.trainer is not None and getattr(self.trainer, "world_size", 1) > 1:
            train_sampler = DistributedSampler(dataset, shuffle=should_shuffle)
            should_shuffle = False

        print(f"split: {split} - shuffle: {should_shuffle}")

        loader = DataLoader(
            dataset=dataset,
            batch_size=self.hparams.batch_size,
            shuffle=should_shuffle,
            sampler=train_sampler,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
        )
        return loader

    def train_dataloader(self):
        dataloader = self.__dataloader(split="train")
        logging.info(f"training data loader called - size: {len(dataloader.dataset)}")
        return dataloader

    def val_dataloader(self):
        dataloader = self.__dataloader(split="val")
        logging.info(f"validation data loader called - size: {len(dataloader.dataset)}")
        return dataloader

    def test_dataloader(self):
        dataloader = self.__dataloader(split="test")
        logging.info(f"test data loader called - size: {len(dataloader.dataset)}")
        return dataloader

    @staticmethod
    def add_module_specific_args(parser):  # pragma: no cover
        regressiontcn = parser.add_argument_group(
            title="regression tcn specific args options"
        )
        regressiontcn.add_argument("--learning_rate", default=0.001, type=float)
        regressiontcn.add_argument("--optimizer_name", default="adam", type=str)
        regressiontcn.add_argument("--batch_size", default=1, type=int)
        return parser