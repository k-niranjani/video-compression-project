import torch
import numpy as np
from typing import Any, Optional
from torchmetrics import Metric


class PrecisionOverClasses:
    def __init__(self, num_classes: int = 1, threshold: float = 0.5, average: str = "micro",
                 multilabel: bool = False, compute_on_step: bool = True,
                 dist_sync_on_step: bool = False, process_group: Optional[Any] = None):
        self.num_classes = num_classes

    def __call__(self, preds: torch.Tensor, target: torch.Tensor):
        preds = preds.view(-1).long()
        target = target.view(-1).long()

        precisions = []
        for c in range(self.num_classes):
            tp = ((preds == c) & (target == c)).sum().float()
            fp = ((preds == c) & (target != c)).sum().float()

            if tp + fp == 0:
                precisions.append(torch.tensor(float("nan"), device=preds.device))
            else:
                precisions.append(tp / (tp + fp))

        return torch.stack(precisions)


class RecallOverClasse:
    def __init__(self, num_classes: int = 1, threshold: float = 0.5, average: str = "micro",
                 multilabel: bool = False, compute_on_step: bool = True,
                 dist_sync_on_step: bool = False, process_group: Optional[Any] = None):
        self.num_classes = num_classes

    def __call__(self, preds: torch.Tensor, target: torch.Tensor):
        preds = preds.view(-1).long()
        target = target.view(-1).long()

        recalls = []
        for c in range(self.num_classes):
            tp = ((preds == c) & (target == c)).sum().float()
            fn = ((preds != c) & (target == c)).sum().float()

            if tp + fn == 0:
                recalls.append(torch.tensor(float("nan"), device=preds.device))
            else:
                recalls.append(tp / (tp + fn))

        return torch.stack(recalls)


class AccuracyStages(Metric):
    def __init__(self, num_stages=1, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.num_stages = num_stages

        self.add_state(
            "correct",
            default=torch.zeros(num_stages, dtype=torch.float),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "total",
            default=torch.zeros(num_stages, dtype=torch.float),
            dist_reduce_fx="sum",
        )

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        preds expected shape: [S, B, C, T]
        target expected shape: [B, T] or [T]
        """
        target = target.squeeze().long()

        for s in range(self.num_stages):
            stage_pred = preds[s].squeeze()

            if stage_pred.dim() == 2:
                # [C, T] -> [T]
                pred_labels = torch.argmax(stage_pred, dim=0)
            elif stage_pred.dim() == 3:
                # [B, C, T] -> [B, T]
                pred_labels = torch.argmax(stage_pred, dim=1)
            else:
                raise ValueError(f"Unexpected stage_pred shape: {stage_pred.shape}")

            assert pred_labels.shape == target.shape, (
                f"Prediction/target shape mismatch: {pred_labels.shape} vs {target.shape}"
            )

            correct = (pred_labels == target).float().sum()
            total = torch.tensor(target.numel(), device=target.device, dtype=torch.float)

            self.correct[s] += correct
            self.total[s] += total

    def compute(self):
        acc_list = []
        for s in range(self.num_stages):
            acc_list.append(self.correct[s] / torch.clamp(self.total[s], min=1.0))
        return acc_list


def calc_average_over_metric(metric_list, normlist):
    for i in metric_list:
        metric_list[i] = np.asarray([0 if value == "None" else value for value in metric_list[i]])
        if normlist[i] == 0:
            metric_list[i] = 0
        else:
            metric_list[i] = metric_list[i].sum() / normlist[i]
    return metric_list


def create_print_output(print_dict, space_desc, space_item):
    msg = ""
    for key, value in print_dict.items():
        msg += f"{key:<{space_desc}}"
        for i in value:
            msg += f"{i:>{space_item}}"
        msg += "\n"
    msg = msg[:-1]
    return msg