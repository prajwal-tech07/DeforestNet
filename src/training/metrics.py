"""
DeforestNet - Evaluation Metrics
Segmentation metrics for monitoring training and evaluating model performance.

All metrics operate on:
  - predictions: [B, H, W] integer class predictions (argmax of logits)
  - targets:     [B, H, W] integer ground truth labels (0 or 1)

Provides both per-batch computation and an accumulator class for
computing metrics across an entire epoch/dataset.
"""

import torch
import numpy as np


def compute_confusion_matrix(preds, targets, num_classes=2):
    """
    Compute confusion matrix from predictions and targets.
    
    Args:
        preds:   [B, H, W] predicted class labels
        targets: [B, H, W] ground truth labels
        num_classes: Number of classes
    
    Returns:
        [num_classes, num_classes] confusion matrix where
        cm[i, j] = count of pixels with true=i, pred=j
    """
    preds_flat = preds.view(-1)
    targets_flat = targets.view(-1)
    
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long,
                     device=preds.device)
    for t in range(num_classes):
        for p in range(num_classes):
            cm[t, p] = ((targets_flat == t) & (preds_flat == p)).sum()
    
    return cm


def iou_from_cm(cm):
    """
    Compute per-class IoU (Intersection over Union) from confusion matrix.
    
    IoU_c = TP_c / (TP_c + FP_c + FN_c)
    """
    iou = torch.zeros(cm.shape[0], dtype=torch.float32)
    for c in range(cm.shape[0]):
        tp = cm[c, c].float()
        fp = cm[:, c].sum().float() - tp
        fn = cm[c, :].sum().float() - tp
        denom = tp + fp + fn
        iou[c] = tp / denom if denom > 0 else 0.0
    return iou


def dice_from_cm(cm):
    """
    Compute per-class Dice coefficient from confusion matrix.
    
    Dice_c = 2*TP_c / (2*TP_c + FP_c + FN_c)
    """
    dice = torch.zeros(cm.shape[0], dtype=torch.float32)
    for c in range(cm.shape[0]):
        tp = cm[c, c].float()
        fp = cm[:, c].sum().float() - tp
        fn = cm[c, :].sum().float() - tp
        denom = 2.0 * tp + fp + fn
        dice[c] = (2.0 * tp) / denom if denom > 0 else 0.0
    return dice


def precision_from_cm(cm):
    """
    Compute per-class precision from confusion matrix.
    
    Precision_c = TP_c / (TP_c + FP_c)
    """
    precision = torch.zeros(cm.shape[0], dtype=torch.float32)
    for c in range(cm.shape[0]):
        tp = cm[c, c].float()
        fp = cm[:, c].sum().float() - tp
        denom = tp + fp
        precision[c] = tp / denom if denom > 0 else 0.0
    return precision


def recall_from_cm(cm):
    """
    Compute per-class recall from confusion matrix.
    
    Recall_c = TP_c / (TP_c + FN_c)
    """
    recall = torch.zeros(cm.shape[0], dtype=torch.float32)
    for c in range(cm.shape[0]):
        tp = cm[c, c].float()
        fn = cm[c, :].sum().float() - tp
        denom = tp + fn
        recall[c] = tp / denom if denom > 0 else 0.0
    return recall


def f1_from_cm(cm):
    """
    Compute per-class F1 score from confusion matrix.
    
    F1_c = 2 * Precision_c * Recall_c / (Precision_c + Recall_c)
    """
    prec = precision_from_cm(cm)
    rec = recall_from_cm(cm)
    f1 = torch.zeros(cm.shape[0], dtype=torch.float32)
    for c in range(cm.shape[0]):
        denom = prec[c] + rec[c]
        f1[c] = (2.0 * prec[c] * rec[c]) / denom if denom > 0 else 0.0
    return f1


def overall_accuracy(cm):
    """
    Compute overall pixel accuracy.
    
    Accuracy = (TP_0 + TP_1) / total_pixels
    """
    correct = cm.diagonal().sum().float()
    total = cm.sum().float()
    return correct / total if total > 0 else torch.tensor(0.0)


class MetricTracker:
    """
    Accumulates confusion matrix across batches and computes
    epoch-level metrics.
    
    Usage:
        tracker = MetricTracker(num_classes=2, class_names=['Non-Deforest', 'Deforest'])
        for batch in loader:
            preds = model(images).argmax(dim=1)
            tracker.update(preds, masks)
        metrics = tracker.compute()
        tracker.reset()
    """
    
    def __init__(self, num_classes=2, class_names=None):
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class_{i}" for i in range(num_classes)]
        self.cm = torch.zeros(num_classes, num_classes, dtype=torch.long)
    
    def reset(self):
        """Reset accumulated confusion matrix for new epoch."""
        self.cm.zero_()
    
    def update(self, preds, targets):
        """
        Update with a batch of predictions and targets.
        
        Args:
            preds:   [B, H, W] predicted class labels (integer)
            targets: [B, H, W] ground truth labels (integer)
        """
        batch_cm = compute_confusion_matrix(
            preds.detach().cpu(), targets.detach().cpu(), self.num_classes
        )
        self.cm += batch_cm
    
    def compute(self):
        """
        Compute all metrics from the accumulated confusion matrix.
        
        Returns:
            Dictionary with all computed metrics
        """
        iou = iou_from_cm(self.cm)
        dice = dice_from_cm(self.cm)
        prec = precision_from_cm(self.cm)
        rec = recall_from_cm(self.cm)
        f1 = f1_from_cm(self.cm)
        acc = overall_accuracy(self.cm)
        
        metrics = {
            'accuracy': acc.item(),
            'mean_iou': iou.mean().item(),
            'mean_dice': dice.mean().item(),
            'mean_f1': f1.mean().item(),
        }
        
        # Per-class metrics
        for c in range(self.num_classes):
            name = self.class_names[c]
            metrics[f'{name}_iou'] = iou[c].item()
            metrics[f'{name}_dice'] = dice[c].item()
            metrics[f'{name}_precision'] = prec[c].item()
            metrics[f'{name}_recall'] = rec[c].item()
            metrics[f'{name}_f1'] = f1[c].item()
        
        return metrics
    
    def summary(self):
        """Return a formatted string summary of current metrics."""
        m = self.compute()
        lines = [
            f"Accuracy: {m['accuracy']:.4f}",
            f"Mean IoU: {m['mean_iou']:.4f}",
            f"Mean Dice: {m['mean_dice']:.4f}",
        ]
        for c in range(self.num_classes):
            name = self.class_names[c]
            lines.append(
                f"  {name}: IoU={m[f'{name}_iou']:.4f}  "
                f"Dice={m[f'{name}_dice']:.4f}  "
                f"P={m[f'{name}_precision']:.4f}  "
                f"R={m[f'{name}_recall']:.4f}  "
                f"F1={m[f'{name}_f1']:.4f}"
            )
        return "\n".join(lines)
