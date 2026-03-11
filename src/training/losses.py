"""
DeforestNet - Loss Functions
Loss functions for binary deforestation segmentation with class imbalance handling.

All losses expect:
  - logits:  [B, 2, H, W] raw model output (before softmax)
  - targets: [B, H, W] integer class labels (0 or 1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation.
    
    Measures overlap between predicted and ground truth masks.
    Dice = 2 * |A ∩ B| / (|A| + |B|)
    Loss = 1 - Dice
    
    Naturally handles class imbalance since it measures overlap
    rather than per-pixel accuracy.
    """
    
    def __init__(self, smooth=1.0):
        """
        Args:
            smooth: Smoothing factor to avoid division by zero.
        """
        super().__init__()
        self.smooth = smooth
    
    def forward(self, logits, targets):
        """
        Args:
            logits:  [B, 2, H, W] raw model output
            targets: [B, H, W] ground truth labels (0 or 1)
        Returns:
            Scalar dice loss
        """
        probs = F.softmax(logits, dim=1)
        
        # One-hot encode targets: [B, H, W] -> [B, 2, H, W]
        targets_onehot = F.one_hot(targets, num_classes=2).permute(0, 3, 1, 2).float()
        
        # Compute per-class dice and average
        dice_sum = 0.0
        for c in range(2):
            pred_c = probs[:, c].contiguous().view(-1)
            true_c = targets_onehot[:, c].contiguous().view(-1)
            
            intersection = (pred_c * true_c).sum()
            union = pred_c.sum() + true_c.sum()
            
            dice_sum += (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        return 1.0 - dice_sum / 2.0


class FocalLoss(nn.Module):
    """
    Focal Loss (Lin et al., 2017).
    
    Down-weights easy examples and focuses on hard misclassified pixels.
    FL = -alpha * (1 - p_t)^gamma * log(p_t)
    
    Especially useful for deforestation detection where the deforestation
    class is rarer and harder to detect.
    """
    
    def __init__(self, alpha=None, gamma=2.0):
        """
        Args:
            alpha: Per-class weights as a list/tensor [w_class0, w_class1].
                   If None, no class weighting is applied.
            gamma: Focusing parameter. Higher = more focus on hard examples.
                   gamma=0 reduces to standard CE. Typical: 1.0 - 3.0.
        """
        super().__init__()
        self.gamma = gamma
        if alpha is not None:
            self.register_buffer('alpha', torch.tensor(alpha, dtype=torch.float32))
        else:
            self.alpha = None
    
    def forward(self, logits, targets):
        """
        Args:
            logits:  [B, 2, H, W] raw model output
            targets: [B, H, W] ground truth labels (0 or 1)
        Returns:
            Scalar focal loss
        """
        ce_loss = F.cross_entropy(logits, targets, reduction='none')  # [B, H, W]
        
        probs = F.softmax(logits, dim=1)  # [B, 2, H, W]
        # Gather the probability of the true class for each pixel
        p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1)  # [B, H, W]
        
        focal_weight = (1.0 - p_t) ** self.gamma
        
        loss = focal_weight * ce_loss
        
        if self.alpha is not None:
            alpha_t = self.alpha[targets]  # [B, H, W]
            loss = alpha_t * loss
        
        return loss.mean()


class CombinedLoss(nn.Module):
    """
    Combined loss: weighted sum of CrossEntropy + Dice + optional Focal.
    
    This is the recommended loss for DeforestNet. Each component contributes:
      - CE: Stable pixel-wise gradients, good for overall learning
      - Dice: Overlap-based, counteracts class imbalance
      - Focal: Hard example mining for boundary pixels
    """
    
    def __init__(self, class_weights=None, dice_weight=0.5, ce_weight=0.5,
                 focal_weight=0.0, focal_gamma=2.0, dice_smooth=1.0):
        """
        Args:
            class_weights: Per-class weights [w_class0, w_class1] for CE loss.
            dice_weight: Weight for dice loss component.
            ce_weight: Weight for cross-entropy component.
            focal_weight: Weight for focal loss component (0 = disabled).
            focal_gamma: Gamma for focal loss.
            dice_smooth: Smoothing for dice loss.
        """
        super().__init__()
        
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        
        # Cross-Entropy with optional class weights
        if class_weights is not None:
            weight_tensor = torch.tensor(class_weights, dtype=torch.float32)
            self.ce_loss = nn.CrossEntropyLoss(weight=weight_tensor)
        else:
            self.ce_loss = nn.CrossEntropyLoss()
        
        self.dice_loss = DiceLoss(smooth=dice_smooth)
        
        if focal_weight > 0:
            self.focal_loss = FocalLoss(alpha=class_weights, gamma=focal_gamma)
        else:
            self.focal_loss = None
    
    def forward(self, logits, targets):
        """
        Args:
            logits:  [B, 2, H, W] raw model output
            targets: [B, H, W] ground truth labels (0 or 1)
        Returns:
            total_loss: Scalar combined loss
            loss_dict: Dictionary with individual loss components
        """
        ce = self.ce_loss(logits, targets)
        dice = self.dice_loss(logits, targets)
        
        total = self.ce_weight * ce + self.dice_weight * dice
        
        loss_dict = {'ce': ce.item(), 'dice': dice.item()}
        
        if self.focal_loss is not None:
            focal = self.focal_loss(logits, targets)
            total = total + self.focal_weight * focal
            loss_dict['focal'] = focal.item()
        
        loss_dict['total'] = total.item()
        
        return total, loss_dict


def build_loss(loss_type='combined', class_weights=None, **kwargs):
    """
    Factory function to create loss functions.
    
    Args:
        loss_type: 'ce', 'dice', 'focal', or 'combined'
        class_weights: [w_class0, w_class1] for imbalance handling
        **kwargs: Additional arguments passed to the loss constructor
    
    Returns:
        Loss module instance
    """
    if loss_type == 'ce':
        weight = torch.tensor(class_weights, dtype=torch.float32) if class_weights else None
        return nn.CrossEntropyLoss(weight=weight)
    elif loss_type == 'dice':
        return DiceLoss(**kwargs)
    elif loss_type == 'focal':
        return FocalLoss(alpha=class_weights, **kwargs)
    elif loss_type == 'combined':
        return CombinedLoss(class_weights=class_weights, **kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
