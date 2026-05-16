import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        # Flatten
        pred = pred.view(-1)
        target = target.view(-1)
        
        # Intersection
        intersection = (pred * target).sum()
        
        # Dice Coeff
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        
        return 1 - dice

class CombinedLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, pred_logits, target, **kwargs):
        # BCEWithLogitsLoss takes logits
        bce_loss = self.bce(pred_logits, target)
        
        # DiceLoss expects probabilities (0-1)
        pred_probs = torch.sigmoid(pred_logits)
        dice_loss = self.dice(pred_probs, target)
        
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


class PhysicsInformedLoss(nn.Module):
    """
    Physics-Informed Loss for UHI Detection.
    
    Extends BCE+Dice with two soft physics penalty terms:
    1. Vegetation penalty: penalizes UHI predictions where NDVI > threshold (vegetation)
    2. Temperature penalty: penalizes UHI predictions where LST is below scene mean (cold areas)
    
    This embeds domain knowledge directly into the training loop rather than
    applying it only as post-processing.
    
    L_total = w_bce * L_BCE + w_dice * L_Dice + λ_veg * L_veg + λ_temp * L_temp
    """
    def __init__(self, bce_weight=0.5, dice_weight=0.5, 
                 lambda_veg=0.1, lambda_temp=0.1,
                 ndvi_threshold=0.4, water_ndvi_threshold=0.0):
        super(PhysicsInformedLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.lambda_veg = lambda_veg
        self.lambda_temp = lambda_temp
        self.ndvi_threshold = ndvi_threshold
        self.water_ndvi_threshold = water_ndvi_threshold
    
    def forward(self, pred_logits, target, ndvi=None, lst=None):
        """
        Args:
            pred_logits: (B, 1, H, W) raw logits from U-Net
            target: (B, 1, H, W) ground truth weak labels
            ndvi: (B, 1, H, W) NDVI values (un-normalized). Optional.
            lst: (B, 1, H, W) LST values in Kelvin (un-normalized). Optional.
        """
        # Standard supervised losses
        bce_loss = self.bce(pred_logits, target)
        pred_probs = torch.sigmoid(pred_logits)
        dice_loss = self.dice(pred_probs, target)
        
        total = self.bce_weight * bce_loss + self.dice_weight * dice_loss
        
        # --- Physics Penalty 1: Vegetation / Water Constraint ---
        # If NDVI > 0.4 (dense vegetation) or NDVI < 0 (water), 
        # the model should NOT predict UHI. Penalize any positive prediction there.
        if ndvi is not None and self.lambda_veg > 0:
            veg_mask = (ndvi > self.ndvi_threshold).float()  # vegetation areas
            water_mask = (ndvi < self.water_ndvi_threshold).float()  # water areas
            invalid_mask = torch.clamp(veg_mask + water_mask, 0, 1)
            
            # Penalize: mean of (prediction probability * invalid_mask)
            # The model is punished for predicting UHI in vegetation/water
            veg_penalty = (pred_probs * invalid_mask).mean()
            total = total + self.lambda_veg * veg_penalty
        
        # --- Physics Penalty 2: Temperature Consistency ---
        # Cold areas (below scene mean LST) should not be UHI.
        # Penalize predictions where LST is significantly below mean.
        if lst is not None and self.lambda_temp > 0:
            # Compute per-image mean LST (only for valid pixels > 200K)
            valid_mask = (lst > 200.0).float()
            
            # Per-sample mean: sum(lst * valid) / sum(valid)
            lst_sum = (lst * valid_mask).sum(dim=(1, 2, 3), keepdim=True)
            valid_count = valid_mask.sum(dim=(1, 2, 3), keepdim=True).clamp(min=1)
            lst_mean = lst_sum / valid_count
            
            # Cold pixels: where LST < mean and pixel is valid
            cold_mask = ((lst < lst_mean) & (lst > 200.0)).float()
            
            # Penalize predictions in cold areas
            temp_penalty = (pred_probs * cold_mask).mean()
            total = total + self.lambda_temp * temp_penalty
        
        return total
