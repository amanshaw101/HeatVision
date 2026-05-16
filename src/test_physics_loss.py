"""
Quick sanity check for PhysicsInformedLoss.
Verifies that the physics penalties work correctly.

Run with: python src/test_physics_loss.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import numpy as np
from loss import PhysicsInformedLoss, CombinedLoss

def test_basic_forward():
    """Test that PhysicsInformedLoss runs without error."""
    loss_fn = PhysicsInformedLoss(lambda_veg=0.1, lambda_temp=0.1)
    
    B, C, H, W = 1, 1, 64, 64
    pred = torch.randn(B, C, H, W)
    target = torch.randint(0, 2, (B, C, H, W)).float()
    ndvi = torch.rand(B, C, H, W)        # 0-1 range
    lst = torch.ones(B, C, H, W) * 300   # 300K
    
    loss = loss_fn(pred, target, ndvi=ndvi, lst=lst)
    assert loss.item() > 0, "Loss should be positive"
    assert not torch.isnan(loss), "Loss should not be NaN"
    print("✅ test_basic_forward PASSED")

def test_vegetation_penalty():
    """Predictions in high-NDVI areas should be penalized more."""
    loss_fn_with = PhysicsInformedLoss(lambda_veg=1.0, lambda_temp=0.0)
    loss_fn_without = PhysicsInformedLoss(lambda_veg=0.0, lambda_temp=0.0)
    
    B, C, H, W = 1, 1, 64, 64
    
    # Create predictions: all positive (high probability)
    pred = torch.ones(B, C, H, W) * 3.0  # logits → sigmoid ≈ 0.95
    target = torch.ones(B, C, H, W)      # all positive
    
    # NDVI > 0.4 everywhere (dense vegetation)
    ndvi = torch.ones(B, C, H, W) * 0.6
    lst = torch.ones(B, C, H, W) * 310
    
    loss_with_penalty = loss_fn_with(pred, target, ndvi=ndvi, lst=lst)
    loss_without_penalty = loss_fn_without(pred, target, ndvi=ndvi, lst=lst)
    
    assert loss_with_penalty > loss_without_penalty, \
        f"Vegetation penalty should increase loss: {loss_with_penalty:.4f} vs {loss_without_penalty:.4f}"
    print("✅ test_vegetation_penalty PASSED")

def test_temperature_penalty():
    """Predictions in cold areas should be penalized."""
    loss_fn = PhysicsInformedLoss(lambda_veg=0.0, lambda_temp=1.0)
    
    B, C, H, W = 1, 1, 64, 64
    pred = torch.ones(B, C, H, W) * 3.0  # high probability
    target = torch.ones(B, C, H, W)
    ndvi = torch.ones(B, C, H, W) * 0.2  # no vegetation penalty
    
    # Cold: LST = 280K (below mean of 300K)
    lst_cold = torch.ones(B, C, H, W) * 280
    lst_cold[:, :, 0:32, :] = 320  # half is hot so mean > 280
    
    # Hot: LST = 320K everywhere (all above mean)
    lst_hot = torch.ones(B, C, H, W) * 320
    
    loss_cold = loss_fn(pred, target, ndvi=ndvi, lst=lst_cold)
    loss_hot = loss_fn(pred, target, ndvi=ndvi, lst=lst_hot)
    
    assert loss_cold > loss_hot, \
        f"Cold-area predictions should have higher loss: cold={loss_cold:.4f} vs hot={loss_hot:.4f}"
    print("✅ test_temperature_penalty PASSED")

def test_reduces_to_combined_when_lambdas_zero():
    """With λ_veg=0 and λ_temp=0, should behave like CombinedLoss."""
    physics_loss = PhysicsInformedLoss(lambda_veg=0.0, lambda_temp=0.0)
    combined_loss = CombinedLoss(bce_weight=0.5, dice_weight=0.5)
    
    torch.manual_seed(42)
    B, C, H, W = 1, 1, 64, 64
    pred = torch.randn(B, C, H, W)
    target = torch.randint(0, 2, (B, C, H, W)).float()
    ndvi = torch.rand(B, C, H, W)
    lst = torch.ones(B, C, H, W) * 300
    
    l1 = physics_loss(pred, target, ndvi=ndvi, lst=lst)
    l2 = combined_loss(pred, target)
    
    assert abs(l1.item() - l2.item()) < 1e-5, \
        f"Should match CombinedLoss when lambdas=0: {l1:.6f} vs {l2:.6f}"
    print("✅ test_reduces_to_combined PASSED")

def test_backward_pass():
    """Verify gradients flow through physics penalties."""
    loss_fn = PhysicsInformedLoss(lambda_veg=0.1, lambda_temp=0.1)
    
    B, C, H, W = 1, 1, 64, 64
    pred = torch.randn(B, C, H, W, requires_grad=True)
    target = torch.randint(0, 2, (B, C, H, W)).float()
    ndvi = torch.rand(B, C, H, W)
    lst = torch.ones(B, C, H, W) * 300
    
    loss = loss_fn(pred, target, ndvi=ndvi, lst=lst)
    loss.backward()
    
    assert pred.grad is not None, "Gradients should exist"
    assert not torch.any(torch.isnan(pred.grad)), "Gradients should not be NaN"
    print("✅ test_backward_pass PASSED")


if __name__ == '__main__':
    print("Running PhysicsInformedLoss tests...\n")
    test_basic_forward()
    test_vegetation_penalty()
    test_temperature_penalty()
    test_reduces_to_combined_when_lambdas_zero()
    test_backward_pass()
    print("\n🎉 All tests passed!")
