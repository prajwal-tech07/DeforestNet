"""
Test script for Step 5: U-Net + ResNet-34 Model Architecture.
Validates model construction, forward pass, output shapes, gradient flow,
and integration with the real DataLoader.
"""

import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn
from src.models.unet import UNetResNet34, BasicBlock, ResNet34Encoder, DecoderBlock, build_model
from src.data.dataset import get_dataloaders
from configs.config import (
    IN_CHANNELS, NUM_CLASSES, PATCH_SIZE, BATCH_SIZE,
    PREPROCESSED_DIR, DROPOUT_P
)


def test_encoder_shapes():
    """Test that the ResNet-34 encoder produces correct feature map shapes."""
    print("TEST 1: Encoder feature map shapes...")
    
    encoder = ResNet34Encoder(in_channels=IN_CHANNELS)
    x = torch.randn(2, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE)
    
    x0, x1, x2, x3, x4 = encoder(x)
    
    expected = {
        'x0': (2, 64, 128, 128),
        'x1': (2, 64, 64, 64),
        'x2': (2, 128, 32, 32),
        'x3': (2, 256, 16, 16),
        'x4': (2, 512, 8, 8),
    }
    
    actual = {
        'x0': tuple(x0.shape),
        'x1': tuple(x1.shape),
        'x2': tuple(x2.shape),
        'x3': tuple(x3.shape),
        'x4': tuple(x4.shape),
    }
    
    for name in expected:
        assert actual[name] == expected[name], \
            f"{name}: {actual[name]} != {expected[name]}"
        print(f"  {name}: {actual[name]} ✓")
    
    print("  PASSED")


def test_decoder_block():
    """Test a single decoder block."""
    print("\nTEST 2: Decoder block...")
    
    block = DecoderBlock(in_channels=512, skip_channels=256, out_channels=256)
    x = torch.randn(2, 512, 8, 8)
    skip = torch.randn(2, 256, 16, 16)
    
    out = block(x, skip)
    assert out.shape == (2, 256, 16, 16), f"Got {out.shape}"
    print(f"  Output shape: {out.shape} ✓")
    print("  PASSED")


def test_full_model_forward():
    """Test complete forward pass with correct input/output shapes."""
    print("\nTEST 3: Full model forward pass...")
    
    model = build_model(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES, 
                        dropout_p=DROPOUT_P)
    model.eval()
    
    x = torch.randn(2, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE)
    
    with torch.no_grad():
        out = model(x)
    
    expected_shape = (2, NUM_CLASSES, PATCH_SIZE, PATCH_SIZE)
    assert out.shape == expected_shape, f"Output {out.shape} != {expected_shape}"
    assert torch.isfinite(out).all(), "Output contains NaN/Inf"
    
    print(f"  Input:  {x.shape}")
    print(f"  Output: {out.shape} ✓")
    print(f"  Output range: [{out.min():.4f}, {out.max():.4f}]")
    print("  PASSED")


def test_gradient_flow():
    """Test that gradients flow through the entire model."""
    print("\nTEST 4: Gradient flow...")
    
    model = build_model(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES)
    model.train()
    
    x = torch.randn(2, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE)
    target = torch.randint(0, NUM_CLASSES, (2, PATCH_SIZE, PATCH_SIZE))
    
    out = model(x)
    loss = nn.CrossEntropyLoss()(out, target)
    loss.backward()
    
    # Check that all parameters have gradients
    total_params = 0
    params_with_grad = 0
    for name, param in model.named_parameters():
        total_params += 1
        if param.grad is not None and param.grad.abs().sum() > 0:
            params_with_grad += 1
    
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Parameters with gradient: {params_with_grad}/{total_params}")
    assert params_with_grad == total_params, \
        f"Only {params_with_grad}/{total_params} parameters have gradients"
    print("  PASSED")


def test_parameter_count():
    """Test model parameter count is in expected range for ResNet-34 U-Net."""
    print("\nTEST 5: Parameter count...")
    
    model = build_model(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES)
    
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"  Total parameters:     {total:,}")
    print(f"  Trainable parameters: {trainable:,}")
    print(f"  Model size: ~{total * 4 / 1024 / 1024:.1f} MB (float32)")
    
    # ResNet-34 has ~21M params; with decoder + 11-ch input it should be ~24-28M
    assert 20_000_000 < total < 35_000_000, \
        f"Parameter count {total:,} outside expected range"
    assert total == trainable, "Some parameters are frozen unexpectedly"
    print("  PASSED")


def test_with_real_dataloader():
    """Test model with real data from the DataLoader."""
    print("\nTEST 6: Forward pass with real DataLoader batch...")
    
    model = build_model(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES)
    model.eval()
    
    loaders = get_dataloaders(PREPROCESSED_DIR, batch_size=4)
    images, masks = next(iter(loaders['val']))
    
    print(f"  Input batch:  {images.shape}, dtype={images.dtype}")
    print(f"  Target batch: {masks.shape}, dtype={masks.dtype}")
    
    with torch.no_grad():
        logits = model(images)
    
    assert logits.shape == (4, NUM_CLASSES, PATCH_SIZE, PATCH_SIZE), \
        f"Output shape: {logits.shape}"
    assert torch.isfinite(logits).all(), "Output contains NaN/Inf on real data"
    
    # Check predictions make sense
    preds = torch.argmax(logits, dim=1)
    assert preds.shape == masks.shape, f"Pred shape {preds.shape} != mask shape {masks.shape}"
    
    print(f"  Logits:      {logits.shape}, range=[{logits.min():.4f}, {logits.max():.4f}]")
    print(f"  Predictions: {preds.shape}, unique={torch.unique(preds).tolist()}")
    print("  PASSED")


def test_training_step():
    """Simulate one complete training step with loss computation."""
    print("\nTEST 7: Simulated training step...")
    
    model = build_model(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES)
    model.train()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    loaders = get_dataloaders(PREPROCESSED_DIR, batch_size=4)
    images, masks = next(iter(loaders['train']))
    
    # Forward
    t0 = time.time()
    logits = model(images)
    loss = criterion(logits, masks)
    
    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    elapsed = time.time() - t0
    
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Step time: {elapsed:.2f}s (CPU)")
    assert loss.item() > 0, "Loss should be positive"
    assert torch.isfinite(torch.tensor(loss.item())), "Loss is NaN/Inf"
    print("  PASSED")


def test_different_batch_sizes():
    """Test model works with various batch sizes."""
    print("\nTEST 8: Different batch sizes...")
    
    model = build_model(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES)
    model.eval()
    
    for bs in [1, 2, 8]:
        x = torch.randn(bs, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (bs, NUM_CLASSES, PATCH_SIZE, PATCH_SIZE)
        print(f"  Batch size {bs}: {out.shape} ✓")
    
    print("  PASSED")


if __name__ == "__main__":
    print("=" * 55)
    print("  DeforestNet — Step 5: Model Architecture Tests")
    print("=" * 55)
    
    test_encoder_shapes()
    test_decoder_block()
    test_full_model_forward()
    test_gradient_flow()
    test_parameter_count()
    test_with_real_dataloader()
    test_training_step()
    test_different_batch_sizes()
    
    print("\n" + "=" * 55)
    print("  ALL 8 TESTS PASSED!")
    print("=" * 55)
