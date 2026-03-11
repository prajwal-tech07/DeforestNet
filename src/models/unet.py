"""
DeforestNet - U-Net with ResNet-34 Encoder
Binary semantic segmentation model for deforestation detection.

Architecture:
  Encoder: ResNet-34 backbone adapted for 11-channel satellite input.
  Decoder: U-Net style with skip connections and upsampling blocks.

Input:  [B, 11, 256, 256]  (11 feature bands)
Output: [B, 2, 256, 256]   (2-class logits: non-deforestation, deforestation)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# RESNET-34 ENCODER BLOCKS
# ============================================================

class BasicBlock(nn.Module):
    """ResNet basic block with two 3x3 convolutions and a residual connection."""
    
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
    
    def forward(self, x):
        identity = x
        
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        return out


class ResNet34Encoder(nn.Module):
    """
    ResNet-34 encoder adapted for multi-channel satellite imagery.
    
    Produces feature maps at 5 resolution levels:
      - layer0: [B, 64, H/2, W/2]     (after initial conv + pool)
      - layer1: [B, 64, H/4, W/4]     (after pool + residual blocks)
      - layer2: [B, 128, H/8, W/8]
      - layer3: [B, 256, H/16, W/16]
      - layer4: [B, 512, H/32, W/32]  (bottleneck)
    """
    
    def __init__(self, in_channels=11):
        super().__init__()
        
        # Initial convolution: adapt to N input channels
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual layers (ResNet-34: [3, 4, 6, 3] blocks)
        self.layer1 = self._make_layer(64, 64, blocks=3, stride=1)
        self.layer2 = self._make_layer(64, 128, blocks=4, stride=2)
        self.layer3 = self._make_layer(128, 256, blocks=6, stride=2)
        self.layer4 = self._make_layer(256, 512, blocks=3, stride=2)
    
    def _make_layer(self, in_channels, out_channels, blocks, stride):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        
        layers = [BasicBlock(in_channels, out_channels, stride, downsample)]
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Returns feature maps at each resolution for skip connections
        x0 = self.relu(self.bn1(self.conv1(x)))  # [B, 64, H/2, W/2]
        x1 = self.layer1(self.maxpool(x0))        # [B, 64, H/4, W/4]
        x2 = self.layer2(x1)                      # [B, 128, H/8, W/8]
        x3 = self.layer3(x2)                      # [B, 256, H/16, W/16]
        x4 = self.layer4(x3)                      # [B, 512, H/32, W/32]
        
        return x0, x1, x2, x3, x4


# ============================================================
# U-NET DECODER BLOCKS
# ============================================================

class DecoderBlock(nn.Module):
    """
    U-Net decoder block: upsample → concat skip → double conv.
    
    Uses bilinear upsampling (no learnable upsampling artifacts)
    followed by two 3×3 convolutions with BatchNorm and ReLU.
    """
    
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear',
                                     align_corners=True)
        # After concat: in_channels + skip_channels
        self.conv1 = nn.Conv2d(in_channels + skip_channels, out_channels, 3,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x, skip):
        x = self.upsample(x)
        
        # Handle spatial size mismatches from odd dimensions
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear',
                              align_corners=True)
        
        x = torch.cat([x, skip], dim=1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


# ============================================================
# FULL U-NET MODEL
# ============================================================

class UNetResNet34(nn.Module):
    """
    U-Net with ResNet-34 encoder for binary deforestation segmentation.
    
    Encoder path (ResNet-34):
      Input [B, 11, 256, 256]
        → x0 [B, 64, 128, 128]
        → x1 [B, 64, 64, 64]
        → x2 [B, 128, 32, 32]
        → x3 [B, 256, 16, 16]
        → x4 [B, 512, 8, 8]    (bottleneck)
    
    Decoder path (U-Net):
      d4 = Up(x4) + x3 → [B, 256, 16, 16]
      d3 = Up(d4) + x2 → [B, 128, 32, 32]
      d2 = Up(d3) + x1 → [B, 64, 64, 64]
      d1 = Up(d2) + x0 → [B, 32, 128, 128]
      out = Up(d1)      → [B, num_classes, 256, 256]
    """
    
    def __init__(self, in_channels=11, num_classes=2, dropout_p=0.2):
        super().__init__()
        
        self.encoder = ResNet34Encoder(in_channels=in_channels)
        
        # Decoder blocks: (input_from_below, skip_channels, output)
        self.decoder4 = DecoderBlock(512, 256, 256)
        self.decoder3 = DecoderBlock(256, 128, 128)
        self.decoder2 = DecoderBlock(128, 64, 64)
        self.decoder1 = DecoderBlock(64, 64, 32)
        
        # Final upsampling to restore original resolution
        self.final_upsample = nn.Upsample(scale_factor=2, mode='bilinear',
                                           align_corners=True)
        
        self.dropout = nn.Dropout2d(p=dropout_p)
        
        # Classification head
        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)
        
        # Initialize weights (Kaiming Normal for Conv, standard for BN)
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Apply Kaiming Normal init (He et al.) for stable convergence."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                       nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """
        Args:
            x: [B, in_channels, H, W] input tensor
        Returns:
            [B, num_classes, H, W] logits (not softmax)
        """
        # Encoder
        x0, x1, x2, x3, x4 = self.encoder(x)
        
        # Decoder with skip connections
        d4 = self.decoder4(x4, x3)  # [B, 256, 16, 16]
        d3 = self.decoder3(d4, x2)  # [B, 128, 32, 32]
        d2 = self.decoder2(d3, x1)  # [B, 64, 64, 64]
        d1 = self.decoder1(d2, x0)  # [B, 32, 128, 128]
        
        # Final upsample to input resolution
        d0 = self.final_upsample(d1)  # [B, 32, 256, 256]
        d0 = self.dropout(d0)
        
        out = self.final_conv(d0)  # [B, num_classes, 256, 256]
        return out


def build_model(in_channels=11, num_classes=2, dropout_p=0.2):
    """
    Factory function to create the model.
    
    Args:
        in_channels: Number of input bands (default: 11)
        num_classes: Number of output classes (default: 2)
        dropout_p: Dropout probability before final conv
    
    Returns:
        UNetResNet34 model instance
    """
    return UNetResNet34(
        in_channels=in_channels,
        num_classes=num_classes,
        dropout_p=dropout_p,
    )
