#!/usr/bin/env python3
"""
LTX-2 Latent Upsampler Compilation for Neuron

Model: ~1GB, uses Conv3d for spatial upsampling
"""

import os
import sys
import json
import argparse
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("LTX2-Upsampler")

MODEL_DIR = "/home/ubuntu/models/ltx2/latent_upsampler"
OUTPUT_DIR = "/home/ubuntu/ltx2/neuron/compiled"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--temporal", type=int, default=16, help="Temporal frames in latent")
    parser.add_argument("--height", type=int, default=17, help="Height in latent space")
    parser.add_argument("--width", type=int, default=30, help="Width in latent space")
    parser.add_argument("--model_dir", type=str, default=MODEL_DIR)
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--load_weights", action="store_true")
    return parser.parse_args()


class ResBlock3D(nn.Module):
    """3D Residual Block with GroupNorm."""

    def __init__(self, channels: int = 1024):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, channels)
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(32, channels)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm1(x)
        x = F.silu(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = F.silu(x)
        x = self.conv2(x)
        return x + residual


class SpatialUpsampler(nn.Module):
    """Spatial 2x upsampler using pixel shuffle."""

    def __init__(self, in_channels: int = 1024, scale: int = 2):
        super().__init__()
        self.scale = scale
        # Conv2d for pixel shuffle (applied per frame)
        self.conv = nn.Conv2d(in_channels, in_channels * scale * scale, kernel_size=3, padding=1)
        # Blur kernel for anti-aliasing
        self.register_buffer("blur_kernel", self._make_blur_kernel())

    def _make_blur_kernel(self):
        """Create 5x5 blur kernel."""
        kernel = torch.tensor([1, 4, 6, 4, 1], dtype=torch.float32)
        kernel = kernel.outer(kernel)
        kernel = kernel / kernel.sum()
        return kernel.view(1, 1, 5, 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T, H, W]
        B, C, T, H, W = x.shape

        # Process each frame
        x = x.permute(0, 2, 1, 3, 4)  # [B, T, C, H, W]
        x = x.reshape(B * T, C, H, W)  # [B*T, C, H, W]

        # Conv + pixel shuffle
        x = self.conv(x)  # [B*T, C*4, H, W]
        x = F.pixel_shuffle(x, self.scale)  # [B*T, C, H*2, W*2]

        # Reshape back
        x = x.reshape(B, T, C, H * self.scale, W * self.scale)
        x = x.permute(0, 2, 1, 3, 4)  # [B, C, T, H*2, W*2]

        return x


class LTX2LatentUpsampler(nn.Module):
    """LTX-2 Latent Space Upsampler (spatial 2x)."""

    def __init__(
        self,
        in_channels: int = 128,
        mid_channels: int = 1024,
        num_blocks: int = 4,
        spatial_scale: int = 2,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.mid_channels = mid_channels

        # Initial projection
        self.initial_conv = nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1)
        self.initial_norm = nn.GroupNorm(32, mid_channels)

        # Pre-upsample residual blocks
        self.res_blocks = nn.ModuleList([
            ResBlock3D(mid_channels) for _ in range(num_blocks)
        ])

        # Spatial upsampler
        self.upsampler = SpatialUpsampler(mid_channels, spatial_scale)

        # Post-upsample residual blocks
        self.post_upsample_res_blocks = nn.ModuleList([
            ResBlock3D(mid_channels) for _ in range(num_blocks)
        ])

        # Final projection
        self.final_conv = nn.Conv3d(mid_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, T, H, W] latent tensor

        Returns:
            [B, C, T, H*2, W*2] upsampled latent
        """
        # Initial projection
        x = self.initial_conv(x)
        x = self.initial_norm(x)
        x = F.silu(x)

        # Pre-upsample blocks
        for block in self.res_blocks:
            x = block(x)

        # Spatial upsample 2x
        x = self.upsampler(x)

        # Post-upsample blocks
        for block in self.post_upsample_res_blocks:
            x = block(x)

        # Final projection
        x = self.final_conv(x)

        return x


def load_weights(model: LTX2LatentUpsampler, model_dir: str):
    """Load weights from safetensors."""
    from safetensors.torch import load_file

    weights_path = os.path.join(model_dir, "diffusion_pytorch_model.safetensors")
    logger.info(f"Loading weights from {weights_path}")

    state_dict = load_file(weights_path)
    model_state = model.state_dict()

    # Direct mapping (most keys match)
    loaded = 0
    for key in model_state.keys():
        if key in state_dict:
            if model_state[key].shape == state_dict[key].shape:
                model_state[key] = state_dict[key]
                loaded += 1
            else:
                logger.warning(f"Shape mismatch: {key}")

    model.load_state_dict(model_state)
    logger.info(f"Loaded {loaded}/{len(model_state)} weights")


def test_cpu_forward(args):
    """Test forward pass on CPU."""
    logger.info("Testing CPU forward...")

    model = LTX2LatentUpsampler()
    model = model.to(torch.bfloat16)
    model.eval()

    # Sample input: [B, C, T, H, W]
    x = torch.randn(
        args.batch_size, 128, args.temporal, args.height, args.width,
        dtype=torch.bfloat16
    )

    logger.info(f"Input shape: {x.shape}")

    with torch.no_grad():
        output = model(x)

    logger.info(f"Output shape: {output.shape}")
    logger.info("CPU forward OK!")

    return True


def compile_model(args):
    """Compile model with torch_neuronx."""
    import torch_neuronx

    logger.info("Creating model...")
    model = LTX2LatentUpsampler()
    model = model.to(torch.bfloat16)

    if args.load_weights:
        load_weights(model, args.model_dir)

    model.eval()

    # Sample input
    x = torch.randn(
        args.batch_size, 128, args.temporal, args.height, args.width,
        dtype=torch.bfloat16
    )

    logger.info(f"Tracing with input shape: {x.shape}")
    logger.info("Starting compilation (Conv3d may take time)...")

    try:
        traced = torch_neuronx.trace(
            model,
            x,
            compiler_args=[
                "--model-type=unet-inference",  # Use unet type for conv-heavy models
                "-O2",
            ],
        )

        # Save
        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(
            args.output_dir,
            f"latent_upsampler_bs{args.batch_size}_t{args.temporal}_h{args.height}_w{args.width}.pt"
        )
        torch.jit.save(traced, output_path)
        logger.info(f"Saved to {output_path}")

        return True

    except Exception as e:
        logger.error(f"Compilation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    args = parse_args()

    logger.info("=" * 60)
    logger.info("LTX-2 Latent Upsampler Compilation")
    logger.info("=" * 60)
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Input shape: [B, 128, {args.temporal}, {args.height}, {args.width}]")
    logger.info(f"Output shape: [B, 128, {args.temporal}, {args.height*2}, {args.width*2}]")

    # Test CPU first
    if not test_cpu_forward(args):
        return 1

    # Set Neuron flags
    os.environ["NEURON_CC_FLAGS"] = "--model-type=unet-inference -O2"

    # Compile
    if compile_model(args):
        logger.info("Compilation succeeded!")
        return 0
    else:
        logger.error("Compilation failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
