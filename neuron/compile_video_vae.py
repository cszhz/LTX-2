#!/usr/bin/env python3
"""
LTX-2 Video VAE Decoder Compilation for Neuron

3D VAE decoder for video decoding with Conv3d.
~1.2GB decoder, spatial compression 32x, temporal compression 8x.
"""

import os
import sys
import argparse
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("LTX2-VideoVAE")

MODEL_DIR = "/home/ubuntu/models/ltx2/vae"
OUTPUT_DIR = "/home/ubuntu/ltx2/neuron/compiled"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--latent_t", type=int, default=2, help="Latent temporal dim")
    parser.add_argument("--latent_h", type=int, default=8, help="Latent height")
    parser.add_argument("--latent_w", type=int, default=8, help="Latent width")
    parser.add_argument("--model_dir", type=str, default=MODEL_DIR)
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--load_weights", action="store_true")
    return parser.parse_args()


class CausalConv3d(nn.Module):
    """3D convolution wrapper matching weight structure."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ):
        super().__init__()
        self.padding = padding
        self.conv = nn.Conv3d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ResBlock3D(nn.Module):
    """3D Residual block matching decoder.mid_block/up_blocks structure."""

    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.conv1 = CausalConv3d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = CausalConv3d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.group_norm(x, 1, eps=self.eps)
        x = F.silu(x)
        x = self.conv1(x)
        x = F.group_norm(x, 1, eps=self.eps)
        x = F.silu(x)
        x = self.conv2(x)
        return x + residual


class PixelShuffle3D(nn.Module):
    """3D pixel shuffle for upsampling."""

    def __init__(self, factor: int = 2):
        super().__init__()
        self.factor = factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C*f^3, T, H, W] -> [B, C, T*f, H*f, W*f]
        B, C, T, H, W = x.shape
        f = self.factor
        out_c = C // (f ** 3)

        x = x.view(B, out_c, f, f, f, T, H, W)
        x = x.permute(0, 1, 5, 2, 6, 3, 7, 4)  # [B, out_c, T, f, H, f, W, f]
        x = x.reshape(B, out_c, T * f, H * f, W * f)
        return x


class Upsampler3D(nn.Module):
    """Upsampler block: conv -> pixel shuffle."""

    def __init__(self, in_channels: int, out_channels: int, factor: int = 2):
        super().__init__()
        shuffle_channels = out_channels * (factor ** 3)
        self.conv = CausalConv3d(in_channels, shuffle_channels, kernel_size=3, padding=1)
        self.shuffle = PixelShuffle3D(factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.shuffle(x)
        return x


class VideoDecoder(nn.Module):
    """Video VAE Decoder matching exact weight structure.

    Structure from weights:
    - conv_in: 128 -> 1024
    - mid_block.resnets: 5 blocks at 1024
    - up_blocks.0: upsample 1024->512, then 5 resnets at 512
    - up_blocks.1: upsample 512->256, then 5 resnets at 256
    - up_blocks.2: upsample 256->128, then 5 resnets at 128
    - conv_out: 128 -> 48
    """

    def __init__(self, patch_size: int = 4):
        super().__init__()
        self.patch_size = patch_size

        # conv_in: 128 -> 1024
        self.conv_in = CausalConv3d(128, 1024, kernel_size=3, padding=1)

        # mid_block: 5 resnets at 1024
        self.mid_block = nn.ModuleDict({
            'resnets': nn.ModuleList([ResBlock3D(1024) for _ in range(5)])
        })

        # up_blocks
        self.up_blocks = nn.ModuleList()

        # up_blocks.0: 1024 -> 512 (upsample first, then resnets)
        block0 = nn.ModuleDict({
            'upsamplers': nn.ModuleList([Upsampler3D(1024, 512, factor=2)]),
            'resnets': nn.ModuleList([ResBlock3D(512) for _ in range(5)])
        })
        self.up_blocks.append(block0)

        # up_blocks.1: 512 -> 256
        block1 = nn.ModuleDict({
            'upsamplers': nn.ModuleList([Upsampler3D(512, 256, factor=2)]),
            'resnets': nn.ModuleList([ResBlock3D(256) for _ in range(5)])
        })
        self.up_blocks.append(block1)

        # up_blocks.2: 256 -> 128
        block2 = nn.ModuleDict({
            'upsamplers': nn.ModuleList([Upsampler3D(256, 128, factor=2)]),
            'resnets': nn.ModuleList([ResBlock3D(128) for _ in range(5)])
        })
        self.up_blocks.append(block2)

        # conv_out: 128 -> 48 (3 * 4 * 4)
        self.conv_out = CausalConv3d(128, 48, kernel_size=3, padding=1)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """Convert patches back to video: [B, 48, T, H, W] -> [B, 3, T, H*4, W*4]"""
        B, C, T, H, W = x.shape
        p = self.patch_size
        c = C // (p * p)  # 3
        x = x.view(B, c, p, p, T, H, W)
        x = x.permute(0, 1, 4, 5, 2, 6, 3)  # [B, 3, T, H, p, W, p]
        x = x.reshape(B, c, T, H * p, W * p)
        return x

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: [B, 128, T, H, W] latent

        Returns:
            [B, 3, T*8, H*32, W*32] video
        """
        x = self.conv_in(z)

        # Mid block
        for resnet in self.mid_block['resnets']:
            x = resnet(x)

        # Up blocks (each does: upsample 2x -> resnets)
        for block in self.up_blocks:
            x = block['upsamplers'][0](x)
            for resnet in block['resnets']:
                x = resnet(x)

        # Output
        x = F.group_norm(x, 1)
        x = F.silu(x)
        x = self.conv_out(x)

        # Unpatchify: [B, 48, T, H, W] -> [B, 3, T, H*4, W*4]
        x = self.unpatchify(x)

        return x


def load_weights(model: nn.Module, model_dir: str, prefix: str = ""):
    """Load weights from safetensors."""
    from safetensors.torch import load_file

    weights_path = os.path.join(model_dir, "diffusion_pytorch_model.safetensors")
    logger.info(f"Loading weights from {weights_path}")

    state_dict = load_file(weights_path)
    model_state = model.state_dict()

    loaded = 0
    for key in model_state.keys():
        weight_key = prefix + key if prefix else key

        if weight_key in state_dict:
            if model_state[key].shape == state_dict[weight_key].shape:
                model_state[key] = state_dict[weight_key]
                loaded += 1
            else:
                logger.warning(f"Shape mismatch: {key} model={model_state[key].shape} weight={state_dict[weight_key].shape}")
        else:
            logger.debug(f"Missing key: {weight_key}")

    model.load_state_dict(model_state)
    logger.info(f"Loaded {loaded}/{len(model_state)} weights")
    return loaded


def test_cpu_forward(args):
    """Test forward pass on CPU."""
    logger.info("Testing CPU forward...")

    decoder = VideoDecoder()
    decoder = decoder.to(torch.bfloat16)
    decoder.eval()

    # Latent: [B, 128, T, H, W]
    z = torch.randn(args.batch_size, 128, args.latent_t, args.latent_h, args.latent_w, dtype=torch.bfloat16)
    logger.info(f"Decoder input shape: {z.shape}")

    with torch.no_grad():
        out = decoder(z)

    logger.info(f"Decoder output shape: {out.shape}")
    # Expected: [B, 3, T*8, H*32, W*32]
    expected_t = args.latent_t * 8
    expected_h = args.latent_h * 32
    expected_w = args.latent_w * 32
    logger.info(f"Expected output: [1, 3, {expected_t}, {expected_h}, {expected_w}]")

    logger.info("CPU forward OK!")
    return True


def compile_decoder(args):
    """Compile decoder."""
    import torch_neuronx

    logger.info("Compiling decoder...")
    model = VideoDecoder()
    model = model.to(torch.bfloat16)

    if args.load_weights:
        load_weights(model, args.model_dir, prefix="decoder.")

    model.eval()

    z = torch.randn(args.batch_size, 128, args.latent_t, args.latent_h, args.latent_w, dtype=torch.bfloat16)
    logger.info(f"Latent input shape: {z.shape}")

    try:
        traced = torch_neuronx.trace(
            model, z,
            compiler_args=["--model-type=unet-inference", "-O2"],
        )

        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(
            args.output_dir,
            f"video_vae_decoder_bs{args.batch_size}_t{args.latent_t}_h{args.latent_h}_w{args.latent_w}.pt"
        )
        torch.jit.save(traced, output_path)
        logger.info(f"Saved decoder to {output_path}")
        return True
    except Exception as e:
        logger.error(f"Decoder compilation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    args = parse_args()

    logger.info("=" * 60)
    logger.info("LTX-2 Video VAE Decoder Compilation")
    logger.info("=" * 60)
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Latent shape: [B, 128, {args.latent_t}, {args.latent_h}, {args.latent_w}]")

    out_t = args.latent_t * 8
    out_h = args.latent_h * 32
    out_w = args.latent_w * 32
    logger.info(f"Output video: [B, 3, {out_t}, {out_h}, {out_w}]")

    # Test CPU first
    if not test_cpu_forward(args):
        return 1

    os.environ["NEURON_CC_FLAGS"] = "--model-type=unet-inference -O2"

    if compile_decoder(args):
        logger.info("Compilation succeeded!")
        return 0
    else:
        logger.error("Compilation failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
