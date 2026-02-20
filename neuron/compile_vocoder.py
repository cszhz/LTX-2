#!/usr/bin/env python3
"""
LTX-2 Vocoder Compilation for Neuron

HiFi-GAN style vocoder for audio synthesis.
~107MB model, 240x temporal upsampling.
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
logger = logging.getLogger("LTX2-Vocoder")

MODEL_DIR = "/home/ubuntu/models/ltx2/vocoder"
OUTPUT_DIR = "/home/ubuntu/ltx2/neuron/compiled"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=100, help="Input sequence length")
    parser.add_argument("--model_dir", type=str, default=MODEL_DIR)
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--load_weights", action="store_true")
    return parser.parse_args()


class ResBlock1d(nn.Module):
    """HiFi-GAN style residual block with dilated convolutions."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilations: tuple = (1, 3, 5),
        leaky_relu_slope: float = 0.1,
    ):
        super().__init__()
        self.leaky_relu_slope = leaky_relu_slope

        # convs1: dilated convolutions
        self.convs1 = nn.ModuleList()
        for dilation in dilations:
            padding = (kernel_size * dilation - dilation) // 2
            self.convs1.append(
                nn.Conv1d(channels, channels, kernel_size, dilation=dilation, padding=padding)
            )

        # convs2: standard convolutions
        self.convs2 = nn.ModuleList()
        for _ in dilations:
            padding = (kernel_size - 1) // 2
            self.convs2.append(
                nn.Conv1d(channels, channels, kernel_size, padding=padding)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv1, conv2 in zip(self.convs1, self.convs2):
            residual = x
            x = F.leaky_relu(x, self.leaky_relu_slope)
            x = conv1(x)
            x = F.leaky_relu(x, self.leaky_relu_slope)
            x = conv2(x)
            x = x + residual
        return x


class LTX2Vocoder(nn.Module):
    """HiFi-GAN style vocoder matching weight structure.

    Structure:
    - conv_in: 128 -> 1024
    - 5 upsamplers (ConvTranspose1d) with factors [6, 5, 2, 2, 2]
    - 15 resnets (3 per level) with kernel sizes [3, 7, 11]
    - conv_out: 32 -> 2
    """

    def __init__(
        self,
        in_channels: int = 128,
        hidden_channels: int = 1024,
        out_channels: int = 2,
        upsample_factors: tuple = (6, 5, 2, 2, 2),
        upsample_kernel_sizes: tuple = (16, 15, 8, 4, 4),
        resnet_kernel_sizes: tuple = (3, 7, 11),
        resnet_dilations: tuple = ((1, 3, 5), (1, 3, 5), (1, 3, 5)),
        leaky_relu_slope: float = 0.1,
    ):
        super().__init__()
        self.leaky_relu_slope = leaky_relu_slope

        # conv_in
        self.conv_in = nn.Conv1d(in_channels, hidden_channels, kernel_size=7, padding=3)

        # Upsamplers (ConvTranspose1d)
        self.upsamplers = nn.ModuleList()
        in_ch = hidden_channels
        for i, (factor, kernel) in enumerate(zip(upsample_factors, upsample_kernel_sizes)):
            out_ch = in_ch // 2
            self.upsamplers.append(
                nn.ConvTranspose1d(
                    in_ch, out_ch,
                    kernel_size=kernel,
                    stride=factor,
                    padding=(kernel - factor) // 2,
                )
            )
            in_ch = out_ch

        # Resnets (3 per upsampler level, with different kernel sizes)
        self.resnets = nn.ModuleList()
        channels_per_level = [512, 256, 128, 64, 32]

        for level_ch in channels_per_level:
            for kernel in resnet_kernel_sizes:
                self.resnets.append(
                    ResBlock1d(
                        level_ch,
                        kernel_size=kernel,
                        dilations=resnet_dilations[0],
                        leaky_relu_slope=leaky_relu_slope,
                    )
                )

        # conv_out
        self.conv_out = nn.Conv1d(32, out_channels, kernel_size=7, padding=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 128, T] audio latent

        Returns:
            [B, 2, T*240] stereo audio waveform
        """
        x = self.conv_in(x)

        resnet_idx = 0
        for i, upsampler in enumerate(self.upsamplers):
            x = F.leaky_relu(x, self.leaky_relu_slope)
            x = upsampler(x)

            # 3 resnets per level
            for _ in range(3):
                x = self.resnets[resnet_idx](x)
                resnet_idx += 1

        x = F.leaky_relu(x, self.leaky_relu_slope)
        x = self.conv_out(x)
        x = torch.tanh(x)

        return x


def load_weights(model: nn.Module, model_dir: str):
    """Load weights from safetensors."""
    from safetensors.torch import load_file

    weights_path = os.path.join(model_dir, "diffusion_pytorch_model.safetensors")
    logger.info(f"Loading weights from {weights_path}")

    state_dict = load_file(weights_path)
    model_state = model.state_dict()

    loaded = 0
    for key in model_state.keys():
        if key in state_dict:
            if model_state[key].shape == state_dict[key].shape:
                model_state[key] = state_dict[key]
                loaded += 1
            else:
                logger.warning(f"Shape mismatch: {key} model={model_state[key].shape} weight={state_dict[key].shape}")
        else:
            logger.warning(f"Missing key: {key}")

    model.load_state_dict(model_state)
    logger.info(f"Loaded {loaded}/{len(model_state)} weights")
    return loaded


def test_cpu_forward(args):
    """Test forward pass on CPU."""
    logger.info("Testing CPU forward...")

    model = LTX2Vocoder()
    model = model.to(torch.bfloat16)
    model.eval()

    # Input: [B, 128, T]
    x = torch.randn(args.batch_size, 128, args.seq_len, dtype=torch.bfloat16)
    logger.info(f"Input shape: {x.shape}")

    with torch.no_grad():
        out = model(x)

    logger.info(f"Output shape: {out.shape}")
    expected_len = args.seq_len * 240  # 6*5*2*2*2 = 240
    logger.info(f"Expected output length: {expected_len}")

    logger.info("CPU forward OK!")
    return True


def compile_model(args):
    """Compile vocoder."""
    import torch_neuronx

    logger.info("Compiling vocoder...")
    model = LTX2Vocoder()
    model = model.to(torch.bfloat16)

    if args.load_weights:
        load_weights(model, args.model_dir)

    model.eval()

    x = torch.randn(args.batch_size, 128, args.seq_len, dtype=torch.bfloat16)
    logger.info(f"Input shape: {x.shape}")

    try:
        traced = torch_neuronx.trace(
            model, x,
            compiler_args=["--model-type=unet-inference", "-O2"],
        )

        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(
            args.output_dir,
            f"vocoder_bs{args.batch_size}_seq{args.seq_len}.pt"
        )
        torch.jit.save(traced, output_path)
        logger.info(f"Saved vocoder to {output_path}")
        return True
    except Exception as e:
        logger.error(f"Compilation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    args = parse_args()

    logger.info("=" * 60)
    logger.info("LTX-2 Vocoder Compilation")
    logger.info("=" * 60)
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Input seq len: {args.seq_len}")
    logger.info(f"Output samples: {args.seq_len * 240} (240x upsample)")

    # Test CPU first
    if not test_cpu_forward(args):
        return 1

    os.environ["NEURON_CC_FLAGS"] = "--model-type=unet-inference -O2"

    if compile_model(args):
        logger.info("Compilation succeeded!")
        return 0
    else:
        logger.error("Compilation failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
