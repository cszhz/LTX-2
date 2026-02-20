#!/bin/bash
# LTX-2 Transformer Compilation Script

set -e

# Activate virtual environment
source ~/nki_venv/bin/activate

# Set environment variables
export NEURON_CC_FLAGS="--model-type=transformer -O2"
export MALLOC_ARENA_MAX=64

# Change to script directory
cd ~/ltx2/neuron

echo "============================================"
echo "LTX-2 Transformer Neuron Compilation"
echo "============================================"
echo "Python: $(which python3)"
echo "PyTorch: $(python3 -c 'import torch; print(torch.__version__)')"
echo "torch-neuronx: $(python3 -c 'import torch_neuronx; print(torch_neuronx.__version__)')"
echo "============================================"

# Run compilation
python3 compile_transformer.py \
    --tp_degree 4 \
    --batch_size 1 \
    --max_video_tokens 4096 \
    --model_dir /home/ubuntu/models/ltx2/transformer \
    --output_dir /home/ubuntu/ltx2/neuron/compiled \
    "$@"

echo "============================================"
echo "Compilation finished!"
echo "============================================"
