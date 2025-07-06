#!/bin/bash

# This script runs the PyTorch DDP training for the SeqVAE model.
# It utilizes torchrun for distributed training on multiple GPUs.

# Exit immediately if a command exits with a non-zero status.
set -e

# Execute the training command using torchrun
torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=6 \
  --master_addr=127.0.0.1 \
  --master_port=29500 \
  model/graph_model.py

echo "Training command initiated." 