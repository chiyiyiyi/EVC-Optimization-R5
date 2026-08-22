#!/usr/bin/env bash
# Round-4 environment setup for EVC-evsod-main on the school server.
# Usage in every new terminal/room:
#   conda activate dsy_py3.8_cuda10.2
#   source /home/biiteam/Storage-4T/LHM/EVC-evsod-main/env_round4.sh

export CUDA_HOME=/usr/local/cuda-11.6
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

export PROJECT_DIR=/home/biiteam/Storage-4T/LHM/EVC-evsod-main
export PYTHONPATH="$PROJECT_DIR/lib/hais_ops/build/lib.linux-x86_64-cpython-38:$PROJECT_DIR/lib/hais_ops:$PYTHONPATH"

export DATA_ROOT="/home/biiteam/Storage-4T/LHM/EV-UAV-main/dataset"

echo "round4 env ready"
python -c "import torch, HAIS_OP; print('HAIS_OP ok', torch.__version__)"
