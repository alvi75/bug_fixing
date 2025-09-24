# Bug Fixing with RNNs

This project trains and evaluates RNN-based models (LSTM and GRU) for automated bug fixing.

## Quick Start

```bash
# 1. Create and activate environment
conda create -n bug_fixing python=3.10 -y
conda activate bug_fixing

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train a model
python train.py lstm

# 4. Evaluate the trained model
python eval.py checkpoints/lstm_small.pt lstm