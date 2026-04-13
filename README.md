# Quantum Machine Learning for Weather Forecasting

## Overview
This project aims to develop machine learning models for spatiotemporal weather prediction, including a classical ConvLSTM and a hybrid quantum-classical model (QuanvLSTM). The models are designed to jointly predict multi-dimensional atmospheric variables such as isoprene.

<p align="center">
  <img src="assets/banner.png" width="500"/>
</p>

## Environment Setup
The project is based on Python 3.11, PyTorch, and TorchQuantum. It is recommended to use Conda to create an isolated environment.
```bash
conda create -n qweather python=3.11
conda activate qweather
```

### Install dependencies
```bash
pip install -r requirements.txt
```

### Install TorchQuantum separately
```bash
git clone https://github.com/mit-han-lab/torchquantum.git
cd torchquantum
pip install --editable .
cd ..
```

## Project Structure
```bash
.
├── README.md
├── checkpoints     # Model checkpoints
│   ├── convlstm
│   └── quanvlstm
├── configs         # Configuration files
├── data_preprocess # Data preprocessing
├── datasets        # Dataset handling
│   ├── data
│   ├── dataset.py
│   ├── scaler
│   └── split       # Dataset splits
│       ├── large
│       └── small
├── logs            # Training logs
│   ├── convlstm
│   └── quanvlstm
├── models          # Model architectures
│   ├── convlstm
│   └── quanvlstm
├── pretrained      # Pretrained models
├── requirements.txt
├── train_convlstm.py
├── train_quanvlstm.py
├── evaluate_convlstm.py
├── evaluate_quanvlstm.py
├── utils           # Utilities and training code
└── visualizations  # Visualization tools
```

## Data Preparation
Raw data is stored in `datasets/data`, divided by resolution into `data_large.nc` and `data_small.nc`. By default, `data_large.nc` is used.

Dataset splits (train/validation/test) are located in `datasets/split`.

## Training
### Classical Model — ConvLSTM
Configure configs/convlstm.yaml, then run:

```bash
python train_convlstm.py
```

### Hybrid Model — QuanvLSTM
Configure configs/quanvlstm.yaml, then run:

```bash
python train_quanvlstm.py
```

Pretrained models are provided in the `pretrained/` directory (`convlstm.pth`, `quanvlstm.pth`).

## Evaluation
Evaluation is performed on the test set using mean MSE loss as the metric.

Update the `model_path` in:

`evaluate_convlstm.py`
`evaluate_quanvlstm.py`

By default, the pretrained models are evaluated:

`pretrained/convlstm.pth`
`pretrained/quanvlstm.pth`

Run:

```bash
python evaluate_convlstm.py
python evaluate_quanvlstm.py
```
