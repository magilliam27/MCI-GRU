# MCI-GRU: Stock Prediction Model Based on Multi-Head Cross-Attention and Improved GRU (Neurocomputing).

[![arXiv](http://img.shields.io/badge/cs.LG-arXiv%3A2410.20679-B31B1B.svg)](https://arxiv.org/abs/2410.20679)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 📖 Overview

MCI-GRU is a stock forecasting architecture that replaces the GRU reset gate with an attention mechanism for dynamic temporal feature selection, incorporates a Graph Attention Network (GAT) to model inter-stock dependencies, and employs multi-head cross-attention to infer latent market states. By fusing temporal, cross-sectional, and latent representations, MCI-GRU achieves state-of-the-art performance on CSI 300/500, S&P 500, and NASDAQ 100 benchmarks, delivering higher ARR, Sharpe, and Calmar ratios, and has been deployed in live fund management systems.

<img width="1782" height="774" alt="image" src="https://github.com/user-attachments/assets/1acf271a-5ef6-4f40-8e1c-aad188caead2" />

## 🏆 Experiment Results

### Performance Comparison
<img width="1641" height="744" alt="Performance Results" src="https://github.com/user-attachments/assets/018689e3-1d6f-4c62-a7a7-21d8a5ed83e8" />

<img width="1650" height="762" alt="Ablation Study" src="https://github.com/user-attachments/assets/2e046607-f00f-4e10-9d28-ad5e204d4d94" />

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for training)
- 8GB+ RAM

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/WinstonLiyt/MCI-GRU.git
   cd MCI-GRU
   ```

2. **Create and activate virtual environment**
   ```bash
   # Using conda
   conda create -n mcigru python=3.8
   conda activate mcigru
   
   # Or using venv
   python -m venv mcigru_env
   source mcigru_env/bin/activate  # Linux/Mac
   # or
   mcigru_env\Scripts\activate     # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Data Preparation

1. **Prepare your dataset** in CSV format with the following columns:
   - `kdcode`: Stock code identifier
   - `dt`: Date (YYYY-MM-DD format)
   - `close`, `open`, `high`, `low`: OHLC prices
   - `turnover`: Trading turnover
   - `volume`: Trading volume

2. **Update the data path** in the script:
   ```python
   filename = '/path/to/your/dataset.csv'
   ```

### Usage

#### Training and Prediction

Run the main training script for different market indices:

```bash
# CSI 300 Index
python code/csi300.py
```

#### Ablation Studies

Run ablation studies to analyze component contributions:

```bash
# Ablation studies for CSI 300 Index
python ablation/csi300.py
```

## 📁 Project Structure

```
MCI-GRU/
├── code/                   # Main implementation
│   ├── csi300.py           # CSI 300 index model
│   ├── csi500.py           # CSI 500 index model
│   ├── sp500.py            # S&P 500 index model
│   └── nasdaq100.py        # NASDAQ 100 index model
├── ablation/               # Ablation study implementations
│   ├── csi300.py           # CSI 300 ablation studies
│   ├── csi500.py           # CSI 500 ablation studies
│   ├── sp500.py            # S&P 500 ablation studies
│   └── nasdaq100.py        # NASDAQ 100 ablation studies
├── requirements.txt        # Python dependencies
├── README.md               # This file
└── LICENSE                 # MIT License
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Citation

If you find this repository helpful, please cite our paper:

```bibtex
@article{zhu2025mci,
  title={MCI-GRU: Stock prediction model based on multi-head cross-attention and improved GRU},
  author={Zhu, Peng and Li, Yuante and Hu, Yifan and Xiang, Sheng and Liu, Qinyuan and Cheng, Dawei and Liang, Yuqi},
  journal={Neurocomputing},
  volume={638},
  pages={130168},
  year={2025},
  publisher={Elsevier}
}
```

---

**Disclaimer**: This code is for research purposes only. Past performance does not guarantee future results. Please conduct your own research before making investment decisions.
