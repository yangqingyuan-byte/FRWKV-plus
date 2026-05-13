# FRWKV

<div align="center">
  <h2><b>FRWKV: Frequency-Domain Linear Attention for Long-Term Time Series Forecasting</b></h2>
</div>

[![](http://img.shields.io/badge/cs.LG-arXiv%3A2512.07539-B31B1B.svg)](https://www.arxiv.org/abs/2512.07539)

> If you find our work useful in your research, please consider giving a star ⭐ and citation 📚:

**BibTeX:**

```bibtex
@article{yang2025frwkv,
  title={FRWKV: Frequency-Domain Linear Attention for Long-Term Time Series Forecasting},
  author={Yang, Qingyuan and Chen, Dongyue and Teng, Da and Gan, Zehua and others},
  journal={arXiv preprint arXiv:2512.07539},
  year={2025}
}
```

**GB/T 7714:**

> Yang Q, Chen D, Teng D, et al. FRWKV: Frequency-Domain Linear Attention for Long-Term Time Series Forecasting\[J\]. arXiv preprint arXiv:2512.07539, 2025.

## Environment Setup

### 1. Create Conda Environment

```bash
conda create -n frwkv python=3.13.7
conda activate frwkv
```

### 2. Install PyTorch

Install PyTorch according to your CUDA version:

```bash
# For CUDA 12.9
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CPU only
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 3. Install Other Dependencies

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install pandas
pip install scikit-learn
pip install matplotlib
pip install timm
pip install seaborn
pip install reformer_pytorch
pip install fvcore
```

## Usage

### Running Experiments

```bash
# Activate environment
conda activate frwkv

# Run ETTh1 dataset experiment
bash scripts/FRWKV/ETTh1.sh

# Run ETTm1 dataset experiment
bash scripts/FRWKV/ETTm1.sh

# Run ETTm2 dataset experiment
bash scripts/FRWKV/ETTm2.sh

# Run ECL dataset experiment
bash scripts/FRWKV/ECL.sh

# Run Weather dataset experiment
bash scripts/FRWKV/Weather.sh
```

## Dataset

### Download Links

The datasets can be downloaded from the following links:

- **Google Drive**: https://drive.google.com/file/d/1Im29DpQakC-9M6Q4PV01YwhXkrnp_L0y/view?usp=sharing
- **Baidu Pan**: https://pan.baidu.com/s/1epgeVtY1xo1S8juxJpymOQ?pwd=g6fi

### Directory Structure

Please ensure that datasets are placed in the following directory structure:

```
dataset/
├── ETT-small/
│   ├── ETTh1.csv
│   ├── ETTh2.csv
│   ├── ETTm1.csv
│   └── ETTm2.csv
├── electricity/
│   └── electricity.csv
└── weather/
    └── weather.csv
```

## Experimental Results

Experimental results will be saved in the `logs/frwkv/` directory, with corresponding log files generated for each experiment.

## Notes

1. Please choose the appropriate PyTorch installation command based on your CUDA version
2. Ensure sufficient GPU memory to run experiments
3. Experiment logs will display detailed training process and results
