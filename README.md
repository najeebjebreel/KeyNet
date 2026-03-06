# KeyNet: An Asymmetric Key-Style Framework for Watermarking Deep Learning Models

[![Paper](https://img.shields.io/badge/Paper-Applied%20Sciences%202021-blue)](https://www.mdpi.com/2076-3417/11/3/999)
[![Python](https://img.shields.io/badge/Python-3.6%2B-green)](https://www.anaconda.com/download)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.6%2B-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

> Official PyTorch implementation of the paper **"KeyNet: An Asymmetric Key-Style Framework for Watermarking Deep Learning Models"**, published in *Applied Sciences*, vol. 11, no. 3, p. 999, MDPI, 2021.

---

## Overview

**KeyNet** is a deep learning model watermarking framework inspired by asymmetric cryptography. It enables model owners to embed verifiable watermarks into neural networks — functioning as a private key — that can be publicly verified without exposing the secret. The framework supports watermark embedding both by fine-tuning a pre-trained model and by training from scratch, and is robust against common removal attacks such as fine-tuning, pruning, and watermark overwriting.

This repository provides reproducible implementations of all experiments reported in the paper for two benchmark datasets (CIFAR-10 and FMNIST5) with multiple model architectures (ResNet-18, VGG-16).

---

## Paper

**KeyNet: An Asymmetric Key-Style Framework for Watermarking Deep Learning Models**
Najeeb M. Jebreel, Josep Domingo-Ferrer, David Sánchez, Alberto Blanco-Justicia
*Applied Sciences*, vol. 11, no. 3, p. 999, MDPI, 2021
🔗 [Read on MDPI](https://www.mdpi.com/2076-3417/11/3/999)

---

## Repository Structure

```
KeyNet/
├── CIFAR10/                        # Experiments for the CIFAR-10 benchmark
│   ├── Experiments.ipynb           # Main notebook: training, watermarking, evaluation
│   ├── PruningNetworks.ipynb       # Pruning attack experiments
│   ├── keynet.py                   # KeyNet core class
│   ├── dataset.py                  # Dataset utilities
│   ├── utils.py                    # Helper functions
│   ├── models/                     # Model architectures
│   │   ├── resnet_models.py
│   │   ├── vgg_models.py
│   │   └── mlp_private.py          # Private MLP for watermark verification
│   ├── data/                       # Watermark carrier images (different distributions)
│   ├── figures/                    # Result figures
│   └── Results/                    # Saved experiment results
├── FMNIST5/                        # Experiments for the FMNIST5 benchmark
│   ├── Experiments.ipynb
│   ├── PruningNetworks.ipynb
│   ├── keynet.py
│   ├── dataset.py
│   ├── utils.py
│   ├── models/
│   │   ├── fmnistmodels.py
│   │   └── mlp_private.py
│   ├── data/
│   ├── figures/
│   └── Results/
├── requirements.txt
└── README.md
```

---

## Installation

### Prerequisites

| Dependency | Version |
|---|---|
| [Python](https://www.anaconda.com/download) | ≥ 3.6 |
| [PyTorch](https://pytorch.org/) | ≥ 1.6 |

### Setup

```bash
git clone https://github.com/najeebjebreel/KeyNet.git
cd KeyNet
pip install -r requirements.txt
```

---

## Pretrained Models

Pretrained models are available for download from [Google Drive](https://drive.google.com/file/d/1P7CuPe8lHp_V44_qXalQWeqAmgr0b_o0/view?usp=sharing). Place the downloaded files in the corresponding `Results/` folder of each benchmark.

---

## Reproducing Experiments

Each benchmark folder contains self-contained Jupyter notebooks. Open the one corresponding to the dataset of interest:

```bash
# CIFAR-10 experiments
jupyter notebook CIFAR10/Experiments.ipynb

# FMNIST5 experiments
jupyter notebook FMNIST5/Experiments.ipynb
```

The `Experiments.ipynb` notebook covers: training the original task, embedding the watermark (by fine-tuning or from scratch), ownership verification, and robustness evaluation against fine-tuning, pruning, and overwriting attacks.

For pruning-specific experiments:

```bash
jupyter notebook CIFAR10/PruningNetworks.ipynb
jupyter notebook FMNIST5/PruningNetworks.ipynb
```

---

## Citation

If you use this code or build upon this work, please cite:

```bibtex
@article{jebreel2021keynet,
  title     = {KeyNet: An Asymmetric Key-Style Framework for Watermarking Deep Learning Models},
  author    = {Jebreel, Najeeb Moharram and Domingo-Ferrer, Josep and S{\'a}nchez, David and Blanco-Justicia, Alberto},
  journal   = {Applied Sciences},
  volume    = {11},
  number    = {3},
  pages     = {999},
  year      = {2021},
  publisher = {MDPI},
  doi       = {10.3390/app11030999}
}
```

---

## Acknowledgment

This research was funded by the European Commission (projects H2020-871042 "SoBigData++" and H2020-101006879 "MobiDataLab"), the Government of Catalonia (ICREA Acadèmia Prizes to J. Domingo-Ferrer and D. Sánchez, FI grant to N. Jebreel, and grant 2017 SGR 705), and the Spanish Government (projects RTI2018-095094-B-C21 "Consent" and TIN2016-80250-R "Sec-MCloud"). The authors are with the UNESCO Chair in Data Privacy, but the views in this paper are their own and are not necessarily shared by UNESCO.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Affiliation

Developed at the **[CRISES Research Group](https://crises-deim.urv.cat/)**, [Universitat Rovira i Virgili (URV)](https://www.urv.cat/en/), Tarragona, Catalonia.

---

## Contact

For questions or issues, please open a [GitHub issue](https://github.com/najeebjebreel/KeyNet/issues) or contact [Najeeb Jebreel](mailto:nageeb3gebreel@gmail.com).
