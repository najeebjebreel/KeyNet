# Watermarking Deep Learning Models

This repository is a primary version of the source code and models of the paper KeyNet: An Asymmetric Key-Style Framework for Watermarking Deep Learning Models. The repository uses PyTorch to implement the experiments and provides scripts for watermarking neural networks by fine tuning a pre-trained model or by embedding the watermark from scratch.

## Paper 

[KeyNet: An Asymmetric Key-Style Framework forWatermarking Deep Learning Models]
</br>
[Najeeb Moharram Jebreel](https://crises-deim.urv.cat/)<sup>1</sup>, [Josep Domingo-Ferrer](https://crises-deim.urv.cat/)<sup>1</sup>, [David Sánchez](https://crises-deim.urv.cat/)<sup>1</sup>, [Alberto Blanco-Justicia](https://crises-deim.urv.cat/)<sup>1</sup>
</br>
<sup>1 </sup> Universitat Rovira i Virgili, Department of Computer Engineering and Mathematics, CYBERCAT-Center for
Cybersecurity Research of Catalonia, UNESCO Chair in Data Privacy, Av. Països Catalans 26, 43007 Tarragona,
Catalonia
</br>

## Content
The repository contains one main jupyter notebook: `Experiments.IPYNB` in each data set folder. These notebooks can be used to train (with and without watermark), predict, embed watermarks and fine-tune models. 

Additionally, this repo contains some images from different distributions that used to embed the watermarks.

The code supports training and evaluating on [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html) and [FMNIST5] datasets.

## Pre-trained models

The pretrained models can be accessed using this link: https://drive.google.com/file/d/1P7CuPe8lHp_V44_qXalQWeqAmgr0b_o0/view?usp=sharing.

## Dependencies

[Python 3.6](https://www.anaconda.com/download)

[PyTorch 1.6](https://pytorch.org/)

## Training and testing
The hyperparameters, the training of the original task, the embedding of the watermark and the performing of the other experiments can be easily done using the jupyter notebook: `Experiments.IPYNB`.

## Citation 
If you find our work useful please cite:

Jebreel, N.;
Domingo-Ferrer, J.; Sánchez, D.;
Blanco-Justicia, A. KeyNet: An
Asymmetric Key-Style Framework
forWatermarking Deep Learning
Models. Appl. Sci. 2021, 11, 999.
https://doi.org/10.3390/
app11030999



## Funding
This research was funded by the European Commission (projects H2020-871042 “SoBigData++” and
603 H2020-101006879 “MobiDataLab”), the Government of Catalonia (ICREA Acadèmia Prizes to J. Domingo-Ferrer
604 and D. Sánchez, FI grant to N. Jebreel and grant 2017 SGR 705), and the Spanish Government (projects
605 RTI2018-095094-B-C21 “Consent” and TIN2016-80250-R “Sec-MCloud”).




