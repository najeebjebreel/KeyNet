# Watermarking Deep Learning Models

This repository provides a PyTorch implementation of the paper KeyNet: An Asymmetric Key-Style Framework for Watermarking Deep Learning Models. This repository provides scripts for watermarking neural networks by fine tuning a pre-trained model or by embedding the watermark from scratch.

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
The repository contains one main jupyter notebook: `Experiments.IPYNB` where you can train (with and without watermark), predict and fine-tune models. 

Additionally, this repo contains some images from different distributions that used to embed the watermarks.

The code supports training and evaluating on [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html) and [FMNIST5] datasets.

## Dependencies

[Python 3.6](https://www.anaconda.com/download)

[PyTorch 1.6](https://pytorch.org/)

## Training and testing
The hyperparameters, the training of the original task, the embedding of the watermark and the performing of the other experiments can be easily done using the jupyter notebook: `Experiments.IPYNB`.

## Citation 
If you find our work useful please cite: 

## Funding
This research was funded by the European Commission (projects H2020-871042 “SoBigData++” and
603 H2020-101006879 "MobiDataLab), the Government of Catalonia (ICREA Acadèmia Prize to J. Domingo-Ferrer,
604 FI grant to N. Jebreel and grant 2017 SGR 705), and the Spanish Government (projects RTI2018-095094-B-C21
605 “Consent” and TIN2016-80250-R “Sec-MCloud”).




