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

The pretrained models can be accessed using this link: https://drive.google.com/drive/folders/1X2b3zbFpq7mkiQTCWBlyZiOle_ciqiZv?usp=sharing. 

## Dependencies

[Python 3.6](https://www.anaconda.com/download)

[PyTorch 1.6](https://pytorch.org/)

## Training and testing
The hyperparameters, the training of the original task, the embedding of the watermark and the performing of the other experiments can be easily done using the jupyter notebook: `Experiments.IPYNB`.

## Citation 
If you find our work useful please cite:

@Article{app11030999,
AUTHOR = {Jebreel, Najeeb Moharram and Domingo-Ferrer, Josep and Sánchez, David and Blanco-Justicia, Alberto},
TITLE = {KeyNet: An Asymmetric Key-Style Framework for Watermarking Deep Learning Models},
JOURNAL = {Applied Sciences},
VOLUME = {11},
YEAR = {2021},
NUMBER = {3},
ARTICLE-NUMBER = {999},
URL = {https://www.mdpi.com/2076-3417/11/3/999},
ISSN = {2076-3417},
ABSTRACT = {Many organizations devote significant resources to building high-fidelity deep learning (DL) models. Therefore, they have a great interest in making sure the models they have trained are not appropriated by others. Embedding watermarks (WMs) in DL models is a useful means to protect the intellectual property (IP) of their owners. In this paper, we propose KeyNet, a novel watermarking framework that satisfies the main requirements for an effective and robust watermarking. In KeyNet, any sample in a WM carrier set can take more than one label based on where the owner signs it. The signature is the hashed value of the owner&rsquo;s information and her model. We leverage multi-task learning (MTL) to learn the original classification task and the watermarking task together. Another model (called the private model) is added to the original one, so that it acts as a private key. The two models are trained together to embed the WM while preserving the accuracy of the original task. To extract a WM from a marked model, we pass the predictions of the marked model on a signed sample to the private model. Then, the private model can provide the position of the signature. We perform an extensive evaluation of KeyNet&rsquo;s performance on the CIFAR10 and FMNIST5 data sets and prove its effectiveness and robustness. Empirical results show that KeyNet preserves the utility of the original task and embeds a robust WM.},
DOI = {10.3390/app11030999}
}





## Funding
This research was funded by the European Commission (projects H2020-871042 “SoBigData++” and
603 H2020-101006879 “MobiDataLab”), the Government of Catalonia (ICREA Acadèmia Prizes to J. Domingo-Ferrer
604 and D. Sánchez, FI grant to N. Jebreel and grant 2017 SGR 705), and the Spanish Government (projects
605 RTI2018-095094-B-C21 “Consent” and TIN2016-80250-R “Sec-MCloud”).




