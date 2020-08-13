# SpinalNet: Deep Neural Network with Gradual Input

This repository contains scripts for training different variations of the SpinalNet and its counterparts.

## Abstract
Deep neural networks (DNNs) have achieved the state of the art performance in numerous fields. However, DNNs need high computation times, and people always expect better performance with lower computation. Therefore, we study the human somatosensory system and design a neural network (SpinalNet) to achieve higher accuracy with lower computation time. This paper aims to present the SpinalNet. Hidden layers of the proposed SpinalNet consist of three parts: 1) Input row, 2) Intermediate row, and 3) output row. The intermediate row of the SpinalNet usually contains a small number of neurons. Input segmentation enables each hidden layer to receive a part of the input and outputs of the previous layer.  Therefore, the number of incoming weights in a hidden layer is significantly lower than traditional DNNs.  As the network directly contributes to outputs in each layer, the vanishing gradient problem of DNN does not exist. We integrate the SpinalNet as the fully-connected layer of the convolutional neural network (CNN), residual neural network (ResNet), and Dense Convolutional Network (DenseNet), Visual Geometry Group (VGG) network. We observe a significant error reduction with lower computation in most situations. We have received state-of-the-art performance for the QMNIST, Kuzushiji-MNIST, and EMNIST(digits) datasets. Scripts of the proposed SpinalNet is available at the following link: https://github.com/dipuk0506/SpinalNet 


### Packages Used

torch,
torchvision,
numpy,
random,
math.

Scripts are independent. The user can download an individual script and run. Scripts are downloading data from PyTorch during the execution.

### SOTA
KMNIST, QMNINT, EMNIST (Digits, Letters, Balanced)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/spinalnet-deep-neural-network-with-gradual/image-classification-on-emnist-balanced)](https://paperswithcode.com/sota/image-classification-on-emnist-balanced?p=spinalnet-deep-neural-network-with-gradual)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/spinalnet-deep-neural-network-with-gradual/image-classification-on-emnist-digits)](https://paperswithcode.com/sota/image-classification-on-emnist-digits?p=spinalnet-deep-neural-network-with-gradual)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/spinalnet-deep-neural-network-with-gradual/image-classification-on-qmnist)](https://paperswithcode.com/sota/image-classification-on-qmnist?p=spinalnet-deep-neural-network-with-gradual)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/spinalnet-deep-neural-network-with-gradual/image-classification-on-emnist-letters)](https://paperswithcode.com/sota/image-classification-on-emnist-letters?p=spinalnet-deep-neural-network-with-gradual)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/spinalnet-deep-neural-network-with-gradual/image-classification-on-kuzushiji-mnist)](https://paperswithcode.com/sota/image-classification-on-kuzushiji-mnist?p=spinalnet-deep-neural-network-with-gradual)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/spinalnet-deep-neural-network-with-gradual/image-classification-on-mnist)](https://paperswithcode.com/sota/image-classification-on-mnist?p=spinalnet-deep-neural-network-with-gradual)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/spinalnet-deep-neural-network-with-gradual/image-classification-on-fashion-mnist)](https://paperswithcode.com/sota/image-classification-on-fashion-mnist?p=spinalnet-deep-neural-network-with-gradual)

### Motivation
The SpinalNet tries to mimic the human somatosensory system to receive large data efficiently and to achieve better performance. (a) Half part of the human somatosensory system, presenting how our spinal cord receives sensory signals from our body. (b) Structure of the proposed SpinalNet. The proposed NN consists of the input row, the intermediate row, and the output row. The intermediate row contains multiple hidden layers. Each hidden layer receives a portion of the input. All layers except the first layer also receive outputs of the previous layer. The output layer adds the weighted outputs of all hidden neurons of the intermediate row. The user can also construct and train a SpinalNet for any arbitrary number of inputs, intermediate neurons, and outputs.


<img src="https://github.com/dipuk0506/SpinalNet/blob/master/Human_sensory.png" width="500">


### Universal Approximation

- Single hidden layer NN of large width is a universal approximator.

- If we can prove that, SpinalNet of a large depth can be equivalent to the single hidden layer NN of large width, the universal approximation is proved.

Following figure presents the visual proof of the universal approximation theorem for the proposed SpinalNet. A simplified version of SpinalNet in (a) can act as a NN of a single hidden layer, drawn in (b). Similarly, a 4 layer SpinalNet in (d)can be equal to a NN of one hidden layer (HL), containing four neurons, shown in (c). 


<img src="https://github.com/dipuk0506/SpinalNet/blob/master/UA_one_layer.png" width="500">


### Traditional hidden layer to Spinal Hidden Layer

Any traditional hidden layer can be converted to a spinal hidden layer. The traditional hidden layer in (a) is converted to a spinal hidden layer in (b). A spinal hidden layer has the structure of the proposed SpinalNet.

<img src="https://github.com/dipuk0506/SpinalNet/blob/master/SpinalHL.png" width="400">

## Results
### Regression

<img src="https://github.com/dipuk0506/SpinalNet/blob/master/Spinal_Regression.png" width="500">

### Classification
Detailed classification results are available in the paper.
Link to the paper:  https://arxiv.org/abs/2007.03347

### News
The script 'EMNIST_letters_VGG_and _SpinalVGG.py' uploaded on 10th July 2020 provides 95.88% accuracy for EMNIST-Letters and 91.05% accuracy for EMNIST-Balanced datasets with VGG-5 (Spinal FC) model. It is the new SOTA for these datasets.
