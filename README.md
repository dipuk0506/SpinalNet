# SpinalNet: Deep Neural Network with Gradual Input

This repository contains scripts for training different variations of the SpinalNet and its counterparts.

## Abstract
Over the past few years, deep neural networks (DNNs) have garnered remarkable success in a diverse range of real-world applications. However, DNNs consider a large number of inputs and consist of a large number of parameters, resulting in high computational demand. We study the human somatosensory system and propose the SpinalNet to achieve higher accuracy with less computational resources. In a typical neural network (NN) architecture, the hidden layers receive inputs in the first layer and then transfer the intermediate outcomes to the next layer. In the proposed SpinalNet, the structure of hidden layers allocates to three sectors: 1) Input row, 2) Intermediate row, and 3) output row. The intermediate row of the SpinalNet contains a few neurons. The role of input segmentation is in enabling each hidden layer to receive a part of the inputs and outputs of the previous layer. Therefore, the number of incoming weights in a hidden layer is significantly lower than traditional DNNs. As all layers of the SpinalNet directly contributes to the output row, the vanishing gradient problem does not exist. We also investigate the SpinalNet fully-connected layer to several well-known DNN models and perform traditional learning and transfer learning. We observe significant error reductions with lower computational costs in most of the DNNs. We have also obtained the state-of-the-art (SOTA) performance for QMNIST, Kuzushiji-MNIST, EMNIST (Letters, Digits, and Balanced), STL-10, Bird225, Fruits 360, and Caltech-101 datasets. The scripts of the proposed SpinalNet are available at the following link: https://github.com/dipuk0506/SpinalNet


### Packages Used

torch, torchvision, numpy, random, matplotlib, time, os, copy, math.

Scripts are independent. The user can download an individual script and run. Except for scripts of the 'Transfer Learning' folder, scripts are downloading data from PyTorch during the execution.

### SOTA
KMNIST, QMNINT, EMNIST (Digits, Letters, Balanced), STL-10, Bird-225, Caltech-101,Fruits-360


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

### Simulation results on Kaggle:
[1] https://www.kaggle.com/dipuk0506/spinalnet-tl-pytorch-bird225-99-5
[2] https://www.kaggle.com/dipuk0506/spinalnet-cifar10-97-5-accuracy
[3] https://www.kaggle.com/dipuk0506/spinalnet-fruit360-99-99-accuracy
