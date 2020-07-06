# SpinalNet: SpinalNet: Deep Neural Network with Gradual Input

This repository contains scripts for training different variations of the SpinalNet and its counterparts.

## Abstract
Deep neural networks (DNNs) have achieved the state of the art performance in numerous fields. However, DNNs need high computation times, and people always expect better performance with lower computation. Therefore, we study the human somatosensory system and design a neural network (SpinalNet) to achieve higher accuracy with lower computation time. This paper aims to present the SpinalNet. Hidden layers of the proposed SpinalNet consist of three parts: 1) Input row, 2) Intermediate row, and 3) output row. The intermediate row of the SpinalNet usually contains a small number of neurons. Input segmentation enables each hidden layer to receive a part of the input and outputs of the previous layer.  Therefore, the number of incoming weights in a hidden layer is significantly lower than traditional DNNs.  As the network directly contributes to outputs in each layer, the vanishing gradient problem of DNN does not exist. We integrate the SpinalNet as the fully-connected layer of the convolutional neural network (CNN), residual neural network (ResNet), and Dense Convolutional Network (DenseNet), Visual Geometry Group (VGG) network. We observe a significant error reduction with lower computation in most situations. Scripts of the proposed SpinalNet is available at the following link: https://github.com/dipuk0506/SpinalNet 

### Packages Used

torch,
torchvision,
numpy,
random,
math.

[![INSERT YOUR GRAPHIC HERE](https://github.com/dipuk0506/SpinalNet/blob/master/Human_sensory.png)]()


