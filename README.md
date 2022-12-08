# SpinalNet: Deep Neural Network with Gradual Input
#### Paper Link: IEEE: https://ieeexplore.ieee.org/document/9802918
#### Paper Link: arXiv: https://arxiv.org/abs/2007.03347

This repository contains scripts for training different variations of the SpinalNet and its counterparts.

## Abstract
Deep neural networks (DNNs) have achieved the state of the art performance in numerous fields. However, DNNs need high computation times, and people always expect better performance in a lower computation. Therefore, we study the human somatosensory system and design a neural network (SpinalNet) to achieve higher accuracy with fewer computations. Hidden layers in traditional NNs receive inputs in the previous layer, apply activation function, and then transfer the outcomes to the next layer. In the proposed SpinalNet, each layer is split into three splits: 1) input split, 2) intermediate split, and 3) output split. Input split of each layer receives a part of the inputs. The intermediate split of each layer receives outputs of the intermediate split of the previous layer and outputs of the input split of the current layer. The number of incoming weights becomes significantly lower than traditional DNNs. The SpinalNet can also be used as the fully connected or classification layer of DNN and supports both traditional learning and transfer learning. We observe significant error reductions with lower computational costs in most of the DNNs. Traditional learning on the VGG-5 network with SpinalNet classification layers provided the state-of-the-art (SOTA) performance on QMNIST, Kuzushiji-MNIST, EMNIST (Letters, Digits, and Balanced) datasets. Traditional learning with ImageNet pre-trained initial weights and SpinalNet classification layers provided the SOTA performance on STL-10, Fruits 360, Bird225, and Caltech-101 datasets. The scripts of the proposed SpinalNet are available at the following link: https://github.com/dipuk0506/SpinalNet


### Independent Scripts
Scripts are independent. The user can download an individual script and run. Except for scripts of the 'Transfer Learning' folder, scripts are downloading data from PyTorch during the execution.

### SOTA
KMNIST, QMNINT, EMNIST (Digits, Letters, Balanced), STL-10, Bird-225, Caltech-101, Fruits-360.


### Motivation
We develop SpinalNet by mimicking several characteristics of the human somatosensory system to receive large input efficiently and to achieve better performance. (a) How our spinal cord is connected to our body for receiving and sending sensory signals from our body. (b) Structure of the proposed SpinalNet. Each layer of the proposed NN is split into input split, intermediate split, and output split. Each intermediate split receives a portion of the input. All intermediate splits except the intermediate split of the first layer also receive outputs of the previous intermediate split. The output split adds the weighted outputs of all intermediate splits. The user can also construct and train a SpinalNet for any arbitrary number of inputs, intermediate neurons, and outputs.


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

[4] https://www.kaggle.com/code/dipuk0506/stl-10-spinalnet

