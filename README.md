# Autoencoders

 Implementation of different types of autoencoders in PyTorch


## Introduction

Neural Networks are remarkably efficient tools to solve a number of really difficult problems. The first application of neural networks usually revolved around classification problems. Classification means that we have an image as an input and the output is for example simple decision whether it is a car or a chair. The input will have as many nodes as there are pixels in the input image, and the output will have to units, so we look at this two to find which fires the most to make the decision. Between this two there are hidden layers where the neural network is asked to build an inner representation of the problem that is efficient at recognizing these objects.

So the question is what is autoencoder?
An autoencoder is an interesting variant with two important changes:
- First, the number of neurons is the same in the input and the output, therefore we can expect that the output is an image that is not only the same size as the input but actually is the same image.

<p align="center">
  <img width="700" src="https://github.com/mlaskowski17/Autoencoders/blob/master/images/first.jpg">
</p>

- Second, we have a bottleneck (latent space) in one of these layers. This means that the number of neurons in that layer is much less than we would normally see, therefore it has to find the way to represent this kind of data the best it can with a much smaller number of neurons.

<p align="center">
  <img width="700" src="https://github.com/mlaskowski17/Autoencoders/blob/master/images/second.png">
</p>

To sum up:
**Autoencoders are neural networks that are capable of creating sparse representations of the input data and can therefore be used for image compression.** There are denoising autoencoders that after learning these sparse representations, can be presented with noisy images. What is even better is a variant that is called the variational autoencoder that not only learns these sparse representations, but can also draw new images as well. We can, for instance, ask it to create new handwritten digits and we can actually expect the results to make sense.



## Simple AutoEncoder

Autoencoders are one of the unsupervised deep learning models. The aim of an auto encoder is dimensionality reduction and feature discovery. An auto encoder is trained to predict its own input, but to prevent the model from learning the identity mapping, some constraints are applied to the hidden units.

The simplest form of an autoencoder is a feedforward neural network where the input x is fed to the hidden layer of h(x) and h(x) is then feed to calculate the output xˆ. A simple autoencoder used in the code is shown in the Figure below

<p align="center">
  <img width="700" src="https://github.com/mlaskowski17/Autoencoders/blob/master/images/simple_autoencoder.png">
</p>

In the code with the simple Autoencoder in PyTorch as the dataset was used MNIST. The input is binarized and Binary Cross Entropy has been used as the loss function. The hidden layer contains 64 units.


## Denoising AutoEncoder

n a denoising autoencoder the goal is to create a more robust model to noise. The motivation is that the hidden layer should be able to capture high level representations and be robust to small changes in the input. The input of a DAE is noisy data but the target is the original data without noise. So the DAE can be used to denoise the input.

In the PyTorch implementation of a DAE it was added some random noise to the data. Thank to that it was obtained corrupted inputs. In this case 20% noise has been added to the input.
