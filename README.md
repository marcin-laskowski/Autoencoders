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



## Simple AutoEncoders

Autoencoders are one of the unsupervised deep learning models. The aim of an auto encoder is dimensionality reduction and feature discovery. An auto encoder is trained to predict its own input, but to prevent the model from learning the identity mapping, some constraints are applied to the hidden units.

The simplest form of an autoencoder is a feedforward neural network where the input x is fed to the hidden layer of h(x) and h(x) is then feed to calculate the output xˆ. A simple autoencoder used in the code is shown in the Figure below

<p align="center">
  <img width="700" src="https://github.com/mlaskowski17/Autoencoders/blob/master/images/simple_autoencoder.png">
</p>

In the code with the simple Autoencoder in PyTorch as the dataset was used MNIST. The input is binarized and Binary Cross Entropy has been used as the loss function. The hidden layer contains 64 units.

The fundamental problem with autoencoders, for generation, is that the latent space they convert their inputs to and where their encoded vectors lie, may not be continuous, or allow easy interpolation.


## Denoising AutoEncoders

In a denoising autoencoder the goal is to create a more robust model to noise. The motivation is that the hidden layer should be able to capture high level representations and be robust to small changes in the input. The input of a DAE is noisy data but the target is the original data without noise. So the DAE can be used to denoise the input.

In the PyTorch implementation of a DAE it was added some random noise to the data. Thank to that it was obtained corrupted inputs. In this case 20% noise has been added to the input.


## Variational AutoEncoders

The idea behind [variational autoencoders](https://arxiv.org/abs/1312.6114) is that instead of mapping any input to a fixed vector we want to map our input onto a distribution. So the only thing that is different in the variational autoencoder is that our normal bottleneck vector C is replaced by two separate vectors: one representing the mean of our distribution and the other one representing the standard deviation of that distribution. So whenever we need a vector to feed through our decoder network the only thing that we have to do is take a sample from the distribution and then feed it to the decoder.

A loss function is introduced which essentially is a sum of two losses — a generative loss, (which is a mean squared error that measures how well the output is generated from the given input. This error encourages the decoder to learn to reconstruct the input data accurately. If the decoder doesn’t reconstruct the data well in its output, this term would show a huge loss) and a latent loss (regulariser, a KL divergence between the encoder’s distribution and Gaussian distribution. It captures how closely the latent variables match the normal distribution or unit Gaussian)

As a property of VAE, the output from encoder is expected in Gaussian distribution with mean zero and variance one. This is done to ensure that similar features from the input data don’t end up with completely different representations. Thus, any deviation from the Gaussian distribution, is captured by the KL divergence loss. Further, to optimize KL divergence, instead of encoder generating a vector of real values, it’s made to generate a vector of means and a vector of standard deviations.

<p align="center">
  <img width="700" src="https://github.com/mlaskowski17/Autoencoders/blob/master/images/variational_AE.png">
</p>


## Disentangled Variational AutoEncoders

[Disentangled variational autoencoders](https://arxiv.org/abs/1606.05579) are a new class of the Variational autoencoders that has a lot of promising results. The basic idea behind this encoders is that you want to make sure that the different neurons in our latent distribution are uncorrelated, which means that they all try and learn something different about the input data. So to implement this the only thing we have to change is add one hyperparameter through our loss function that weighs how much this KL divergence is present in the loss function.


## Additional Materials
1. [What is Variational Autoencoder](https://jaan.io/what-is-variational-autoencoder-vae-tutorial/)
2. [Variational Autoencoders Explained](http://kvfrans.com/variational-autoencoders-explained/)
3. [Intuitively Understanding Variational Autoencoders](https://towardsdatascience.com/intuitively-understanding-variational-autoencoders-1bfe67eb5daf)
4. [Variational Autoencoders](https://www.jeremyjordan.me/variational-autoencoders/)
