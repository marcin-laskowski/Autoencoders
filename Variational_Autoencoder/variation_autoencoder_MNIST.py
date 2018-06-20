'''
Implementation of the Denoising Autoencoder in PyTorch. Dataset used to perform this task is MNIST.
The input is binarized and Binary Cross Entropy has been used as the loss function.
The hidden layer contains 64 units.
In a variational autoEncoders, there is a strong assumption for the distribution that is learned in
the hidden representation. The hidden representation is constrained to be a multivariate gaussian.
'''

import os
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image


###################################################################################################
# Define Hyperparameters
num_epochs = 200
batch_size = 128
learning_rate = 1e-3


###################################################################################################
# FUNCTIONS - DATA OPERATIONS


# create directory 'auto_output' in which input and output will be saved
if not os.path.exists('./var_auto_output'):
    os.mkdir('./var_auto_output')


# function to visualise the image
def to_img(x):
    x = x.view(x.size(0), 1, 28, 28)
    return x


# function to plot sample image
def plot_sample_img(img, name):
    img = img.view(1, 28, 28)
    save_image(img, './sample_{}.png'.format(name))


# normalization where the entire range of values of X from min_value to max_value
def min_max_normalization(tensor, min_value, max_value):
    min_tensor = tensor.min()
    tensor = (tensor - min_tensor)
    max_tensor = tensor.max()
    tensor = tensor / max_tensor
    tensor = tensor * (max_value - min_value) + min_value
    return tensor


# Rounds the values of a tensor to the nearest integer, element-wise.
def tensor_round(tensor):
    return torch.round(tensor)


###################################################################################################
# INPUT DATA

# create Transforms which will perform operations on the dataset. All operations in one varible
img_transform = transforms.Compose([
    transforms.ToTensor(),  # transform input images to Tensor
    transforms.Lambda(lambda tensor:min_max_normalization(tensor, 0, 1)),  # normalization (0,1)
    transforms.Lambda(lambda tensor:tensor_round(tensor))  # round the values of tensor
])

# download MNIST dataset if not downloaded
dataset = MNIST('./data', transform=img_transform, download=True)
# load previously downloaded data in batches and shuffle
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


###################################################################################################
# NEURAL NETWORK

class variation_autoencoder(nn.Module):
    def __init__(self):
        super(variation_autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 400),
            nn.ReLU(True),
            nn.Linear(400, 40),
            nn.ReLU(True))
        self.decoder = nn.Sequential(
            nn.Linear(20, 400),
            nn.ReLU(True),
            nn.Linear(400, 28 * 28),
            nn.Sigmoid())

    # reparametrization trick which allow to compute backpropagation over random sampling
    def reparametrize(self, mu, logvar):
        var = logvar.exp()
        std = var.sqrt()
        eps = Variable(torch.cuda.FloatTensor(std.size()).normal_())
        return eps.mul(std).add(mu)

    # forward path
    def forward(self, x):
        h = self.encoder(x)  # encoder part
        mu = h[:, :20]  # calculate mean of the sample data
        logvar = h[:, 20:]  # lag scale of the value to remove negative values
        z = self.reparametrize(mu, logvar)  #
        x_hat = self.decoder(z)
        return x_hat, mu, logvar

    def generation_with_interpolation(self, x_one, x_two, alpha):
        hidden_one = self.encoder(x_one)
        hidden_two = self.encoder(x_two)
        mu_one = hidden_one[:, :20]
        logvar_one = hidden_one[:, 20:]
        mu_two = hidden_two[:, :20]
        logvar_two = hidden_two[:, 20:]
        mu = (1 - alpha) * mu_one + alpha * mu_two
        logvar = (1 - alpha) * logvar_one + alpha * logvar_two
        z = self.reparametrize(mu, logvar)
        generated_image = self.decoder(z)
        return generated_image


model = variation_autoencoder().cuda()
# definition of the LOSS FUNCTION: Binary Cross Entropy
BCE = nn.BCELoss()
# type of OPTIMIZER: Adam
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


###################################################################################################
# TRAINING

for epoch in range(num_epochs):
    for data in dataloader:
        img, _ = data
        img = img.view(img.size(0), -1)
        img = Variable(img).cuda()
        # ===================forward=====================
        x_hat, mu, logvar = model(img)
        NKLD = mu.pow(2).add(logvar.exp()).mul(-1).add(logvar.add(1))
        KLD = torch.sum(NKLD).mul(-0.5)
        KLD /= 128 * 784
        loss = BCE(x_hat, img) + KLD
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # ===================log========================
    print('epoch [{}/{}], loss:{:.4f}'
          .format(epoch + 1, num_epochs, loss.data[0]))
    if epoch % 10 == 0:
        x = to_img(img.cpu().data)
        x_hat = to_img(x_hat.cpu().data)

        save_image(x, './var_auto_output/x_{}.png'.format(epoch))
        save_image(x_hat, './var_auto_output/x_hat_{}.png'.format(epoch))

        batch = iter(dataloader).next()[0]
        batch = batch.view(batch.size(0), -1)
        batch = Variable(batch).cuda()
        x_one = batch[0:1]
        x_two = batch[1:2]
        generated_images = []
        for alpha in torch.arange(0.0, 1.0, 0.1):
            generated_images.append(model.generation_with_interpolation(x_one, x_two, alpha))
        generated_images = torch.cat(generated_images, 0).cpu().data
        save_image(generated_images.view(-1, 1, 28, 28),
                   './var_auto_output/generated_output_interpolate_{}.png'.format(epoch), nrow=1)
torch.save(model.state_dict(), './sim_variational_autoencoder.pth')
