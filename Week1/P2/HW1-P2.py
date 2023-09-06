#!/usr/bin/env python
# coding: utf-8

# ### Can you generate faces?
# ##### We expect you to use images to generate new ones with a Generative Model of your choice. You have to write a dataloader to read images from the folder 'cropped/', write a Generative Model class, a loss function, a training loop, and visualize your generated images.

# In[1]:


import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import imageio as Image
from torchvision.utils import make_grid
from torchsummary import summary
from torch.utils.data import Dataset, DataLoader
from mpl_toolkits.axes_grid1 import ImageGrid
from tqdm import tqdm
import numpy as np
import cv2


# ### Nothing to change here (This cell downloads and unzips the data).

# In[2]:


get_ipython().system('wget https://www.dropbox.com/s/g0w7a3x1aw3oonf/SimpsonFaces.zip?dl=0')

get_ipython().system('unzip SimpsonFaces.zip?dl=0')

get_ipython().system('ls')


# ## Dataloader
# ####  Write a dataloader to read images from the folder 'cropped/' (Note that the transform *trans* resizes the images to 32x32)

# In[3]:


trans = transforms.Compose([transforms.ToPILImage(),transforms.ToTensor(),transforms.Resize([32,32]),transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])

# Replace the question marks '?' by the necessary code
batch_size = 64

class MyDataset(Dataset):
  def __init__(self, image_path, transform = trans):
    self.image_path = image_path
    self.images = os.listdir(image_path)
    self.transform = transform

  def __len__(self):
    return len(self.images)

  def __getitem__(self,idx):
    im = self.images[idx]
    im = cv2.imread(self.image_path+im)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = self.transform(im)
    return im


# In[4]:


dataset = MyDataset("cropped/")

train_loader = DataLoader(dataset, batch_size = batch_size)


# #### Visualize the data
# ##### Get a batch from the dataloader and visualize its images

# In[ ]:


# ADD CODE HERE
batch = next(iter(train_loader))
figure, ax = plt.subplots(nrows=8, ncols=8, figsize=(12, 8))
for i in range(batch_size):
    idx = np.random.randint(1,len(dataset))
    image = dataset[idx]
    ax.ravel()[i].imshow(image.permute(1,2,0))
    ax.ravel()[i].set_title(i)
    ax.ravel()[i].set_axis_off()
plt.tight_layout(pad=1)
plt.show()


# ### Generative Model class
# #### Write a Generative Model class in the following cell

# In[6]:


device = torch.device("cuda:0")


# In[7]:


class Generator(nn.Module):
    def __init__(self,dz):
        super(Generator, self).__init__()

        self.genc = nn.Sequential(
            nn.Linear(dz, 1024 * 8 * 8),
            nn.ReLU(),
            nn.Unflatten(1,(1024,8,8)),
            nn.ConvTranspose2d(1024, 512, 5,2,2,output_padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.ConvTranspose2d(512, 256, 5,1,2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 5,1,2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.ConvTranspose2d(128, 64, 5,2,2,output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 5,1,2),
            nn.Tanh()
        )

    def forward(self, z):
        return self.genc(z)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.disc = nn.Sequential(
        nn.Conv2d(3, 64, 5,2,2),
        nn.BatchNorm2d(64),
        nn.LeakyReLU(0.2),
        nn.Conv2d(64, 128, 5,1,2),
        nn.BatchNorm2d(128),
        nn.LeakyReLU(0.2),
        nn.Conv2d(128, 256, 5,1,2),
        nn.BatchNorm2d(256),
        nn.LeakyReLU(0.2),
        nn.Conv2d(256, 512, 5,2,2),
        nn.BatchNorm2d(512),
        nn.LeakyReLU(0.2),
        nn.Flatten(),
        nn.Linear(8 * 8 * 512, 1),
        nn.Sigmoid()
        )

    def forward(self, x):
        return self.disc(x)


# In[8]:


def weights_init(m):
    """Reinitialize model weights. GAN authors recommend them to be sampled from N(0,0.2)"""
    classname = m.__class__.__name__
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
        nn.init.normal_(m.weight.data, mean=0.0, std=0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# In[9]:


dz = 100
disc = Discriminator().to(device)
disc.apply(weights_init)
gen = Generator(dz).to(device)
gen.apply(weights_init)


# In[10]:


summary(gen,(100,))


# In[13]:


summary(disc,(3,32,32))


# ## Loss
# #### Define Loss function in the following cellPedarSag

# In[14]:


criterion = nn.BCELoss()
discOptim = torch.optim.Adam(disc.parameters(), lr=2e-4, betas = (0.5,0.999))
genOptim = torch.optim.Adam(gen.parameters(), lr=2e-4, betas = (0.5,0.999))


# ### Training Loop
# #### Define optimizer, write the training loop in the following cell, and plot the loss

# In[15]:


glosslist = []
dlosslist = []


# In[16]:


def display_image_grid(images, num_rows, num_cols, title_text):
    images = images.cpu()
    fig = plt.figure(figsize=(num_cols*3., num_rows*3.), )
    grid = ImageGrid(fig, 111, nrows_ncols=(num_rows, num_cols), axes_pad=0.15)

    for ax, im in zip(grid, images):
        ax.imshow(im.permute(1,2,0))
        ax.axis("off")

    plt.suptitle(title_text, fontsize=20)
    plt.show()


# In[20]:


def train(epochs):
  for i in range(0,epochs):
      print(i)
      for n, data in enumerate(train_loader):
          images = data.to(device, dtype=torch.float)
          # discriminator real training
          disc.zero_grad()
          y = disc(images)

          real_labels = (torch.rand(len(images))/2 + 0.7).to(device, dtype=torch.float)
          flipReals = (torch.rand(len(images))<0.03).to(device)
          for i in range(len(real_labels)):
            if flipReals[i]:
              real_labels[i]=0
          dLossReal = criterion(torch.squeeze(y), real_labels)
          dLossReal.backward()
          # discriminator fake training
          noise = torch.randn(len(images),dz).to(device)
          fake_labels = (torch.rand(len(images)) * 0.3).to(device, dtype=torch.float)
          flipFakes = (torch.rand(len(images))<0.03).to(device)
          for i in range(len(fake_labels)):
            if flipFakes[i]:
              fake_labels[i]=1
          xhat = gen(noise)
          yhat = disc(xhat.detach())
          dLossFake = criterion(torch.squeeze(yhat), fake_labels)
          dLossFake.backward()
          discOptim.step()

          # generator training
          gen.zero_grad()
          gen_reverse_labels = (torch.rand(len(images))/2 + 0.7).to(device, dtype=torch.float)
          gyhat = disc(xhat)
          gLoss = criterion(torch.squeeze(gyhat),gen_reverse_labels)
          gLoss.backward()
          genOptim.step()

      with torch.no_grad():
          out = gen(torch.randn(len(images),dz).to(device))
          display_image_grid(out, 2, 4, "")

          print(f"Generator Loss: {gLoss.item()}, Descriminator Loss: {dLossFake.item()+dLossReal.item()}")

          dlosslist.append(dLossFake.item()+dLossReal.item())
          glosslist.append(gLoss.item())

          if i % 4 == 0:
              plt.figure(figsize=(2,2))
              plt.plot(glosslist)
              plt.plot(dlosslist)
              plt.legend(['gen loss','disc loss'])
              plt.show()


# In[ ]:


train(200) # I deleted the outputs to decrease the file size


# ## Generate and Plot Data
# #### Generate a batch of 64 images and plot them in subplots of 8 rows and 8 columns.

# In[ ]:


# ADD CODE HERE
generations = gen(torch.randn(batch_size,dz).to(device))
images = generations.cpu()
fig = plt.figure(figsize=(8., 8.), )
grid = ImageGrid(fig, 111, nrows_ncols=(8, 8), axes_pad=0.1)

for ax, im in zip(grid, images):
    ax.imshow((im.permute(1,2,0)+1)/2)
    ax.axis("off")
plt.show()

