import torch.nn as nn
import os
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import util
import random
import numpy as np
import torch

class DnCNN(nn.Module):
    def __init__(self):
        super(DnCNN, self).__init__()
        # in layer
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        # hidden layers
        hidden_layers = []
        for i in range(18):
          hidden_layers.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False))
          hidden_layers.append(nn.BatchNorm2d(64))
          hidden_layers.append(nn.ReLU(inplace=True))
        self.mid_layer = nn.Sequential(*hidden_layers)
        # out layer
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.relu1(self.conv1(x))
        out = self.mid_layer(out)
        out = self.conv3(out)
        return out

class ImageDataset(Dataset):
  def __init__(self, image_dir, patch_size=40):
      super(ImageDataset, self).__init__()
      self.image_filenames = [os.path.join(image_dir, x) for x in os.listdir(image_dir) if is_image_file(x)]
      self.patch_size = patch_size
      self.sigma = 25

  def __getitem__(self, index):
      input = np.asarray(Image.open(self.image_filenames[index]))
      H, W = input.shape[:2]
      rnd_h = random.randint(0, max(0, H - self.patch_size))
      rnd_w = random.randint(0, max(0, W - self.patch_size))

      patch_H = input[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]
    
      patch_H = util.augment_img(patch_H, random.randint(0, 7))
      patch_H = util.uint2tensor(patch_H)
      patch_L = patch_H.clone()

      # ----------------------------------- Noise ---------------------------------- #
      noise = torch.randn(patch_H.size()).mul_(self.sigma/255.0)
      patch_L.add_(noise)

      return { 'L': patch_L, 'H': patch_H, 'N': noise }
  
  def __len__(self):
      return len(self.image_filenames)

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".gif"])