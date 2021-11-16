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

        head = []
        head.append(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True))
        head.append(nn.ReLU(inplace=True))

        body = []
        for _ in range(15):
          body.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=True))
          body.append(nn.BatchNorm2d(64, momentum=0.9, eps=1e-04, affine=True))
          body.append(nn.ReLU(inplace=True))
        
        tail = []
        tail.append(nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, padding=1, bias=True))

        self.model = nn.Sequential(*(head + body + tail))

    def forward(self, img):
        noise = self.model(img)
        return img - noise

class ImageDataset(Dataset):
  def __init__(self, image_dir, patch_size=40, mode="train"):
      super(ImageDataset, self).__init__()
      self.image_filenames = np.array([os.path.join(image_dir, x) for x in os.listdir(image_dir) if is_image_file(x)])
      if mode == "test":
        self.image_filenames = self.image_filenames[np.random.choice(len(self.image_filenames), size=30, replace=False)]
      self.patch_size = patch_size
      self.sigma = 25
      self.mode = mode

  def __getitem__(self, index):
      img_H = np.asarray(Image.open(self.image_filenames[index]))

      if self.mode == "train":
        H, W = img_H.shape[:2]
        rnd_h = random.randint(0, max(0, H - self.patch_size))
        rnd_w = random.randint(0, max(0, W - self.patch_size))

        patch_H = img_H[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]
        
        patch_H = util.augment_img(patch_H, random.randint(0, 7))
        img_H = util.uint2tensor(patch_H)
        img_L = img_H.clone()

        # ----------------------------------- Noise ---------------------------------- #
        noise = torch.randn(img_L.size()).mul_(self.sigma/255.0)
        img_L.add_(noise)
      else:
        img_H = util.uint2float(img_H)
        img_L = np.copy(img_H)
        img_L += np.random.normal(0, self.sigma/255.0, img_L.shape)

        img_L = util.float2tensor(img_L)
        img_H = util.float2tensor(img_H)

      return { 'L': img_L, 'H': img_H }
  
  def __len__(self):
      return len(self.image_filenames)

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".gif"])