import requests
from archive import extract
import numpy as np
import torch
import math
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from dataclasses import dataclass
from PIL import Image

def fetch_dataset():
    url = 'https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/segbench/BSDS300-images.tgz'
    target_path = '../data/BSDS300-images.tgz'

    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(target_path, 'wb') as f:
            f.write(response.raw.read())
    # archive = tarfile.open('../data/BSDS300-images.tgz' , "r:*")
    # archive.extract('images/train', '../data/train')

# ------------------------ Data manupulation functions ----------------------- #
def uint2float(img):
    return np.float32(img / 255.)

def float2tensor(img):
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float()

def tensor2uint(img):
    img = img.data.squeeze().float().clamp_(0, 1).cpu().numpy()
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))
    return np.uint8((img*255.0).round())

def uint2tensor(img):
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().div(255.)
# ---------------------------------------------------------------------------- #


# ----------------------- Image manipulation functions ----------------------- #

def augment_img(img, mode=0):
    """Augment input image (np array format)
    """
    if mode == 0:
        return img
    elif mode == 1:
        return np.flipud(np.rot90(img))
    elif mode == 2:
        return np.flipud(img)
    elif mode == 3:
        return np.rot90(img, k=3)
    elif mode == 4:
        return np.flipud(np.rot90(img, k=2))
    elif mode == 5:
        return np.rot90(img)
    elif mode == 6:
        return np.rot90(img, k=2)
    elif mode == 7:
        return np.flipud(np.rot90(img, k=3))

def addNoise(img, db=25):
    """
    Function to add gaussian noise to target image

    INPUT - numpy uint8 array image

    OUTPUT - tensor
    """
    print(img.shape)
    noise = np.random.normal(0, db / 255., img.shape)
    return float2tensor(noise + (img/255.)), float2tensor(noise)

# ----------------------------------- Stats ---------------------------------- #
def calcPSNR(clean, noisy):
    """
    Calculate PSNR

    INPUT - two tensors of clean and noisy image
    """
    mse = torch.mean((noisy - clean) ** 2)
    return 20 * math.log10(1. / (mse ** .5))
# ---------------------------------------------------------------------------- #

# ---------------------------- Image segmentation ---------------------------- #
def patchify(img, size=256):
    H, W = img.shape[:2];
    output = np.empty((int(H/size), int(W/size), size, size, 3), dtype=np.uint8)
    for y in range(output.shape[0]):
        for x in range(output.shape[1]):
            output[y][x] = img[y * size:y * size + size, x * size: x * size + size]
    return output

def unpatchify(patches):
    rows, cols, size = patches.shape[:3]
    output = np.empty((rows * size, cols * size, 3), dtype=np.uint8)
    for y in range(rows):
        for x in range(cols):
            output[y * size:y * size + size, x * size: x * size + size] = patches[y][x]
    return output

def patchifyAdaptive(img, baseSize=256):
    """Patchify image by dividing to unevenly sized patches.
    Maximum patch size is 256 + 255 px, width and height do not have
    to be the same.
    """
    sizex = baseSize + (img.shape[1] % baseSize) // (img.shape[1] // baseSize)
    sizey = baseSize + (img.shape[0] % baseSize) // (img.shape[0] // baseSize)
    rows = (img.shape[0] // baseSize) - 1 # Height
    cols = (img.shape[1] // baseSize) - 1 # Width
    
    output = []
    for row in range(rows):
        outputRow = []
        for col in range(cols):
            outputRow.append(img[row*sizey:row*sizey + sizey, col*sizex:col*sizex + sizex])
        outputRow.append(img[row*sizey:row*sizey + sizey, cols*sizex:img.shape[1]])
        output.append(outputRow)
    # Last line
    outputRow = []
    for col in range(cols):
        outputRow.append(img[rows*sizey:img.shape[0], col*sizex:col*sizex + sizex])
    outputRow.append(img[rows*sizey:img.shape[0], cols*sizex:img.shape[1]])
    output.append(outputRow)
    return output

def unpatchifyAdaptive(patches, width, height):
    reconstructed = np.empty((height,width,3), dtype=np.uint8)
    x = 0
    y = 0
    for row in patches:
        for p in row:
            reconstructed[y: y + p.shape[0], x: x + p.shape[1]] = p
            x += p.shape[1]
        x = 0
        y += row[0].shape[0]
    return reconstructed
# ---------------------------------------------------------------------------- #
