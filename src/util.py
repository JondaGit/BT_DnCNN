import requests
from archive import extract
import numpy as np
import torch
import math
import random
import matplotlib.pyplot as plt

def fetch_dataset():
    url = 'https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/segbench/BSDS300-images.tgz'
    target_path = '../data/BSDS300-images.tgz'

    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(target_path, 'wb') as f:
            f.write(response.raw.read())
    # archive = tarfile.open('../data/BSDS300-images.tgz' , "r:*")
    # archive.extract('images/train', '../data/train')

def Im2PatchNP(img, win, stride=1):
    '''
    Patch creating function, input dataset must be of numpy array format
    '''
    k = 0
    endc = img.shape[2]
    endw = img.shape[1]
    endh = img.shape[0]
    patch = img[0:endh-win+0+1:stride, 0:endw-win+0+1:stride, :]
    TotalPatNum = patch.shape[0] * patch.shape[1]
    Y = np.zeros([win*win,TotalPatNum, endc], np.float32)
    for i in range(win):
        for j in range(win):
            patch = img[i:endh-win+i+1:stride, j:endw-win+j+1:stride, :]
            Y[k,:,:] = np.array(patch[:]).reshape(TotalPatNum, endc)
            k = k + 1
    return Y.reshape([win, win, TotalPatNum, endc])

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

def calcPSNR(clean, noisy):
    """
    Calculate PSNR

    INPUT - two tensors of clean and noisy image
    """
    mse = torch.mean((noisy - clean) ** 2)
    return 20 * math.log10(1. / (mse ** .5))


def makePlot(images = [], title="Title", savePath=None, show=False):
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(images[0])
    ax[0].axis("off")
    ax[1].imshow(images[1])
    ax[1].axis("off")
    plt.subplots_adjust(wspace=0.01, hspace=0)
    plt.savefig('plot.png', bbox_inches='tight', dpi=800)

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