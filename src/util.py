import requests
from archive import extract
import numpy as np
import torch
import math

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

# ---------------------------------------------------------------------------- #


# ----------------------- Image manipulation functions ----------------------- #

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

