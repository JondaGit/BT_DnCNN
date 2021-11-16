import requests
from archive import extract
import numpy as np
import torch
import math
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from dataclasses import dataclass

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


# --------------------------------- Plotting --------------------------------- #
@dataclass
class ZoomParams:
    xpos: int
    ypos: int
    size: int
    color = 'white'
    zoom = 2.
    padding = .5


def makePlot(images = [], title="Title", savePath=None, show=False):
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(images[0])
    ax[0].axis("off")
    ax[1].imshow(images[1])
    ax[1].axis("off")
    plt.subplots_adjust(wspace=0.01, hspace=0)
    if savePath:
        plt.savefig(savePath, bbox_inches='tight', dpi=800)
    if show:
        plt.show()

def plotRow(axes, imgs, labels, zoomParams: ZoomParams=None):
    for index, axis in enumerate(axes):
        axis.tick_params(
            bottom=False,
            left=False,
            labelleft=False,
            labelbottom=False)
        axis.imshow(imgs[index])
        axis.set_xlabel(labels[index], fontsize=14)
        if zoomParams:
            insetZoom(axis, imgs[index], zoomParams)

def insetZoom(axis, image, zoomParams: ZoomParams):
    inset_axes = zoomed_inset_axes(axis,
                               zoom=zoomParams.zoom, 
                               loc='upper right', 
                               borderpad=zoomParams.padding)
    inset_axes.set_xlim(zoomParams.xpos, zoomParams.xpos + zoomParams.size)
    inset_axes.set_ylim(zoomParams.ypos + zoomParams.size, zoomParams.ypos)
    for spine in inset_axes.spines.values():
            spine.set_edgecolor(zoomParams.color)
            spine.set_alpha(0.5)
    inset_axes.tick_params(
        bottom=False,      # ticks along the bottom edge are off
        left=False,
        labelleft=False,
        labelbottom=False)
    inset_axes.imshow(image)
    axis.indicate_inset_zoom(inset_axes, edgecolor=zoomParams.color)
            
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
# ---------------------------------------------------------------------------- #