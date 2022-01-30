import os
from util import patchifyAdaptive, uint2float, float2tensor, tensor2uint, calcPSNR, uint2tensor, unpatchifyAdaptive
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from dataclasses import dataclass
from PIL import Image
import numpy as np

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

def makeSamplePlot(model, images, sigmas, title, subtitles=None, zoomParams=None, figLocation='test'):
    fig = plt.figure(constrained_layout=True)
    fig.suptitle(title, fontsize=26)
    fig.set_figheight(6 * len(images))
    fig.set_figwidth(20)
    subfigs = fig.subfigures(nrows=len(images), ncols=1, hspace=.15)
    for row, subfig in enumerate(subfigs):
        if subtitles:
            subfig.suptitle(subtitles[row], fontsize=20)
        img_H = np.asarray(Image.open(images[row]))
        img_H = uint2float(img_H)
        img_L = np.copy(img_H)
        img_L += np.random.normal(0, sigmas[row]/255.0, img_L.shape)
        img_L = float2tensor(img_L)
        img_H = float2tensor(img_H)
        with torch.no_grad():
            img_E = model(img_L.unsqueeze(0).cuda())  
        HL_psnr = calcPSNR(img_H, img_L)
        HE_psnr = calcPSNR(img_H, img_E.cpu())
        img_H = tensor2uint(img_H)
        img_L = tensor2uint(img_L)
        img_E = tensor2uint(img_E.squeeze(0))

        axs = subfig.subplots(nrows=1, ncols=3) 
        labels = [
            f"Noisy image ({HL_psnr:.4f} dB)",
            f"Denoised image ({HE_psnr:.4f} dB)",
            "Reference image"
        ]
        plotRow(axs, [img_L, img_E, img_H], labels, zoomParams[row] if zoomParams else None)
    plt.savefig(figLocation, bbox_inches='tight',  facecolor="w", dpi=400)

def makeSamplePlotISO(model, images, title, zoomParams=None, figLocation='test2'):
    fig, ax = plt.subplots(3,3, gridspec_kw={ "wspace": 0.0 })
    fig.suptitle(title, fontsize=26)
    fig.set_figheight(6 * len(images))
    fig.set_figwidth(16)
    for row, axis in enumerate(ax):
        img_H = images[row]["img_H"]
        img_L = images[row]["img_L"]
        img_L = uint2tensor(img_L)
        img_H = uint2tensor(img_H)
        with torch.no_grad():
            img_E = model(img_L.unsqueeze(0).cuda())  
        HL_psnr = calcPSNR(img_H, img_L)
        HE_psnr = calcPSNR(img_H, img_E.cpu())
        img_H = tensor2uint(img_H)
        img_L = tensor2uint(img_L)
        img_E = tensor2uint(img_E.squeeze(0))

        labels = [
            f"Noisy image ({HL_psnr:.4f} dB)",
            f"Denoised image ({HE_psnr:.4f} dB)",
            "Reference image"
        ]
        plotRow(axis, [img_L, img_E, img_H], labels, zoomParams[row] if zoomParams else None)
    plt.savefig(figLocation, bbox_inches='tight',  facecolor="w", dpi=400)

def makePlotRealISO(images, model, savePath=None, zoomParams=None):
    for idx, image in enumerate(images):
        _, ax = plt.subplots(1, 2)
        ax[0].imshow(image)
        ax[0].axis("off")
        height, width = image.shape[:2]
        patches = patchifyAdaptive(image)
        denoised = []
        for row in patches:
            denoisedrow = []
            for patch in row:
                tensor = uint2tensor(patch).cuda()
                denoisedrow.append(tensor2uint(model(tensor.unsqueeze(0)).squeeze(0)))
            denoised.append(denoisedrow)
        denoisedImg = unpatchifyAdaptive(denoised, width, height)
        ax[1].imshow(denoisedImg)
        ax[1].axis("off")
        plt.subplots_adjust(wspace=0.01, hspace=0)
        if zoomParams:
            insetZoom(ax[0], image, zoomParams[idx])
            insetZoom(ax[1], denoisedImg, zoomParams[idx])
        if savePath:
            dir = os.path.dirname(savePath[idx])
            if not os.path.isdir(dir):
                os.makedirs(dir)
            plt.savefig(savePath[idx], bbox_inches='tight', dpi=800)

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
            
# ---------------------------------- Utility --------------------------------- #

def scanFolder(dir, outputFormat):
    """Scan folder for all files and then return list of them
    and formatted list with filenames formated into outputFormat."""
    dirFiles = []
    formattedFiles = []
    for file in os.listdir(dir):
        dirFiles.append(os.path.join(dir, file))
        formattedFiles.append(outputFormat.format(os.path.splitext(file)[0]))
    return { "input": dirFiles, "output": formattedFiles }