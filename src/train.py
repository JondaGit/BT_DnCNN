from pickletools import optimize
import models
import util
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from torchvision import transforms, models
import torch
import random
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

def trainEpoch(epoch, model, criterion, optimizer, trainingDataLoader):
    epoch_loss = 0
    for iteration, data in enumerate(trainingDataLoader):
        optimizer.zero_grad()
        
        input = data['L'].cuda()
        target = data['H'].cuda()

        output = model(input)

        loss = criterion(output, target)
        epoch_loss += loss.item()
        
        loss.backward()
        optimizer.step()
    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(trainingDataLoader)))


def validateEpoch(epoch, model, testingDataLoader):
    avg_psnr = 0
    model.eval()
    with torch.no_grad():
        for data in testingDataLoader:
            input = data['L'].cuda()
            target = data['H'].cuda()

            output = model(input)
            
            psnr = util.calcPSNR(output, target)
            avg_psnr += psnr

        # images = ['../data/test/253027.jpg','../data/test/102061.jpg','../data/test/101085.jpg']
        # sigmas = [25,12.5,0]
        # # zoomParams = [
        # #     util.ZoomParams(xpos=150, ypos=60, size=80),
        # #     util.ZoomParams(xpos=40, ypos=100, size=80),
        # #     util.ZoomParams(xpos=40, ypos=100, size=80)
        # # ]
        # util.makeSamplePlot(model, images, sigmas, f"DnCNN[R] - Epoch {epoch}", ['Sigma 25', 'Sigma 12.5','Sigma 0'], figLocation=f"../figures/dncnn-r_epochs/dncnn-r_epoch_{epoch}.png")
        print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(testingDataLoader)))


def trainModel():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainDataset = models.ImageDataset("../data/train", mode="train")
    testDataset = models.ImageDataset("../data/test", mode="test")
    trainingDataLoader = DataLoader(dataset=trainDataset, batch_size=64, shuffle=True, pin_memory=True, drop_last=True, num_workers=4)
    testingDataLoader = DataLoader(dataset=testDataset, batch_size=1, shuffle=True, pin_memory=True, drop_last=False)

    # model declaration and hyperparameters
    model = models.DnCNN().to(device)
    criterion = nn.L1Loss().to(device)
    lr = 1e-4
    optimizer = optim.Adam(model.parameters(), lr=lr)

    num_epochs = 100000
    for epoch in range(1, num_epochs + 1):
        trainEpoch(epoch, model, criterion, optimizer, trainingDataLoader)
        if epoch % 500 == 0:
            validateEpoch(epoch, model, testingDataLoader)
            model_out_path = "../models/dncnn-r/dncnn_epoch_{}.pth".format(epoch)
            model.save(model_out_path, epoch, optimizer)
      

