import requests
from archive import extract
import numpy as np

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