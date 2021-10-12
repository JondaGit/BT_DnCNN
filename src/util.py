import requests
from archive import extract
import tarfile

def fetch_dataset():
    url = 'https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/segbench/BSDS300-images.tgz'
    target_path = '../data/BSDS300-images.tgz'

    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(target_path, 'wb') as f:
            f.write(response.raw.read())
    # archive = tarfile.open('../data/BSDS300-images.tgz' , "r:*")
    # archive.extract('images/train', '../data/train')