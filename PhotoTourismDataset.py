from skimage.io import imread
from torch.utils.data import Dataset
import pandas as pd
from os.path import join
import numpy as np

from colmap.scripts.python.read_write_model import read_model, qvec2rotmat
from colmap.scripts.python.read_write_dense import read_array
from imageio.v2 import imread
import matplotlib.pyplot as plt
import h5py
from time import time


class PhotoTourismDataset(Dataset):
    """
        A class representing a dataset of images and their poses
    """

    def __init__(self, src, data_transform=None):
        super(PhotoTourismDataset, self).__init__()
        self.src = src
        self.cameras, self.images, self.points = read_model(path=src+'/dense/sparse', ext='.bin')
        self.indices = [i for i in self.cameras]
        

    def __len__(self):
        return len(self.images)

    def __getitem__(self, id):
        idx = self.indices[id]
        src = self.src

        im = imread(src + '/dense/images/' + self.images[idx].name)
        depth = read_array(src + '/dense/stereo/depth_maps/' + self.images[idx].name + '.photometric.bin')
        min_depth, max_depth = np.percentile(depth, [5, 95])
        depth[depth < min_depth] = min_depth
        depth[depth > max_depth] = max_depth

        # reformat data
        q = self.images[idx].qvec
        R = qvec2rotmat(q)
        T = self.images[idx].tvec
        p = self.images[idx].xys
        pars = self.cameras[idx].params
        K = np.array([[pars[0], 0, pars[2]], [0, pars[1], pars[3]], [0, 0, 1]])
        pids = self.images[idx].point3D_ids
        v = pids >= 0
        print('Number of (valid) points: {}'.format((pids > -1).sum()))
        print('Number of (total) points: {}'.format(v.size))
        
        pose = np.concatenate([T, q], axis=0)
        
        # get also the clean depth maps
        base = '.'.join(self.images[idx].name.split('.')[:-1])
        # with h5py.File(src + '/dense/stereo/depth_maps_clean_300_th_0.10/' + base + '.h5', 'r') as f:
        #     depth_clean = f['depth'][()]

        return {
            'img': im,
            'pose': pose,
            'scene': 0,
        }
            # 'depth_raw': depth,
            # # 'depth': depth_clean,
            # 'K': K,
            # 'q': q,
            # 'R': R,
            # 'T': T,
            # 'xys': p,
            # 'ids': pids,
            # 'valid': v}

# test dataset
if __name__ == '__main__':
    print('Testing PhotoTourismDataset')
    root = '/home/runyi/Desktop'
    seq = 'notre_dame_front_facade'
    src = root + '/' + seq
    dataset = PhotoTourismDataset(src)

    sample = dataset[0]
    print('Sample: {}'.format(sample))
    print('Sample keys: {}'.format(sample.keys()))
    print('Sample img shape: {}'.format(sample['img'].shape))
    print('Sample pose shape: {}'.format(sample['pose'].shape))
    