import numpy as np
from skimage import io
from pathlib import Path
import re
#import ants
from skimage.transform import resize
from tqdm import tqdm
from skimage.morphology import skeletonize_3d, binary_dilation, binary_closing
from scipy.ndimage import distance_transform_edt
import tifffile as tif
from scipy.ndimage import binary_fill_holes
import cc3d
from scipy.io import loadmat, savemat
#import skan
import sknw
import networkx as nx
import pickle
import os
import matplotlib.pyplot as plt
import pandas as pd
import time
import scipy as sp
import vg
from pytransform3d.rotations import matrix_from_axis_angle
import multiprocessing
from scipy.ndimage import convolve as conv
from scipy.stats import multivariate_normal
from skimage import color, data, restoration
from RedLionfishDeconv import doRLDeconvolutionFromNpArrays
import argparse

def remove_small_comps_3d(image, thresh = 500*9):
    """
    

    Parameters
    ----------
    image : binary np array with uint8 elements
        3d numpy matrix, connected components will be removed form this image
    thresh : int64
        smallest connected components to keep

    Returns
    -------
    np.array with uint8 elements, binary
        binary image with connected components below the threshold removed.

    """
    img_lab, N = cc3d.connected_components(image,return_N=True)
    unique, counts = np.unique(img_lab, return_counts=True)
    unique_keep = unique[counts>thresh]
    unique_keep = np.delete(unique_keep,[0])
    img_filt = np.zeros(img_lab.shape).astype('int8')
    img_filt[np.isin(img_lab,unique_keep)] = 1
    return img_filt.astype('uint8')   

def fill_holes(img,thresh=100*9):
    #res = np.zeros(img.shape)
    for i in np.unique(img)[::-1]:
        _tmp = (img==i)*1.0
        _tmp = _tmp.astype('int8')
        _tmp = remove_small_comps_3d(_tmp,thresh=thresh)
        img[_tmp==1] = i
    res = img.astype('int8')
    return res

def _rotmat(vector, points):
    """
    Rotates a 3xn array of 3D coordinates from the +z normal to an
    arbitrary new normal vector.
    """
    
    vector = vg.normalize(vector)
    axis = vg.perpendicular(vg.basis.z, vector)
    angle = vg.angle(vg.basis.z, vector, units='rad')
    
    a = np.hstack((axis, (angle,)))
    R = matrix_from_axis_angle(a)
    
    r = sp.spatial.transform.Rotation.from_matrix(R)
    rotmat = r.apply(points)
    
    return rotmat



directory = Path('matt_raw_warped_single_upsampled')
files  = directory.glob('*-*_mean.npy')
files = sorted([x.as_posix() for x in files])
print(len(files))

parser = argparse.ArgumentParser(description='take hyperparameter inputs')

parser.add_argument('-c','--cfg', type=int, dest='cfg', action='store')

args = parser.parse_args()

file = args.cfg

i = file

print(i)

file = files[i]

min_prob = 0.75
max_var = 0.1

print('start')
if not os.path.exists(re.sub('led/','led_seg/',re.sub('mean','seg_nrn_dst',file))):
    if os.path.exists(re.sub('mean','2x_std',file)):
        mean = np.load(file)
        std = np.load(re.sub('mean','2x_std',file))
        seg = np.zeros(mean.shape[1:])
        seg[(mean[1,:,:,:] > min_prob) * (std[1,:,:,:] < max_var)] = 1
        seg[(mean[2,:,:,:] > min_prob) * (std[2,:,:,:] < max_var)] = 2
        seg = seg.astype('int8')
        seg = (seg==2)*1
        np.save(re.sub('led/','led_seg/',re.sub('mean','seg_nrn',file)),seg)
        np.save(re.sub('led/','led_seg/',re.sub('mean','seg_nrn_dst',file)),distance_transform_edt(1-seg).astype('float16'))
print('done')

np.random.shuffle(files)
for file in tqdm(files[::-1]):
    if not os.path.exists(re.sub('led/','led_seg/',re.sub('mean','seg_nrn_dst',file))):
        if os.path.exists(re.sub('mean','2x_std',file)):
            mean = np.load(file)
            std = np.load(re.sub('mean','2x_std',file))
            seg = np.zeros(mean.shape[1:])
            seg[(mean[1,:,:,:] > min_prob) * (std[1,:,:,:] < max_var)] = 1
            seg[(mean[2,:,:,:] > min_prob) * (std[2,:,:,:] < max_var)] = 2
            seg = seg.astype('int8')
            seg = (seg==2)*1
            np.save(re.sub('led/','led_seg/',re.sub('mean','seg_nrn',file)),seg)
            np.save(re.sub('led/','led_seg/',re.sub('mean','seg_nrn_dst',file)),distance_transform_edt(1-seg).astype('float16'))
            print('done2')
