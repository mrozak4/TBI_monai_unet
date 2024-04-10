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
from utilities import remove_small_comps_3d, fill_holes, _rotmat, closest_node


directory = Path('matt_raw_warped_single_upsampled')
files  = directory.glob('*-*_mean.npy')
files = sorted([x.as_posix() for x in files])
print(len(files))


min_prob = 0.5
max_var = 0.2

np.random.shuffle(files)
for file in tqdm(files[::-1]):
    # Check if the segmented file already exists
    seg_file = re.sub('led/','led_seg/',re.sub('mean','seg',file))
    if not os.path.exists(seg_file):
        # Check if the standard deviation file exists
        std_file = re.sub('mean','2x_std',file)
        if os.path.exists(std_file):
            print(file)
            mean = np.load(file)
            std = np.load(std_file)
            seg = np.zeros(mean.shape[1:]).astype('int8')
            # Segment the image based on mean and standard deviation thresholds
            seg[(mean[1,:,:,:] > min_prob) * (std[1,:,:,:] < max_var)] = 1
            seg[(mean[2,:,:,:] > min_prob) * (std[2,:,:,:] < max_var)] = 2
            seg = seg.astype('int8')
            seg = (seg==1)*1
            # Fill holes in the segmented image
            seg = fill_holes(seg)
            # Remove small connected components from the segmented image
            seg = remove_small_comps_3d(seg)
            print(seg.shape)
            # Save the segmented image
            np.save(seg_file, seg)
