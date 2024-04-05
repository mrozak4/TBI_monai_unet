#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 19:49:36 2020

@author: rozakmat
"""

from skimage.morphology import skeletonize_3d, binary_closing, skeletonize, binary_erosion, binary_dilation, area_closing
from scipy.ndimage import label
from skimage import img_as_bool
from scipy.ndimage.morphology import distance_transform_edt
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import Counter
from tifffile import TiffFile
from tifffile.tifffile import imagej_description_metadata
import pickle
import numpy as np
from skimage import io
import re
from pathlib import Path
from skimage.transform import rescale
from skimage.measure import regionprops
import cv2
import cc3d
import sys, os
import warnings

from VascGraph.Skeletonize import Skeleton
from VascGraph.GraphIO import ReadStackMat
from VascGraph.GraphLab import StackPlot
from VascGraph.Tools.VisTools import visG
from VascGraph.Tools.CalcTools import fixG
from VascGraph.GraphIO import WritePajek
from skimage import io

def blockPrint():
    sys.stdout = open(os.devnull, 'w')

def enablePrint():
    sys.stdout = sys.__stdout__
    
warnings.filterwarnings("ignore")


speed_param=0.05
dist_param=0.5
med_param=0.5
sampling=1

degree_threshold=5.0 # used to check if a node is a skeletal node
clustering_r=1

#contraction
stop_param=0.001 # controls the convergence criterion
n_free_iteration=0 #number of iteration without checking for convergence

#refinement
area_param=50.0 # area of polygens to be decimated 
poly_param=10 # number of nodes forming a polygon    

#update with stiching
size=[507,507,252]
is_parallel=True
n_parallel=16
niter1=10 
niter2=5 

directory = Path('james_preds/')
files = sorted(list(directory.glob('*_seg.npy')))

files_list = [x.as_posix() for x in files]

for file in tqdm(files_list):
    img = np.load(file)
    sk=Skeleton(label=img, 
                speed_param=speed_param,
                dist_param=dist_param,
                med_param=med_param,
                sampling=sampling,
                degree_threshold=degree_threshold,
                clustering_resolution=clustering_r,
                stop_param=stop_param,
                n_free_iteration=n_free_iteration,
                area_param=area_param,
                poly_param=poly_param)
    
    sk.UpdateWithStitching(size=size,
                           niter1=niter1, 
                           niter2=niter2,
                           is_parallel=is_parallel, 
                           n_parallel=n_parallel)
    
    fullgraph=sk.GetOutput()
    WritePajek(path='', 
                name='james_graphs/' + 
                re.sub('_seg.npy','_v1.pajek',re.sub('james_preds/','',file)),#'mygraph.pajek', 
                graph=fixG(fullgraph))
    
    

