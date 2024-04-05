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
from sklearn.metrics import normalized_mutual_info_score, adjusted_mutual_info_score

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


exclude = ['XYZres103',
           'XYZres104',
           'XYZres105',
           'XYZres105',
           'XYZres107',
           'XYZres133',
           'XYZres134',
           'XYZres135',
           'XYZres136',
           'XYZres137',
           'XYZres138',
           'XYZres139',
           'XYZres140',
           'XYZres183',
           'XYZres196',
           'XYZres197',
           'XYZres260',
           'XYZres288',
           'XYZres343',
           'XYZres340',
           'XYZres341',
           'XYZres250',
           'XYZres297',
           'XYZres295']

xls = pd.ExcelFile('TBI_STIM_metalog_local.xlsx')
xls2 = pd.ExcelFile('../TBI_monai_UNET/p3_metalog.xlsx')
df = {}
for sheet_name in xls.sheet_names:
    df[sheet_name] = xls.parse(sheet_name)
for sheet_name in xls2.sheet_names:
    df[sheet_name] = xls2.parse(sheet_name)

dic = {}
for key in df.keys():
    if '3D' in key and ('vbm01' not in key and 'vbm02' not in key and 'SHAM7_3D' not in key and 'TBI45_3D' not in key and 'TBI11_3D' not in key and 'TBI65_3D' not in key and 'TBI22_3D' not in key and 'TBI28_3D' not in key and 'TBI40_3D' not in key and 'TBI51_3D' not in key and 'TBI70_3D' not in key):
        if 'vbm' not in key:
            addition = re.sub('C57','',re.sub('TBI','',re.sub('SHAM','',re.sub('_3D','',key))))
        else:
            addition = ''
        df[key] = df[key][~df[key][df[key].columns[1]].isin(exclude)]
        scans = np.array(df[key][df[key].columns[1]])
        scans = [x for x in scans if 'res' in str(x)]
        bottoms_1 = df[key][df[key][df[key].columns[3]] == 500]
        bottoms_2 = df[key][df[key][df[key].columns[2]] == 500]
        bottoms = pd.concat((bottoms_1,bottoms_2))
        bottoms = np.array(bottoms[bottoms.columns[1]])
        bottoms = [addition + '-' + x for x in bottoms]
        bottoms = [x for x in bottoms if 'res' in x]
        tops_1 = df[key][df[key][df[key].columns[3]] == 0]
        tops_2 = df[key][df[key][df[key].columns[2]] == 0]
        tops = pd.concat((tops_1,tops_2))
        tops = np.array(tops[tops.columns[1]])
        tops = [addition + '-' + x for x in tops]
        tops = [x for x in tops if 'res' in x]
        if len(tops) > 1:
            dic[tops[0]] = list(tops[1:])
        elif len(tops) == 1:
            dic[tops[0]] = tops
        if len(bottoms) > 1:
            dic[bottoms[0]] = list(bottoms[1:])
        elif len(bottoms) == 1:
            dic[bottoms[0]] = bottoms
dic_2 = {'45-XYZres290':['45-XYZres296'],
         '45-XYZres297':['45-XYZres295'],
         '45-XYZres294':['45-XYZres298'],
         '45-XYZres288':['45-XYZres300'],
         '11-XYZres95':['11-XYZres98','11-XYZres102'],
         '11-XYZres92':[],
         '11-XYZres93':['11-XYZres93','11-XYZres96','11-XYZres97','11-XYZres100','11-XYZres101'],
         '11-XYZres91':['11-XYZres94','11-XYZres99'],
         '22-XYZres164':['22-XYZres165','22-XYZres168','22-XYZres169'],
         '22-XYZres160':['22-XYZres161'],
         '22-XYZres163':['22-XYZres166','22-XYZres167','22-XYZres170'],
         '22-XYZres159':['22-XYZres162'],
         '28-XYZres184':['28-XYZres185'],
         '28-XYZres188':['28-XYZres189','28-XYZres193','28-XYZres194'],
         '28-XYZres186':['28-XYZres187','28-XYZres188','28-XYZres190','28-XYZres191','28-XYZres192','28-XYZres195'],
         '28-XYZres183':[],
         '40-XYZres248':['40-XYZres249'],
         '40-XYZres245':['40-XYZres252'],
         '40-XYZres244':[],
         '40-XYZres243':['40-XYZres246','40-XYZres247','40-XYZres251'],
         '51-XYZres297':['51-XYZres298','51-XYZres302','51-XYZres305','51-XYZres301'],
         '51-XYZres296':['51-XYZres306'],
         '51-XYZres299':['51-XYZres300','51-XYZres303','51-XYZres304'],
         '65-XYZres397':['65-XYZres398','65-XYZres401','65-XYZres402','65-XYZres405','65-XYZres408','65-XYZres409'],
         '65-XYZres396':['65-XYZres399','65-XYZres400'],
         '65-XYZres403':[],
         '70-XYZres420':['70-XYZres421','70-XYZres422'],
         '70-XYZres419':['70-XYZres416','70-XYZres413','70-XYZres410','70-XYZres412','70-XYZres416'],
         '70-XYZres414':['70-XYZres417'],
         '70-XYZres411':['70-XYZres418','70-XYZres415'],
         '-XYZres007':[]
        }
dic.update(dic_2) 
len(list(dic.keys()))



directory = Path('matt_raw_warped_single_upsampled_seg')
files_seg_0001 = directory.glob('*_0001_warped_seg.npy')
files_seg_0001 = sorted([x.as_posix() for x in files_seg_0001])
files_seg_0001 = [x for x in files_seg_0001 if  any(y in x for y in list(dic.keys()))]
#files_seg_0001 = [re.sub('_0001','',x) for x in files_seg_0001]
np.random.shuffle(files_seg_0001)
len(files_seg_0001)

parser = argparse.ArgumentParser(description='take hyperparameter inputs')

parser.add_argument('-c','--cfg', type=int, dest='cfg', action='store')

args = parser.parse_args()

file = args.cfg

i = file

print(i)

file_0001 = files_seg_0001[i]

min_prob = 0.75
max_var = 0.1

np.random.shuffle(files_seg_0001)
for file_0001 in tqdm(files_seg_0001[:]):
    if not os.path.exists(re.sub('_0001_warped_seg.npy','_warped.pickle',file_0001)) or (time.time() - os.path.getmtime(re.sub('_0001_warped_seg.npy','_warped.pickle',file_0001)))/3600>48:
        if os.path.exists(re.sub('_0001_warped_seg.npy','_skel_warped_single.mat',file_0001)):
            file = file_0001
            skel_file = re.sub('_0001_warped_seg.npy','_skel_warped_single.mat',file)
            skel = loadmat(skel_file)['FilteredImage']
            if np.sum(skel) != 0:
                graph = sknw.build_sknw(skel, multi=False)
                print(len(graph.edges))
                while len(list(nx.selfloop_edges(graph)))>0:
                    if len(list(nx.selfloop_edges(graph))) !=0:
                        for edge in list(nx.selfloop_edges(graph)):
                            skel[graph[edge[0]][edge[1]]['pts'][1:-1,0],graph[edge[0]][edge[1]]['pts'][1:-1,1],graph[edge[0]][edge[1]]['pts'][1:-1,2]] = 0
                    for edge in graph.edges:
                        if (graph.degree(edge[0]) == 1 and graph.degree(edge[1]) > 2) or (graph.degree(edge[1]) == 1 and graph.degree(edge[0]) > 2):
                            if graph[edge[0]][edge[1]]['weight']<20:
                                skel[graph[edge[0]][edge[1]]['pts'][1:-1,0],graph[edge[0]][edge[1]]['pts'][1:-1,1],graph[edge[0]][edge[1]]['pts'][1:-1,2]] = 0
                    graph = sknw.build_sknw(skel, multi=False)
                for edge in graph.edges:
                    if (graph.degree(edge[0]) == 1 and graph.degree(edge[1]) > 2) or (graph.degree(edge[1]) == 1 and graph.degree(edge[0]) > 2):
                        if graph[edge[0]][edge[1]]['weight']<20:
                            skel[graph[edge[0]][edge[1]]['pts'][1:-1,0],graph[edge[0]][edge[1]]['pts'][1:-1,1],graph[edge[0]][edge[1]]['pts'][1:-1,2]] = 0
                graph = sknw.build_sknw(skel, multi=False)
                graph.remove_nodes_from(list(nx.isolates(graph)))
                print(len(graph.edges))
                #print(len(list(nx.isolates(graph))))
                io.imsave(re.sub('_0001_warped_seg.npy','_single_skel.tif',file),skel)
                nx.write_gpickle(graph,re.sub('_warped_seg.npy','_warped.pickle',file))
                key = [x for x in list(dic.keys()) if x in file][0]
                mov_files = [re.sub(key,x,file) for x in dic[key]]
                mov_files = sorted(mov_files + [re.sub('_0001','',x) for x in mov_files])
                mov_files.append(re.sub('_0001','',file))
                seg = np.load(re.sub('_0001','',file))
                for _file in mov_files:
                    if os.path.exists(_file):
                        seg_0001 = np.load(_file)
                        _mut_info = normalized_mutual_info_score(seg.flatten(),seg_0001.flatten()) # calculate mutual information between masks, judges registration
                        if _mut_info > 0.1:
                            print('matched')
                            nx.write_gpickle(graph,re.sub('_warped_seg.npy','_warped.pickle',_file))