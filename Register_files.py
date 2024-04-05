import numpy as np
from skimage import io
from pathlib import Path
import re
import ants
from skimage.transform import resize
from tqdm import tqdm
from skimage.morphology import skeletonize_3d, binary_closing
from scipy.ndimage import distance_transform_edt, binary_dilation
import tifffile as tif
from scipy.ndimage import binary_fill_holes
import cc3d
from scipy.io import loadmat, savemat
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
from matplotlib.patches import Circle
from skimage.feature import peak_local_max
from statistics import mode
import imageio
from PIL import Image
from PIL.TiffTags import TAGS
from tifffile import TiffFile
from sklearn.metrics import normalized_mutual_info_score, adjusted_mutual_info_score
from registration import register_paired_images
from utilities import remove_small_comps_3d, fill_holes, _rotmat, closest_node
plt.rcParams['figure.figsize'] = [15, 15]

mouse_ids_path = Path('matt_raw_warped')#each mouse has its own folder with raw data in it
mouse_ids = list(mouse_ids_path.glob('*res*.tif'))#grab folder names/mouse ids
mouse_ids = sorted([x.as_posix() for x in mouse_ids])
data_dicts = [
    {"image":image_name}
    for image_name in mouse_ids
]

#data_dicts = [data_dicts[_i]]
re.sub('matt_raw_warped_upsampled','matt_preds_registered',data_dicts[0]["image"])

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
           'XYZres295',
           'XYZres457',
           'XYZres455']

xls = pd.ExcelFile('TBI_STIM_metalog_local.xlsx')
xls2 = pd.ExcelFile('../TBI_monai_UNET/p3_metalog.xlsx')
df = {}
for sheet_name in xls.sheet_names:
    df[sheet_name] = xls.parse(sheet_name)
    #print(sheet_name)
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
        bottoms = [addition + '/' + x for x in bottoms]
        bottoms = [x for x in bottoms if 'res' in x]
        tops_1 = df[key][df[key][df[key].columns[3]] == 0]
        tops_2 = df[key][df[key][df[key].columns[2]] == 0]
        tops = pd.concat((tops_1,tops_2))
        tops = np.array(tops[tops.columns[1]])
        tops = [addition + '/' + x for x in tops]
        tops = [x for x in tops if 'res' in x]
        if len(tops) > 1:
            dic[tops[0]] = list(tops[1:])
        elif len(tops) == 1:
            dic[tops[0]] = tops
        if len(bottoms) > 1:
            dic[bottoms[0]] = list(bottoms[1:])
        elif len(bottoms) == 1:
            dic[bottoms[0]] = bottoms
            
dic_2 = {'45/XYZres290':['45/XYZres296'],
         '45/XYZres297':['45/XYZres295'],
         '45/XYZres294':['45/XYZres298'],
         '45/XYZres288':['45/XYZres300'],
         '11/XYZres95':['11/XYZres98','11/XYZres102'],
         '11/XYZres92':[],
         '11/XYZres93':['11/XYZres93','11/XYZres96','11/XYZres97','11/XYZres100','11/XYZres101'],
         '11/XYZres91':['11/XYZres94','11/XYZres99'],
         '22/XYZres164':['22/XYZres165','22/XYZres168','22/XYZres169'],
         '22/XYZres160':['22/XYZres161'],
         '22/XYZres163':['22/XYZres166','22/XYZres167','22/XYZres170'],
         '22/XYZres159':['22/XYZres162'],
         '28/XYZres184':['28/XYZres185'],
         '28/XYZres188':['28/XYZres189','28/XYZres193','28/XYZres194'],
         '28/XYZres186':['28/XYZres187','28/XYZres188','28/XYZres190','28/XYZres191','28/XYZres192','28/XYZres195'],
         '28/XYZres183':[],
         '40/XYZres248':['40/XYZres249'],
         '40/XYZres245':['40/XYZres252'],
         '40/XYZres244':[],
         '40/XYZres243':['40/XYZres246','40/XYZres247','40/XYZres251'],
         '51/XYZres297':['51/XYZres298','51/XYZres302','51/XYZres305','51/XYZres301'],
         '51/XYZres296':['51/XYZres306'],
         '51/XYZres299':['51/XYZres300','51/XYZres303','51/XYZres304'],
         '65/XYZres397':['65/XYZres398','65/XYZres401','65/XYZres402','65/XYZres405','65/XYZres408','65/XYZres409'],
         '65/XYZres396':['65/XYZres399','65/XYZres400'],
         '65/XYZres403':[],
         '70/XYZres420':['70/XYZres421','70/XYZres422'],
         '70/XYZres419':['70/XYZres416','70/XYZres413','70/XYZres410','70/XYZres412','70/XYZres416'],
         '70/XYZres414':['70/XYZres417'],
         '70/XYZres411':['70/XYZres418','70/XYZres415'],
         'XYZres007':[]
        }
dic.update(dic_2) 

mouse_ids_path = Path('/home/rozakmat/projects/rrg-bojana/data/THY1-TBI')#each mouse has its own folder with raw data in it
mouse_ids = list(mouse_ids_path.glob('*?[0-9]/*res*?[0-9].tif'))#grab folder names/mouse ids
images = sorted([x.as_posix() for x in mouse_ids if '_0001' in x.as_posix()])
#images = [x for x in images if 'vbm' in x]
images = [x for x in images if  any(y in x for y in list(dic.keys()))]
#images = [x for x in images if any(y in x for y in ['14/','49/','56/','68/','65/','61/'])]
unused_keys = [x for x in list(dic.keys()) if not  any(x in y for y in images)]
print(len(images))
print(images[1])
new_file_name = re.sub('matt_raw_warped_upsampled','matt_preds_registered',data_dicts[0]["image"])
#images

np.random.shuffle(images)

res = []
for i in tqdm(range(len(images))[::-1]):
    fix_file = re.sub('_0001','',images[i])
    if not os.path.exists('matt_raw_warped_single/' + re.sub('.tif','_warped.tif',os.path.basename(os.path.dirname(fix_file)) + '-' + os.path.basename(fix_file))):
        key = [x for x in list(dic.keys()) if x in fix_file][0]
        mov_files = [re.sub(key,x,fix_file) for x in dic[key]]
        mov_files = [x for x in mov_files if os.path.exists(x)]
        mov_files = sorted(mov_files + [re.sub('.tif','_0001.tif',x) for x in mov_files])
        mov_files.append(re.sub('.tif','_0001.tif',fix_file))
        mov_files = [x for x in mov_files if x != fix_file]
        mov_files = sorted(mov_files)
        mov_files = np.unique(mov_files)
        print(fix_file)
        res.append(register_paired_images(fix_file, mov_files, 'matt_raw_warped_single/', sigma=2, flip=True))