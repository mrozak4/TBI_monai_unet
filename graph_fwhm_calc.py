#from nipype.interfaces import niftyreg
import numpy as np
from skimage import io
from pathlib import Path
import re
import ants
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
import argparse
import scipy as sp
import vg
from pytransform3d.rotations import matrix_from_axis_angle
import multiprocessing
from scipy.ndimage import convolve as conv
from scipy.stats import multivariate_normal
from skimage import color, data, restoration
from RedLionfishDeconv import doRLDeconvolutionFromNpArrays

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

def twoD_Gaussian(xy, amplitude, sigma, x0, y0, offset):
    theta=0
    x, y = xy   
    a = (np.cos(theta)**2)/(2*sigma**2) + (np.sin(theta)**2)/(2*sigma**2)
    b = -(np.sin(2*theta))/(4*sigma**2) + (np.sin(2*theta))/(4*sigma**2)
    c = (np.sin(theta)**2)/(2*sigma**2) + (np.cos(theta)**2)/(2*sigma**2)
    g = offset + amplitude*np.exp( - (a*((x-x0)**2) + 2*b*(x-x0)*(y-y0) + c*((y-y0)**2)))
    return g.ravel()

directory = Path('matt_preds')
files = directory.glob('*_warped.pickle')
files = sorted([x.as_posix() for x in files])
files = [x for x in files if '-' in x]
files = [x for x in files if not os.path.exists(re.sub('preds','preds_graphs',re.sub('.pickle','_radii.pickle',x)))]
np.random.shuffle(files)

parser = argparse.ArgumentParser(description='take hyperparameter inputs')

parser.add_argument('-c','--cfg', type=int, dest='cfg', action='store')

args = parser.parse_args()

file = args.cfg

i = file

print(i)
print(len(files))


min_prob = 0.75
max_var = 0.1
sampling = 1/5
xls = pd.ExcelFile('TBI_STIM_metalog_local.xlsx')
xls2 = pd.ExcelFile('../TBI_monai_UNET/p3_metalog.xlsx')
df = {}
for sheet_name in xls.sheet_names:
    df[sheet_name] = xls.parse(sheet_name)
for sheet_name in xls2.sheet_names:
    df[sheet_name] = xls2.parse(sheet_name)

for file in tqdm(files):
    if not os.path.exists(re.sub('preds','preds_graphs',re.sub('.pickle','_radii.pickle',file))):
        graph = nx.read_gpickle(file)
        if len(graph.edges) < 1500:
            for sheet_name in xls.sheet_names + xls2.sheet_names:
                if re.sub('matt_preds_registered/','',re.sub('_warped.pickle','',re.sub('xyz','XYZ',file))).split('-')[1] in df[sheet_name].values:
                    subj = sheet_name
                    if (re.sub('matt_raw_warped_upsampled_seg/','',re.sub('_warped.pickle','',file)).split('-')[0].split('_') + [''])[1] in sheet_name or re.sub('matt_preds/','',re.sub('_warped.pickle','',file)).split('-')[0].split(' ')[0] in sheet_name:
                        if subj in ["TBI07_3D",
                                    "TBI11_3D",
                                    "TBI22_3D",
                                    "TBI31_3D",
                                    "TBI38_3D",
                                    "SHAM09_3D",
                                    "SHAM12_3D",
                                    "SHAM23_3D",
                                    "SHAM32_3D",
                                    'TBI38_3D',
                                    'TBI6_3D',
                                    'SHAM7_3D',
                                    'TBI43_3D',
                                    'TBI45_3D',
                                    'SHAM53_3D',
                                    'SHAM56_3D',
                                    'SHAM55_3D']:
                            gender = 'male'
                        else:
                            gender = 'female'
                        treatment = re.sub('SHA','SHAM',subj[0:3])
                        _tmp = df[subj].loc[df[subj]['CHECK WATER'] == re.sub('matt_preds/','',re.sub('_warped.pickle','',re.sub('xyz','XYZ',file))).split('-')[1]]
                        if _tmp['Unnamed: 12'].iloc[0] == 'raster':
                            img_file = re.sub('matt_raw_warped_upsampled_seg','matt_raw_warped',re.sub('_warped.pickle','_warped.tif',file))
                            img_0001_file = re.sub('matt_raw_warped_upsampled_seg','matt_raw_warped',re.sub('_warped.pickle','_0001_warped.tif',file))
                            seg_file = re.sub('_warped.pickle','_warped_seg.npy',file)
                            seg_0001_file = re.sub('_warped.pickle','_0001_warped_seg.npy',file)
                            mean_file = re.sub('_upsampled_seg','_upsampled',re.sub('_warped.pickle','_warped_mean.npy',file))
                            mean_0001_file = re.sub('_upsampled_seg','_upsampled',re.sub('_warped.pickle','_0001_warped_mean.npy',file))
                            std_file = re.sub('_upsampled_seg','_upsampled',re.sub('_warped.pickle','_warped_2x_std.npy',file))
                            std_0001_file = re.sub('_upsampled_seg','_upsampled',re.sub('_warped.pickle','_0001_warped_2x_std.npy',file))
                            img = io.imread(img_file)
                            img_ch2 = sp.ndimage.zoom(np.swapaxes(img[:,1,:,:],0,2),(1,1,2.645833333))
                            img = sp.ndimage.zoom(np.swapaxes(img[:,0,:,:],0,2),(1,1,2.645833333))
                            img_0001 = io.imread(img_0001_file)
                            img_0001_ch2 = sp.ndimage.zoom(np.swapaxes(img_0001[:,1,:,:],0,2),(1,1,2.645833333))
                            img_0001 = sp.ndimage.zoom(np.swapaxes(img_0001[:,0,:,:],0,2),(1,1,2.645833333))
                            seg = np.load(seg_file)
                            seg_0001 = np.load(seg_0001_file)
                            mean = np.load(mean_file)
                            mean_0001 = np.load(mean_0001_file)
                            std = np.load(std_file)
                            std_0001 = np.load(std_0001_file)
                            seg_dst = distance_transform_edt(seg)
                            seg_0001_dst = distance_transform_edt(seg_0001)
                            nrn_dst = np.load(re.sub('_warped.pickle','_warped_seg_nrn_dst.npy',file))
                            wavelength = _tmp['Unnamed: 11'].iloc[0]
                            power_per = _tmp['Unnamed: 10'].iloc[0]
                            start_depth = _tmp['Unnamed: 2'].iloc[0]
                            
                            a, b, c = np.mgrid[-15:16:1, -15:16:1, -15:16:1]
                            abc = np.dstack([a.flat,b.flat, c.flat])
                            mu = np.array([0,0,0])
                            sigma = np.array([0.636,0.127,0.127])
                            covariance = np.diag(sigma**2)
                            d = multivariate_normal.pdf(abc, mean=mu, cov=covariance)
                            d = d.reshape((len(a),len(b),len(c)))
                            deconv_img = np.copy(img)
                            deconv_img =  1023 * restoration.richardson_lucy(img/1023.0, d, iterations=10) - 1023 * restoration.richardson_lucy(img_ch2/1023.0, d, iterations=10)
                            #deconv_img[1] =  1023 * restoration.richardson_lucy(img[1]/1023.0, d, iterations=10)
                            deconv_img = np.int16(deconv_img)
                            deconv_img_0001 = np.copy(img_0001)
                            deconv_img_0001 =  1023 * restoration.richardson_lucy(img_0001/1023.0, d, iterations=10) - 1023 * restoration.richardson_lucy(img_0001_ch2/1023.0, d, iterations=10)
                            #deconv_img_0001[1] =  1023 * restoration.richardson_lucy(img_0001[1]/1023.0, d, iterations=10)
                            deconv_img_0001 = np.int16(deconv_img_0001)
                            
                            if treatment != 'vbm':
                                age = _tmp['Unnamed: 14'].iloc[0]
                                days_post_injury = _tmp['Unnamed: 15'].iloc[0]
                            else:
                                age = 'unknown'
                                days_post_injury = 'unknown'
                            
                            for i in tqdm(range(len(graph.edges))):
                                path = graph[list(graph.edges)[i][0]][list(graph.edges)[i][1]]['pts']
                                _pred_radii = np.mean(seg_dst[path[::-1,0],path[::-1,1],path[::-1,2]])
                                _pred_radii_max = np.max(seg_dst[path[::-1,0],path[::-1,1],path[::-1,2]])
                                if _pred_radii == 0:
                                    _pred_radii =1
                                _pred_radii_0001 = np.mean(seg_0001_dst[path[::-1,0],path[::-1,1],path[::-1,2]])
                                _pred_radii_max_0001 = np.max(seg_0001_dst[path[::-1,0],path[::-1,1],path[::-1,2]])
                                if _pred_radii_0001 == 0:
                                    _pred_radii_0001 =1
                                
                                _box_fit = max([np.int16(_pred_radii_max)+10, np.int16(_pred_radii_max_0001)+10, 10])
                                #path_grad = np.gradient(path,edge_order=2)[0]
                                path_smooth = np.float32(np.copy(path))
                                for k in range(len(path[0])):
                                    path_smooth[:,k] = sp.ndimage.gaussian_filter1d(np.float64(path[:,k]),3,mode='nearest')
                                path_grad = np.gradient(path_smooth,edge_order=2)[0]
                                res_fwhm = []
                                res_fwhm_0001 = []
                                X = np.arange(-1*_box_fit,_box_fit+1,1)
                                Y = np.arange(-1*_box_fit,_box_fit+1,1)
                                x,y = np.meshgrid(X,Y)
                                x = x.flatten()
                                y = y.flatten()
                                z = np.zeros(len(x))
                                xy = np.vstack([x,y,z])
                                
                                
                                res_fwhm = []
                                res_fwhm_0001 = []
                                
                                res_fwhm_sigma = []
                                res_fwhm_sigma_0001 = []
                                
                                def calc_fwhm_path(I):
                                    point_grad = path_grad[I]
                                    point = path[I]
                                    if all(point_grad[0:2] == [0,0]) and abs(point_grad[2]/point_grad[2]) == 1:
                                        rotated = xy.T + point
                                    else:
                                        rotated = _rotmat(point_grad,xy.T) + point
                                    points_img = sp.ndimage.map_coordinates(#np.int16(mean[1]*1024),
                                                                        deconv_img,
                                                                        rotated.T, 
                                                                        order=3,
                                                                        mode='constant')
                                    points_img_0001 = sp.ndimage.map_coordinates(#np.int16(mean_0001[1]*1024),
                                                                             deconv_img_0001,
                                                                             rotated.T, 
                                                                             order=3,
                                                                             mode='constant')
                                            
                                    points_img = np.reshape(points_img,(len(X),len(Y)))
                                    points_img_no_smooth = np.copy(points_img)
                                    points_img = sp.ndimage.gaussian_filter(points_img, sigma = np.min([_pred_radii,_pred_radii_0001])*.4)

                                    points_img_0001 = np.reshape(points_img_0001,(len(X),len(Y)))
                                    points_img_0001_no_smooth = np.copy(points_img_0001)
                                    points_img_0001 = sp.ndimage.gaussian_filter(points_img_0001, sigma = np.min([_pred_radii,_pred_radii_0001])*.4)
                                    
                                    _point = np.array(np.arange(0,np.max([_pred_radii,_pred_radii_0001])+20,sampling))
                                    _zeros = np.zeros(len(_point))
                                    _point = np.array([_point,_zeros])
                                    _centre = closest_node([len(X)//2+1,len(Y)//2+1],peak_local_max(points_img.T))
                                    _centre_0001 = closest_node([len(X)//2+1,len(Y)//2+1],peak_local_max(points_img_0001.T))
                                    
                                    _res = []
                                    
                                    fig,ax = plt.subplots(1)
                                    ax.set_aspect('equal')
                                    ax.imshow(points_img_no_smooth,
                                              vmin = 0,
                                              vmax = np.max(deconv_img))
                                    ax.scatter(_centre[0],_centre[1])
                                    for deg in np.arange(0,360,10):
                                        rot_point = np.dot(np.array([[np.cos(np.deg2rad(deg)),-1*np.sin(np.deg2rad(deg))],[np.sin(np.deg2rad(deg)),np.cos(np.deg2rad(deg))]]),_point)
                                        rot_point[0] = rot_point[0] + _centre[0]
                                        rot_point[1] = rot_point[1] + _centre[1]
                                        points_vals = sp.ndimage.map_coordinates(points_img.T,
                                                                                 rot_point, 
                                                                                 order=3,
                                                                                 cval=0)
                                        points_vals = sp.ndimage.gaussian_filter1d(points_vals,sigma=np.min([_pred_radii,_pred_radii_0001])*.4/sampling)
                                        points_vals_grad = np.gradient(points_vals)
                                        _ = np.array(sp.signal.argrelextrema(points_vals + np.random.uniform(-1e-5,1e-5,len(points_vals)),np.less,order=3))
                                        if _.shape[1] != 0:
                                            points_vals_grad = np.gradient(points_vals[:_[0,0]+3])
                                            _ = np.argmin(points_vals_grad)
                                            _res.append(_*sampling)
                                            ax.scatter(_*sampling*np.cos(np.deg2rad(deg))+_centre[0],_*sampling*np.sin(np.deg2rad(deg))+_centre[1],color='r')
                                        else:
                                            points_vals_grad = np.gradient(points_vals)
                                            _ = np.argmin(points_vals_grad)
                                            _res.append(_*sampling)
                                            ax.scatter(_*sampling*np.cos(np.deg2rad(deg))+_centre[0],_*sampling*np.sin(np.deg2rad(deg))+_centre[1],color='r')
                                    _res = np.array(_res)
                                    _res = _res[np.where(_res!=0)]
                                    _mean = np.mean(_res)
                                    _std = np.std(_res)
                                    _mask = np.where(np.logical_and(_res>_mean-2*_std, _res<_mean+2*_std))
                                    _res = _res[_mask]
                                    radii = np.mean(_res)
                                    radii_std = np.std(_res)
                                    circ = Circle(_centre,radii,fill=False)
                                    ax.add_patch(circ)
                                    ax.set_title(radii)
                                    fig.savefig('/home/rozakmat/scratch/_tmp/'+re.sub('matt_raw_warped_upsampled_seg/','',re.sub('_warped.pickle','',file))+str(i)+'_'+str(I)+'.png')
                                    plt.clf()
                                    _res_0001 = []
                                    fig,ax = plt.subplots(1)
                                    ax.set_aspect('equal')
                                    ax.imshow(points_img_0001_no_smooth,
                                              vmin = 0,
                                              vmax = np.max(deconv_img_0001))
                                    ax.scatter(_centre_0001[0],_centre_0001[1])
                                    for deg in np.arange(0,360,10):
                                        rot_point = np.dot(np.array([[np.cos(np.deg2rad(deg)),-1*np.sin(np.deg2rad(deg))],[np.sin(np.deg2rad(deg)),np.cos(np.deg2rad(deg))]]),_point)
                                        rot_point[0] = rot_point[0] + _centre_0001[0]
                                        rot_point[1] = rot_point[1] + _centre_0001[1]
                                        points_vals = sp.ndimage.map_coordinates(points_img_0001.T,
                                                                                 rot_point, 
                                                                                 order=3,
                                                                                 cval=0)
                                        points_vals = sp.ndimage.gaussian_filter1d(points_vals,sigma=np.min([_pred_radii,_pred_radii_0001])*.4/sampling)
                                        points_vals_grad = np.gradient(points_vals)
                                        _ = np.array(sp.signal.argrelextrema(points_vals + np.random.uniform(-1e-5,1e-5,len(points_vals)),np.less,order=3))
                                        if _.shape[1] != 0:
                                            points_vals_grad = np.gradient(points_vals[:_[0,0]+3])
                                            _ = np.argmin(points_vals_grad)
                                            if _ != 0:
                                                _res_0001.append(_*sampling)
                                                ax.scatter(_*sampling*np.cos(np.deg2rad(deg))+_centre_0001[0],_*sampling*np.sin(np.deg2rad(deg))+_centre_0001[1],color='r')
                                        else:
                                            points_vals_grad = np.gradient(points_vals)
                                            _ = np.argmin(points_vals_grad)
                                            if _ != 0:
                                                _res_0001.append(_*sampling)
                                                ax.scatter(_*sampling*np.cos(np.deg2rad(deg))+_centre_0001[0],_*sampling*np.sin(np.deg2rad(deg))+_centre_0001[1],color='r')
                                    _res_0001 = np.array(_res_0001)
                                    _res_0001 = _res_0001[np.where(_res_0001!=0)]
                                    _mean_0001 = np.mean(_res_0001)
                                    _std_0001 = np.std(_res_0001)
                                    _mask = np.where(np.logical_and(_res_0001>_mean_0001-2*_std_0001, _res_0001<_mean_0001+2*_std_0001))
                                    _res_0001 = _res_0001[_mask]
                                    radii_0001 = np.mean(_res_0001)
                                    radii_0001_std = np.std(_res_0001)
                                    circ = Circle(_centre_0001,radii_0001,fill=False)
                                    ax.add_patch(circ)
                                    ax.set_title(radii_0001)
                                    fig.savefig('/home/rozakmat/scratch/_tmp/'+re.sub('matt_raw_warped_upsampled_seg/','',re.sub('_warped.pickle','',file))+str(i)+'_'+str(I)+'_0001.png')
                                    plt.clf()
                                        
                                    return radii, radii_0001, radii_std, radii_0001_std
                                
                                pool = multiprocessing.Pool(8)
                                _vals, _vals_0001, _vals_sigma, _vals_sigma_0001 = zip(*pool.map(calc_fwhm_path, range(len(path))))
                                
                                images = []
                                for I in tqdm(range(len(path))):
                                    images.append(imageio.imread('/home/rozakmat/scratch/_tmp/'+re.sub('matt_raw_warped_upsampled_seg/','',re.sub('_warped.pickle','',file))+str(i)+'_'+str(I)+'_0001.png'))
                                    os.remove('/home/rozakmat/scratch/_tmp/'+re.sub('matt_raw_warped_upsampled_seg/','',re.sub('_warped.pickle','',file))+str(i)+'_'+str(I)+'_0001.png')
                                imageio.mimsave('/home/rozakmat/scratch/gifs/'+re.sub('matt_raw_warped_upsampled_seg/','',re.sub('_warped.pickle','',file))+str(i)+'_0001.gif', images, format='GIF', fps=2)
                                images = []
                                for I in tqdm(range(len(path))):
                                    images.append(imageio.imread('/home/rozakmat/scratch/_tmp/'+re.sub('matt_raw_warped_upsampled_seg/','',re.sub('_warped.pickle','',file))+str(i)+'_'+str(I)+'.png'))
                                    os.remove('/home/rozakmat/scratch/_tmp/'+re.sub('matt_raw_warped_upsampled_seg/','',re.sub('_warped.pickle','',file))+str(i)+'_'+str(I)+'.png')
                                imageio.mimsave('/home/rozakmat/scratch/gifs/'+re.sub('matt_raw_warped_upsampled_seg/','',re.sub('_warped.pickle','',file))+str(i)+'.gif', images, format='GIF', fps=2)
                                
                                _nrn_dst_vals = nrn_dst[path[::-1,0],path[::-1,1],path[::-1,2]]
                                graph[list(graph.edges)[i][0]][list(graph.edges)[i][1]]['radii'] = np.mean(_vals)
                                graph[list(graph.edges)[i][0]][list(graph.edges)[i][1]]['radii_std'] = np.std(_vals)
                                graph[list(graph.edges)[i][0]][list(graph.edges)[i][1]]['radii_0001'] = np.mean(_vals_0001)
                                graph[list(graph.edges)[i][0]][list(graph.edges)[i][1]]['radii_0001_std'] = np.std(_vals_0001)
                                graph[list(graph.edges)[i][0]][list(graph.edges)[i][1]]['mean_neuron_distance'] = np.mean(_nrn_dst_vals)
                                graph[list(graph.edges)[i][0]][list(graph.edges)[i][1]]['neuron_distance_std'] = np.std(_nrn_dst_vals)
                                graph[list(graph.edges)[i][0]][list(graph.edges)[i][1]]['neuron_distance_min'] = np.min(_nrn_dst_vals)
                                graph[list(graph.edges)[i][0]][list(graph.edges)[i][1]]['delta'] = np.mean(_vals_0001) - np.mean(_vals)
                                graph[list(graph.edges)[i][0]][list(graph.edges)[i][1]]['gender'] = gender
                                graph[list(graph.edges)[i][0]][list(graph.edges)[i][1]]['path_weights'] = _vals
                                graph[list(graph.edges)[i][0]][list(graph.edges)[i][1]]['path_weights_uncertanty'] = _vals_sigma
                                graph[list(graph.edges)[i][0]][list(graph.edges)[i][1]]['path_weights_0001'] = _vals_0001
                                graph[list(graph.edges)[i][0]][list(graph.edges)[i][1]]['path_weights_uncertanty_0001'] = _vals_sigma_0001
                                #graph[list(graph.edges)[i][0]][list(graph.edges)[i][1]]['weight'] = graph[list(graph.edges)[i][0]][list(graph.edges)[i][1]]['weight']
                                graph[list(graph.edges)[i][0]][list(graph.edges)[i][1]]['end-0z'] = path[0][0]
                                graph[list(graph.edges)[i][0]][list(graph.edges)[i][1]]['end-0y'] = path[0][1]
                                graph[list(graph.edges)[i][0]][list(graph.edges)[i][1]]['end-0x'] = path[0][2]
                                graph[list(graph.edges)[i][0]][list(graph.edges)[i][1]]['end-1z'] = path[-1][0]
                                graph[list(graph.edges)[i][0]][list(graph.edges)[i][1]]['end-1y'] = path[-1][1]
                                graph[list(graph.edges)[i][0]][list(graph.edges)[i][1]]['end-1x'] = path[-1][2]
                                graph[list(graph.edges)[i][0]][list(graph.edges)[i][1]]['img_start_depth'] = start_depth
                                graph[list(graph.edges)[i][0]][list(graph.edges)[i][1]]['mean_depth'] = np.mean(path[:,0])
                                graph[list(graph.edges)[i][0]][list(graph.edges)[i][1]]['euclidean-dst'] = np.sqrt(np.sum(np.square(path[-1]-path[0])))
                                graph[list(graph.edges)[i][0]][list(graph.edges)[i][1]]['subject'] = subj
                                graph[list(graph.edges)[i][0]][list(graph.edges)[i][1]]['treatment'] = treatment
                                graph[list(graph.edges)[i][0]][list(graph.edges)[i][1]]['wavelength'] = wavelength
                                graph[list(graph.edges)[i][0]][list(graph.edges)[i][1]]['power'] = power_per
                                graph[list(graph.edges)[i][0]][list(graph.edges)[i][1]]['age'] = age
                                graph[list(graph.edges)[i][0]][list(graph.edges)[i][1]]['days_post_injury'] = days_post_injury
                            nx.write_gpickle(graph, re.sub('preds','preds_graphs_fwhm',re.sub('.pickle','_radii.pickle',file)))

del pool