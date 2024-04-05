from skimage import io
import numpy as np
from pathlib import Path
from tqdm import tqdm
from scipy import ndimage
import re

directory = Path('james_preds')
files  = directory.glob('*_seg.tif')
files = sorted([x.as_posix() for x in files])

for file in tqdm(files):
    img = io.imread(file)
    img = np.abs(img-1)
    img_dst = ndimage.distance_transform_edt(img)
    io.imsave(re.sub('_seg.tif','_ev_dst.tif',re.sub('james_preds','james_ev_dst',file)),img_dst)
    img_dst = np.ndarray.flatten(img_dst)
    img_dst = img_dst[img_dst!=0]
    np.savetxt(re.sub('_seg.tif','.csv',re.sub('james_preds','james_ev_dst',file)),img_dst,delimiter=",")
