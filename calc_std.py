import numpy as np
from pathlib import Path
import re
import os 
from tqdm import tqdm
import time
import argparse

path  = Path('/home/rozakmat/projects/rrg-bojana/rozakmat/TBI_monai_UNET/matt_raw_warped_single_upsampled')
files = list(path.glob('*_pred.npy'))
#files = [x.as_posix() for x in files if not os.path.exists(re.sub('scratch/rrg-bojana/rozakmat','projects/rrg-bojana/rozakmat/TBI_monai_UNET',re.sub('pred','std',x.as_posix())))]
files = sorted([x.as_posix() for x in files])
files = [x for x in files if not os.path.exists(re.sub('projects/rrg-bojana/rozakmat/TBI_monai_UNET','projects/rrg-bojana/rozakmat/TBI_monai_UNET',re.sub('pred','2x_std',x)))]
print(len(files))

parser = argparse.ArgumentParser(description='take hyperparameter inputs')

parser.add_argument('-c','--cfg', type=int, dest='cfg', action='store')

args = parser.parse_args()

file = args.cfg

i = file

print(i)

if not os.path.exists(re.sub('projects/rrg-bojana/rozakmat/TBI_monai_UNET','projects/rrg-bojana/rozakmat/TBI_monai_UNET',re.sub('pred','2x_std',files[i]))):
    pred = np.load(files[i])
    _std = np.std(pred,axis=0)
    np.save(re.sub('projects/rrg-bojana/rozakmat/TBI_monai_UNET','projects/rrg-bojana/rozakmat/TBI_monai_UNET',re.sub('pred','2x_std',files[i])),_std)

np.random.shuffle(files)
i=0
while len(files) > 0:
    if not os.path.exists(re.sub('projects/rrg-bojana/rozakmat/TBI_monai_UNET','projects/rrg-bojana/rozakmat/TBI_monai_UNET',re.sub('pred','2x_std',files[i]))):
        pred = np.load(files[i])
        _std = np.std(pred,axis=0)
        np.save(re.sub('projects/rrg-bojana/rozakmat/TBI_monai_UNET','projects/rrg-bojana/rozakmat/TBI_monai_UNET',re.sub('pred','2x_std',files[i])),_std)
        files = list(path.glob('*_pred.npy'))
        files = sorted([x.as_posix() for x in files])
        files = [x for x in files if not os.path.exists(re.sub('projects/rrg-bojana/rozakmat/TBI_monai_UNET','projects/rrg-bojana/rozakmat/TBI_monai_UNET',re.sub('pred','2x_std',x)))]
        np.random.shuffle(files)