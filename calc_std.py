from numpy import load, save, std
from pathlib import Path
from re import sub
from os.path import exists
from numpy.random import shuffle
from tqdm import tqdm
import time
import argparse

path  = Path('/home/rozakmat/projects/rrg-bojana/rozakmat/TBI_monai_UNET/matt_raw_warped_single_upsampled')
files = list(path.glob('*_pred.npy'))
#files = [x.as_posix() for x in files if not os.path.exists(re.sub('scratch/rrg-bojana/rozakmat','projects/rrg-bojana/rozakmat/TBI_monai_UNET',re.sub('pred','std',x.as_posix())))]
files = sorted([x.as_posix() for x in files])
files = [x for x in files if not exists(sub('projects/rrg-bojana/rozakmat/TBI_monai_UNET','projects/rrg-bojana/rozakmat/TBI_monai_UNET',sub('pred','2x_std',x)))]
print(len(files))


# Define a function to save the standard deviation file
def save_std_file(file):
    """
    Save the standard deviation file.

    Args:
        file (str): The path to the file containing the predicted data.

    Returns:
        None
    """
    # Check if the standard deviation file already exists
    if not exists(sub('projects/rrg-bojana/rozakmat/TBI_monai_UNET','projects/rrg-bojana/rozakmat/TBI_monai_UNET',sub('pred','2x_std',file))):
        # Load the predicted data
        pred = load(file)
        # Calculate the standard deviation along the first axis
        _std = std(pred,axis=0)
        # Save the standard deviation file
        save(sub('projects/rrg-bojana/rozakmat/TBI_monai_UNET','projects/rrg-bojana/rozakmat/TBI_monai_UNET',sub('pred','2x_std',file)),_std)


# Shuffle the files list
shuffle(files)

# Save standard deviation files for remaining files until all files have been processed
while len(files) > 0:
    file = files.pop(0)
    save_std_file(file)
    print(len(files))