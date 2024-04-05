from monai.utils import (
    first, 
    set_determinism, 
    ensure_tuple
)
from monai.transforms import (
    AsDiscrete,
    AsDiscreted,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    RandAffined,
    RandRotate90d,
    RandShiftIntensityd,
    RandFlipd,
    RandGaussianNoised,
    RandAdjustContrastd,
    ScaleIntensityRanged,
    Spacingd,
    EnsureTyped,
    EnsureType,
    Invertd,
    AddChanneld,
    RandGaussianSharpend,
    RandGaussianSmoothd,
    RandHistogramShiftd,
    OneOf,
    Rand3DElasticd,
    Rand3DElastic,
    RandGridDistortiond,
    RandSpatialCropSamplesd,
    FillHoles,
    LabelFilter,
    LabelToContour
)
from monai.handlers.utils import from_engine
from monai.networks.nets import UNETR
from monai.networks.layers import Norm
from monai.metrics import (
    DiceMetric, 
    HausdorffDistanceMetric
)
from monai.losses import (
    DiceLoss, 
    DiceCELoss, 
    DiceFocalLoss
)
from monai.inferers import sliding_window_inference
from monai.data import (
    CacheDataset, 
    DataLoader, 
    Dataset, 
    decollate_batch, 
    ImageReader
)
from monai.data.image_reader import WSIReader
from monai.config import (
    print_config, 
    KeysCollection, 
    PathLike
)
from monai.apps import download_and_extract
import torch
from torchio.transforms import (
    RandomAffine
)
import matplotlib.pyplot as plt
import tempfile
import shutil
import os
import glob
from numpy import random
from pathlib import Path
import re
from skimage import io
from typing import (
    Optional, 
    Union, 
    Sequence, 
    Callable, 
    Dict, 
    List
)
from monai.data.utils import is_supported_format
from monai. data.image_reader import (
    _copy_compatible_dict, 
    _stack_images
)
from nibabel.nifti1 import Nifti1Image
from PIL import Image
import numpy as np
from tqdm import tqdm
import pickle
import pandas as pd
from mlflow import log_metric, log_param, log_artifacts, set_experiment, start_run, end_run
import warnings
warnings.filterwarnings('ignore')


class TIFFReader(ImageReader):
    
    def __init__(self, npz_keys: Optional[KeysCollection] = None, channel_dim: Optional[int] = None, **kwargs):
        super().__init__()
        if npz_keys is not None:
            npz_keys = ensure_tuple(npz_keys)
        self.npz_keys = npz_keys
        self.channel_dim = channel_dim
        self.kwargs = kwargs
    
    def verify_suffix(self, filename: Union[Sequence[PathLike], PathLike]) -> bool:
        """
        Verify whether the specified file or files format is supported by Numpy reader.

        Args:
            filename: file name or a list of file names to read.
                if a list of files, verify all the suffixes.
        """
        suffixes: Sequence[str] = ["tif", "tiff"]
        return is_supported_format(filename, suffixes)

    def read(self, data: Union[Sequence[PathLike], PathLike], **kwargs):
        """
        Read image data from specified file or files, it can read a list of `no-channel` data files
        and stack them together as multi-channels data in `get_data()`.
        Note that the returned object is Numpy array or list of Numpy arrays.

        Args:
            data: file name or a list of file names to read.
            kwargs: additional args for `numpy.load` API except `allow_pickle`, will override `self.kwargs` for existing keys.
                More details about available args:
                https://numpy.org/doc/stable/reference/generated/numpy.load.html

        """
        img_: List[Nifti1Image] = []

        filenames: Sequence[PathLike] = ensure_tuple(data)
        kwargs_ = self.kwargs.copy()
        kwargs_.update(kwargs)
        for name in filenames:
            img = io.imread(name, **kwargs_)
            #print(name)
            img = img.astype('float32')
            if len(img.shape)==4:
                img = np.swapaxes(img,0,1)
                img = np.swapaxes(img,1,3)
            img_.append(img)
        return img_ if len(img_) > 1 else img_[0]
    
    def get_data(self, img):
        """
        Extract data array and meta data from loaded image and return them.
        This function returns two objects, first is numpy array of image data, second is dict of meta data.
        It constructs `affine`, `original_affine`, and `spatial_shape` and stores them in meta dict.
        When loading a list of files, they are stacked together at a new dimension as the first dimension,
        and the meta data of the first image is used to represent the output meta data.

        Args:
            img: a Numpy array loaded from a file or a list of Numpy arrays.

        """
        img_array: List[np.ndarray] = []
        compatible_meta: Dict = {}
        if isinstance(img, np.ndarray):
            img = (img,)

        for i in ensure_tuple(img):
            header = {"affine":np.eye(5),
                     "labels": {"0": "background",
                                "1": "vessels",
                                "2": "neurons",}
                     }
            if isinstance(i, np.ndarray):
                # if `channel_dim` is None, can not detect the channel dim, use all the dims as spatial_shape
                spatial_shape = np.asarray(i.shape)
                if isinstance(self.channel_dim, int):
                    spatial_shape = np.delete(spatial_shape, self.channel_dim)
                header["spatial_shape"] = spatial_shape
            img_array.append(i)
            header["original_channel_dim"] = self.channel_dim if isinstance(self.channel_dim, int) else "no_channel"
            _copy_compatible_dict(header, compatible_meta)

        return _stack_images(img_array, compatible_meta), compatible_meta

parameter_file = 'hyperparameter_pickle_files/parameters436.pickle'

experiment = re.sub('.pickle',
                    '',
                    re.sub('hyperparameter_pickle_files/parameters',
                           '',
                           parameter_file
                          )
                   )

with open(parameter_file, 'rb') as handle:
    params = pickle.load(handle)

directory = re.sub('.pickle',
                   '',
                   re.sub('hyperparameter_pickle_files/parameters',
                          'training_models/',
                           parameter_file
                         )
                  )

# standard PyTorch program style: create UNet, DiceLoss and Adam optimizer
#device = torch.device("cuda:0")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNETR(
    spatial_dims=3,
    in_channels=2,
    out_channels=3,
    img_size = (128,128,128),
    feature_size = 16,
    hidden_size = 768,
    mlp_dim = 3072,
    pos_embed = "perceptron",
    res_block=True,
    norm_name="instance",
    dropout_rate=params["dropout"]
)
model = torch.nn.DataParallel(model)
model.to(device)
model.load_state_dict(torch.load(
    os.path.join(directory, "best_metric_model_rerun.pth")))

mouse_ids_path = Path('/home/rozakmat/projects/rrg-bojana/data/THY1-TBI')#each mouse has its own folder with raw data in it
mouse_ids = list(mouse_ids_path.glob('*?[0-9]/*res*?[0-9].tif'))#grab folder names/mouse ids
mouse_ids = sorted([x.as_posix() for x in mouse_ids])
data_dicts = [
    {"image":image_name}
    for image_name in mouse_ids
]

data_dicts = data_dicts[::-1]

pred_transforms = Compose(
    [
        LoadImaged(keys=["image"],reader = TIFFReader(channel_dim = 0)),
        EnsureChannelFirstd(keys=["image"]),
        Spacingd(keys=["image"], pixdim=(
            1.01, 1.01, 0.3787), mode=("bilinear")),
        ScaleIntensityRanged(
            keys=["image"], a_min=0, a_max=1024,
            b_min=0.0, b_max=1.0, clip=True,
        ),
        EnsureTyped(keys=["image"]),
    ]
)

pred_ds = Dataset(data=data_dicts, transform=pred_transforms)
pred_loader = DataLoader(pred_ds, batch_size=1, shuffle=False)

num_evals = 20
post_pred = Compose([EnsureType(), AsDiscrete(argmax=True,to_onehot=3)])
softmax = torch.nn.Softmax(dim=1)
model.eval()
for m in model.modules():
    if m.__class__.__name__.startswith('Dropout'):
        m.train()
with torch.no_grad():
    for i, pred_data in tqdm(enumerate(pred_loader)):
        pred_array = np.empty((num_evals,3,507,507,252))
        if pred_data["image"].shape[4]==252:
            for j in range(num_evals):
                roi_size = (128, 128, 128)
                sw_batch_size = 4
                pred_outputs = sliding_window_inference(
                    pred_data["image"].to(device), roi_size, sw_batch_size, model
                )
                pred_outputs = softmax(pred_outputs)
                pred_outputs = pred_outputs.cpu().detach().numpy()
                pred_array[j] = pred_outputs[:]
            mean = np.mean(pred_array,axis=0)
            std = np.std(pred_array,axis=0)
            basename = os.path.basename(os.path.dirname(data_dicts[i]["image"])) + '-' + os.path.basename(data_dicts[i]["image"])
            new_file_name = 'matt_preds/' + basename
            np.save(re.sub('.tif','_mean.npy',new_file_name),mean)
            np.save(re.sub('.tif','_std.npy',new_file_name),std)

