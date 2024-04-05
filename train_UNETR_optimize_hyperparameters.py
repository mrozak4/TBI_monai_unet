from monai.utils import first, set_determinism, ensure_tuple
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
    LabelToContour,
    RandCoarseDropoutd
)
from monai.handlers.utils import from_engine
from monai.networks.nets import UNet, UNETR
from monai.networks.layers import Norm
from monai.metrics import DiceMetric, HausdorffDistanceMetric, get_confusion_matrix, ConfusionMatrixMetric
from monai.losses import DiceLoss, DiceCELoss, DiceFocalLoss
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch, ImageReader
from monai.data.image_reader import WSIReader
from monai.config import print_config, KeysCollection, PathLike
from monai.apps import download_and_extract
import torch
import matplotlib.pyplot as plt
import tempfile
import shutil
import os
import glob
from numpy import random
from pathlib import Path
import re
from skimage import io
from typing import Optional, Union, Sequence, Callable, Dict, List
from monai.data.utils import is_supported_format
from monai. data.image_reader import _copy_compatible_dict, _stack_images
from nibabel.nifti1 import Nifti1Image
from PIL import Image
import numpy as np
from tqdm import tqdm
import pickle
import pandas as pd
from mlflow import log_metric, log_param, log_artifacts, set_experiment, start_run, end_run
import warnings
import argparse
from sklearn.metrics import confusion_matrix
warnings.filterwarnings('ignore')

################################
#Parser
################################

parser = argparse.ArgumentParser(description='take hyperparameter inputs')

parser.add_argument('-c','--cfg', type=str, dest='cfg', action='store')

args = parser.parse_args()

file = args.cfg

################################
#TIFF READER
################################



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




##########################
#Hyperparameters
##########################

parameter_file = file#'hyperparameter_pickle_files/parameters57.pickle'

experiment = re.sub('.pickle',
                    '',
                    re.sub('hyperparameter_pickle_files/parameters',
                           '',
                           parameter_file
                          )
                   )
set_experiment("TBI_UNet_adam_optimizer")

with open(parameter_file, 'rb') as handle:
    params = pickle.load(handle)

for i in params.keys():
    log_param(i,params[i])

params["max_epochs"] = params["max_epochs"] * 3

log_param('model','unet')

###########################
#Set up directory to save output
###########################

directory = re.sub('.pickle',
                   '',
                   re.sub('hyperparameter_pickle_files/parameters',
                          'training_models_upsampled/',
                           parameter_file
                         )
                  )
try:
    os.mkdir(directory)
except OSError as error:
    print(error) 
#log_artifacts(directory)
print(directory)

###########################
# Get train, validate , test data
###########################

#make list of data dictionaries
train_images_path = Path('/home/rozakmat/projects/rrg-bojana/rozakmat/TBI/GT_filtered+raw') #raw path images
train_images_paths = list(train_images_path.glob('*.tif'))#get images
train_images = sorted([x.as_posix() for x in train_images_paths])#sort
train_labels_path = Path('/home/rozakmat/projects/rrg-bojana/rozakmat/TBI/GT_filtered+raw')#labels path
train_labels = list(train_labels_path.glob('*sub1.tiff'))#get label images
train_labels = sorted([x.as_posix() for x in train_labels])#sort
#combine images and labels into monai dictionary format
data_dicts = [
    {"image":image_name, "label":label_name}
    for image_name, label_name in zip(train_images,train_labels)
]

mouse_ids_path = Path('/home/rozakmat/projects/rrg-bojana/rozakmat/TBI/raw')#each mouse has its own folder with raw data in it
mouse_ids = list(mouse_ids_path.glob('*'))#grab molder names/mouse ids
images = sorted([y.name for y in train_images_paths])#sort
#get mouse id corresponding to each image i have labels for
mouse_ids_with_raw_tiff = []
for i in mouse_ids:
    for j in images:
        if len(list(i.glob(j))) !=0:
            mouse_ids_with_raw_tiff.append(list(i.glob(j)))
#flatten the list and sort
mouse_ids_with_raw_tiff_flat = [item for sublist in mouse_ids_with_raw_tiff for item in sublist]
mouse_ids_with_raw_tiff_flat = sorted(mouse_ids_with_raw_tiff_flat)

#shuffle mouse ids for a 15/4/6 split train/val/test by mouse id
mouse_ids_present = [i.parent.name for i in mouse_ids_with_raw_tiff_flat]
mouse_ids_present = sorted(list(np.unique(mouse_ids_present)))
np.random.seed(643)
np.random.shuffle(mouse_ids_present)
mouse_ids_present
train = mouse_ids_present[:15]
log_param('train_set',' '.join(train))
val = mouse_ids_present[15:-6]
log_param('val_set',' '.join(val))
test = mouse_ids_present[-6:]

train_files = []
val_files = []
test_files = []
for i in train:
    for j in mouse_ids_with_raw_tiff_flat:
        if i in j.as_posix():
            for k in data_dicts:
                if j.name in k["image"]:
                    train_files.append(k)
for i in val:
    for j in mouse_ids_with_raw_tiff_flat:
        if i in j.as_posix():
            for k in data_dicts:
                if j.name in k["image"]:
                    val_files.append(k)
for i in test:
    for j in mouse_ids_with_raw_tiff_flat:
        if i in j.as_posix():
            for k in data_dicts:
                if j.name in k["image"]:
                    test_files.append(k)

#########################
#Train and validation transforms
#########################

set_determinism(seed=12)

train_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"], reader = TIFFReader(channel_dim = 0)),
        EnsureChannelFirstd(keys=["image", "label"]),
        Spacingd(keys=["image", "label"],  pixdim=(
            1/3, 1/3, 0.375/3), mode=("bilinear", "nearest")),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityRanged(
            keys=["image"], a_min=0, a_max=1024,
            b_min=0.0, b_max=1.0, clip=True,
        ),
        AsDiscreted(keys="label",
                   rounding='torchrounding',
                   to_onehot=True),
        RandSpatialCropSamplesd(
            keys=['image','label'],
            num_samples = params['N_crops'],
            roi_size = params['crop_size'],
            random_size = False
        ),
        #Deformation_transforms
        OneOf(transforms=[Rand3DElasticd(keys = ["image","label"],
                                        sigma_range = params['Rand3DElasticd_sigma_range'],
                                        magnitude_range = params['Rand3DElasticd_magnitude_range'],
                                        prob = params["deformation_transforms_prob"],
                                        mode = ["bilinear","nearest"]),
                          RandGridDistortiond(keys = ["image","label"],
                                             num_cells = params['RandGridDistortiond_num_cells'],
                                             prob = params["deformation_transforms_prob"],
                                             distort_limit = params['RandGridDistortiond_distort_limit'],
                                             mode = ["bilinear","nearest"]
                                             )
                         ]
             ),
        #Intensity_Transforms
        OneOf(transforms = [RandShiftIntensityd(keys = ["image"],
                                                offsets = params['RandShiftIntensityd_offsets'],
                                                prob = params["intensity_transform_probability"]),
                            RandAdjustContrastd(keys = ["image"],
                                                prob = params["intensity_transform_probability"],
                                                gamma = params['RandAdjustContrastd_gamma']),
                            RandHistogramShiftd(keys = ["image"],
                                                prob = params["intensity_transform_probability"],
                                                num_control_points = params['RandHistogramShiftd_num_control_points'])
                           ]
             ),
        #Gaussian_Transforms
        OneOf(transforms = [RandGaussianSharpend(keys = ["image"],
                                                 prob = params["gaussian_transform_probability"]),
                            RandGaussianSmoothd(keys = ["image"],
                                                prob = params["gaussian_transform_probability"]),
                            RandGaussianNoised(keys = ["image"],
                                               prob = params["gaussian_transform_probability"],
                                               mean = params['RandGaussianNoised_mean'],
                                               std = params['RandGaussianNoised_std'])
                           ]
             ),
        RandCoarseDropoutd(keys = ["image"],
                           prob=0.75,
                           holes = 50,
                           spatial_size=2,
                           max_holes = 1000,
                           max_spatial_size=6,
                           fill_value = (0.0001,0.1)
        ),
        #rottion+flip_transforms
        RandRotate90d(
            keys = ["image", "label"],
            prob = params['rotation_flip_transforms_probability'],
            max_k = 3,
        ),
        RandFlipd(
            keys = ["image", "label"],
            spatial_axis = [0],
            prob = params['rotation_flip_transforms_probability'],
        ),
        RandFlipd(
            keys = ["image", "label"],
            spatial_axis = [1],
            prob = params['rotation_flip_transforms_probability'],
        ),
        RandFlipd(
            keys = ["image", "label"],
            spatial_axis = [2],
            prob = params['rotation_flip_transforms_probability'],
        ),
        
        EnsureTyped(keys=["image", "label"]),
    ]
)
val_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"],reader = TIFFReader(channel_dim = 0)),
        EnsureChannelFirstd(keys=["image", "label"]),
        Spacingd(keys=["image", "label"], pixdim=(
            1.01, 1.01, 0.3787), mode=("bilinear", "nearest")),
        ScaleIntensityRanged(
            keys=["image"], a_min=0, a_max=1024,
            b_min=0.0, b_max=1.0, clip=True,
        ),
        AsDiscreted(keys="label",
                   rounding='torchrounding',
                   to_onehot=True),
        EnsureTyped(keys=["image", "label"]),
    ]
)

#################################
#CacheDataset and  DataLoader for training and validation
#################################

train_ds = CacheDataset(
    data=train_files, transform=train_transforms,
    cache_rate=1.0, num_workers=4)
# use batch_size=2 to load images and use RandCropByPosNegLabeld
# to generate 2 x 4 images for network training
train_loader = DataLoader(train_ds, batch_size=params["batch_size"], shuffle=True, num_workers=4)

val_ds = CacheDataset(
    data=val_files, transform=val_transforms, cache_rate=1.0, num_workers=4)
val_loader = DataLoader(val_ds, batch_size=1, num_workers=4)

test_ds = CacheDataset(
    data=test_files, transform=val_transforms, cache_rate=1.0, num_workers=4)
test_loader = DataLoader(test_ds, batch_size=1, num_workers=4)

################################
#Create Model, Loss, Optimizer
################################

# standard PyTorch program style: create UNet, DiceLoss and Adam optimizere = torch.device("cuda:0")
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
loss_function = params['loss_function']
optimizer = params['optimizer'](params = model.parameters(), 
                                lr = params['learning_rate'])
dice_metric = DiceMetric(
    include_background=False,
    reduction="mean"
)
dice_metric_test = DiceMetric(
    include_background=False,
    reduction="mean"
)
confusion_metric = ConfusionMatrixMetric(metric_name= ["sensitivity", 
                                                       "specificity", 
                                                       "precision", 
                                                       "negative predictive value", 
                                                       "miss rate", 
                                                       "fall out", 
                                                       "false discovery rate", 
                                                       "false omission rate", 
                                                       "prevalence threshold", 
                                                       "threat score", 
                                                       "accuracy", 
                                                       "balanced accuracy", 
                                                       "f1 score", 
                                                       "matthews correlation coefficient", 
                                                       "fowlkes mallows index", 
                                                       "informedness", 
                                                       "markedness"],
                                         include_background=False)
confusion_metric_test = ConfusionMatrixMetric(metric_name= ["sensitivity", 
                                                            "specificity", 
                                                            "precision", 
                                                            "negative predictive value", 
                                                            "miss rate", 
                                                            "fall out", 
                                                            "false discovery rate", 
                                                            "false omission rate", 
                                                            "prevalence threshold", 
                                                            "threat score", 
                                                            "accuracy", 
                                                            "balanced accuracy", 
                                                            "f1 score", 
                                                            "matthews correlation coefficient", 
                                                            "fowlkes mallows index", 
                                                            "informedness", 
                                                            "markedness"],
                                              include_background=False)
#hausdorf_distance_metric = HausdorffDistanceMetric(include_background=False,
#                                                   distance_metric='euclidean')
dice_metric_deform = DiceMetric(
    include_background = False,
    reduction = "mean"
)
dice_metric_deform_test = DiceMetric(
    include_background = False,
    reduction = "mean"
)
dice_metric_deform_boundary_difference = DiceMetric(
    include_background = False,
    reduction = "mean"
)
dice_metric_deform_boundary_difference_test = DiceMetric(
    include_background = False,
    reduction = "mean"
)
dice_metric_predicted_deform_boundary_difference = DiceMetric(
    include_background = False,
    reduction = "mean"
)
dice_metric_predicted_deform_boundary_difference_test = DiceMetric(
    include_background = False,
    reduction = "mean"
)
dice_metric_boundary_difference_detection = DiceMetric(
    include_background = False,
    reduction = "mean"
)
dice_metric_boundary_difference_detection_test = DiceMetric(
    include_background = False,
    reduction = "mean"
)
label_filter = Compose(
    [EnsureType(),
     LabelFilter(applied_labels = (1))
    ]
)
deform = Rand3DElastic(
    sigma_range = (1,1.1),
    magnitude_range = (3,4),
    prob = 1
)

################################
#Train
################################

random.seed(12)
max_epochs = 2400#800#params["max_epochs"]
val_interval = 2
best_metric = -1
best_metric_epoch = -1
epoch_loss_values = []
metric_values = []
metric_values_test = []
confusion_matrix_values = []
confusion_matrix_values_test = []
val_confusion_matrix = []
test_confusion_matrix = []
#hausdorf_distance_values = []
metric_values_deform = []
metric_values_deform_test = []
metric_values_deform_boundary_difference = []
metric_values_deform_boundary_difference_test = []
metric_values_predicted_deform_boundary_difference = []
metric_values_predicted_deform_boundary_difference_test = []
metric_values_boundary_difference_detection = []
metric_values_boundary_difference_detection_test = []
post_pred = Compose([EnsureType(), AsDiscrete(argmax=True,to_onehot=3)])
post_label = Compose([EnsureType(), AsDiscrete(to_onehot=3)])
for epoch in tqdm(range(max_epochs)):
    print("-" * 10)
    print(f"epoch {epoch + 1}/{max_epochs}")
    model.train()
    epoch_loss = 0
    step = 0
    for batch_data in train_loader:
        step += 1
        inputs, labels = (
            batch_data["image"].to(device),
            batch_data["label"].to(device),
        )
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        print(
            f"{step}/{len(train_ds) // train_loader.batch_size}, "
            f"train_loss: {loss.item():.4f}")
    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    log_metric('epoch_loss',epoch_loss, step = epoch)
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
    del outputs

    if (epoch + 1) % val_interval == 0:
        model.eval()
        with torch.no_grad():
            for val_data in val_loader:
                seed = random.randint(0,10000000)
                val_inputs, val_labels = (
                    val_data["image"].to(device),
                    val_data["label"].to(device),
                )
                roi_size = (128, 128, 128)
                sw_batch_size = 1
                val_outputs = sliding_window_inference(
                    val_inputs, 
                    roi_size, 
                    sw_batch_size, 
                    model
                )
                #get prediciton output
                val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                #deform raw image
                #deform.set_random_state(seed = seed)
                #deform_val_inputs = torch.unsqueeze(deform(torch.squeeze(val_inputs),mode='bilinear'),axis=0)
                ##deform validation output
                #deform.set_random_state(seed = seed)
                #deform_val_outputs_gt = deform(val_outputs[0],mode='nearest')
                ##predict on deformed raw image
                #deform_val_outputs =sliding_window_inference(
                #    deform_val_inputs, 
                #    roi_size, 
                #    sw_batch_size, 
                #    model
                #)   
                # get predicted outputs
                #val_outputs_deform = [post_pred(i) for i in decollate_batch(deform_val_outputs)]
                #fill holes in prediction
                filled  = [FillHoles(connectivity=2)(i) for i in val_outputs]
                #filled_deform  = [FillHoles(connectivity=2)(i) for i in val_outputs_deform]
                #generated boundaries from predictions
                #filled_vessels_boundary = [LabelToContour()(label_filter(i)) for i in filled]
                #filled_vessels_boundary_deform = [LabelToContour()(label_filter(i)) for i in filled_deform]
                #get labelled data
                val_labels = [post_label(i) for i in decollate_batch(val_labels)]
                #print(val_labels.shape)
                #val_labels_deform = [deform_val_outputs_gt]
                #get validation boundaries
                #val_labels_boundary = [LabelToContour()(i) for i in val_labels]
                #val_outputs_boundary_deform = [LabelToContour()(i) for i in val_labels_deform]
                # compute metric for current iteration
                #dice metric for ground truth and prediction
                dice_metric(
                    y_pred = filled,
                    y = val_labels
                )
                confusion_metric(y_pred = val_outputs,
                                 y = val_labels)
                #hausdorf_distance_metric(y_pred=filled, 
                #                         y=val_labels)
                #dice metric for deformed output, and prediction on deformed raw image
                #dice_metric_deform(
                #    y_pred = filled_deform,
                #    y = val_outputs_deform
                #)
                #dice metric for boundary of ground truth and deformed ground truth
                #measures how much the boundary was deformed
                #dice_metric_deform_boundary_difference(
                #    y_pred = val_outputs_boundary_deform,
                #    y = filled_vessels_boundary
                #)
                
                #dice_metric_predicted_deform_boundary_difference(
                #    y_pred=val_outputs_boundary_deform[0],
                #    y=filled_vessels_boundary_deform[0]
                #)
                #dice metric for boundary difference detection
                #dice_metric_boundary_difference_detection(
                #    y_pred = filled_vessels_boundary_deform[0].ne(filled_vessels_boundary[0])[1],
                #    y = val_outputs_boundary_deform[0].ne(filled_vessels_boundary[0]).int()[1]
                #)
                del val_outputs
            for test_data in test_loader:
                seed = random.randint(0,10000000)
                test_inputs, test_labels = (
                    test_data["image"].to(device),
                    test_data["label"].to(device),
                )
                roi_size = (128, 128, 128)
                sw_batch_size = 1
                test_outputs = sliding_window_inference(
                    test_inputs, 
                    roi_size, 
                    sw_batch_size, 
                    model
                )
                #get prediciton output
                test_outputs = [post_pred(i) for i in decollate_batch(test_outputs)]
                #deform raw image
                #deform.set_random_state(seed = seed)
                #deform_test_inputs = torch.unsqueeze(deform(torch.squeeze(test_inputs),mode='bilinear'),axis=0)
                ##deform validation output
                #deform.set_random_state(seed = seed)
                #deform_test_outputs_gt = deform(test_outputs[0],mode='nearest')
                ##predict on deformed raw image
                #deform_test_outputs =sliding_window_inference(
                #    deform_test_inputs, 
                #    roi_size, 
                #    sw_batch_size, 
                #    model
                #)   
                # get predicted outputs
                #test_outputs_deform = [post_pred(i) for i in decollate_batch(deform_test_outputs)]
                #fill holes in prediction
                filled  = [FillHoles(connectivity=2)(i) for i in test_outputs]
                #filled_deform  = [FillHoles(connectivity=2)(i) for i in test_outputs_deform]
                #generated boundaries from predictions
                #filled_vessels_boundary = [LabelToContour()(label_filter(i)) for i in filled]
                #filled_vessels_boundary_deform = [LabelToContour()(label_filter(i)) for i in filled_deform]
                #get labelled data
                test_labels = [post_label(i) for i in decollate_batch(test_labels)]
                #print(val_labels.shape)
                #test_labels_deform = [deform_test_outputs_gt]
                #get validation boundaries
                #test_labels_boundary = [LabelToContour()(i) for i in test_labels]
                #test_outputs_boundary_deform = [LabelToContour()(i) for i in test_labels_deform]
                # compute metric for current iteration
                #dice metric for ground truth and prediction
                dice_metric_test(
                    y_pred = filled,
                    y = test_labels
                )
                confusion_metric_test(y_pred = filled,
                                      y = test_labels)
                #hausdorf_distance_metric(y_pred=filled, 
                #                         y=val_labels)
                #dice metric for deformed output, and prediction on deformed raw image
                #dice_metric_deform_test(
                #    y_pred = filled_deform,
                #    y = test_outputs_deform
                #)
                #dice metric for boundary of ground truth and deformed ground truth
                #measures how much the boundary was deformed
                #dice_metric_deform_boundary_difference_test(
                #    y_pred = test_outputs_boundary_deform,
                #    y = filled_vessels_boundary
                #)
                
                #dice_metric_predicted_deform_boundary_difference_test(
                #    y_pred=test_outputs_boundary_deform[0],
                #    y=filled_vessels_boundary_deform[0]
                #)
                #dice metric for boundary difference detection
                #dice_metric_boundary_difference_detection_test(
                #    y_pred = filled_vessels_boundary_deform[0].ne(filled_vessels_boundary[0])[1],
                #    y = test_outputs_boundary_deform[0].ne(filled_vessels_boundary[0]).int()[1]
                #)
                del test_outputs

            # aggregate the final mean dice result
            metric = dice_metric.aggregate().item()
            log_metric(
                'dice',
                metric,
                step = epoch
            )
            metric_test = dice_metric_test.aggregate().item()
            log_metric(
                'dice_test',
                metric_test,
                step = epoch
            )
            confusion_matrix_metrics = np.array(torch.tensor(confusion_metric.aggregate()).cpu())
            confusion_matrix_metrics_test = np.array(torch.tensor(confusion_metric_test.aggregate()).cpu())
            #hausdorf_distance = hausdorf_distance_metric.aggregate().item()
            #aggregate the dice score for the prediction fo the deformed raw image againsed the deformed prediction
            #metric_deform = dice_metric_deform.aggregate().item()
            log_metric(
                'dice_deformed_validation',
                metric_deform,
                step = epoch
            )
            metric_deform_test = dice_metric_deform_test.aggregate().item()
            log_metric(
                'dice_deformed_validation_test',
                metric_deform_test,
                step = epoch
            )
            metric_deform_boundary_difference = dice_metric_deform_boundary_difference.aggregate().item()
            log_metric(
                'ground_truths_boundary_dice',
                metric_deform_boundary_difference,
                step = epoch
            )
            metric_deform_boundary_difference_test = dice_metric_deform_boundary_difference_test.aggregate().item()
            log_metric(
                'ground_truths_boundary_dice_test',
                metric_deform_boundary_difference_test,
                step = epoch
            )
            metric_predicted_deform_boundary_difference = dice_metric_predicted_deform_boundary_difference.aggregate().item()
            log_metric(
                'predicted_boundary_dice',
                metric_predicted_deform_boundary_difference,
                step = epoch
            )
            metric_predicted_deform_boundary_difference_test = dice_metric_predicted_deform_boundary_difference_test.aggregate().item()
            log_metric(
                'predicted_boundary_dice_test',
                metric_predicted_deform_boundary_difference_test,
                step = epoch
            )
            metric_boundary_difference_detection = dice_metric_boundary_difference_detection.aggregate().item()
            log_metric(
                'dice_deformed_boundary_ground_truth_to_prediction',
                metric_boundary_difference_detection,
                step = epoch
            )
            metric_boundary_difference_detection_test = dice_metric_boundary_difference_detection_test.aggregate().item()
            log_metric(
                'dice_deformed_boundary_ground_truth_to_prediction_test',
                metric_boundary_difference_detection_test,
                step = epoch
            )
            # reset the status for next validation round
            dice_metric.reset()
            dice_metric_test.reset()
            #hausdorf_distance_metric.reset()
            dice_metric_deform.reset()
            dice_metric_deform_test.reset()
            dice_metric_deform_boundary_difference.reset()
            dice_metric_deform_boundary_difference_test.reset()
            #dice_metric_predicted_deform_boundary_difference.reset()
            dice_metric_boundary_difference_detection.reset()
            dice_metric_boundary_difference_detection_test.reset()

            metric_values.append(
                metric
            )
            metric_values_test.append(
                metric_test
            )
            confusion_matrix_values.append(
                confusion_matrix_metrics
            )
            confusion_matrix_values_test.append(
                confusion_matrix_metrics_test
            )
            #hausdorf_distance_values.append(
            #    hausdorf_distance
            #)
            metric_values_deform.append(
                metric_deform
            )
            metric_values_deform_test.append(
                metric_deform_test
            )
            metric_values_deform_boundary_difference.append(
                metric_deform_boundary_difference
            )
            metric_values_deform_boundary_difference_test.append(
                metric_deform_boundary_difference_test
            )
            metric_values_predicted_deform_boundary_difference.append(
                metric_predicted_deform_boundary_difference
            )
            metric_values_predicted_deform_boundary_difference_test.append(
                metric_predicted_deform_boundary_difference_test
            )
            metric_values_boundary_difference_detection.append(
                metric_boundary_difference_detection
            )
            metric_values_boundary_difference_detection_test.append(
                metric_boundary_difference_detection_test
            )
            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), os.path.join(
                    directory, "best_metric_model_rerun.pth"))
                print("saved new best metric model")
            print(
                f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                #f"\ncurrent mean hausdorf distance: {hausdorf_distance:.4f}"
                f"\ncurrent mean dice of deformed mask to predicted deformed image: {metric_deform:.4f}"
                f"\ncurrent mean dice of boundary of ground truth, and  boundary of deformed ground truth: {metric_deform_boundary_difference:.5f}"
                #f"\ncurrent mean dice of preficted boundary, and predicted deformed boundary: {metric_predicted_deform_boundary_difference:.5f}"
                f"\ncurrent mean dice of the boundary difference due to deformation: {metric_boundary_difference_detection:.5f}"
                #f"\nratio of dice of boundary deformation to ground truth on prediction and ground truth:{metric_predicted_deform_boundary_difference/metric_deform_boundary_difference}"
                f"\nbest mean dice: {best_metric:.4f} "
                f"at epoch: {best_metric_epoch}"
            )

print(
    f"train completed, best_metric: {best_metric:.4f} "
    f"at epoch: {best_metric_epoch}"
)
log_metric("best_epoch",best_metric_epoch)

#################################
#Save Metrics
#################################

df = pd.DataFrame()
df['Epoch_Average_Loss'] = epoch_loss_values
df.to_excel(directory + '/metrics_loss_rerun.xlsx')
np.save(directory + '/confusion_rerun.npy',np.array(confusion_matrix_values))
np.save(directory + '/confusion_test_rerun.npy',np.array(confusion_matrix_values_test))
df = pd.DataFrame()
df['Val_Mean_Dice'] = metric_values
df['test_Mean_Dice'] = metric_values_test
df['boundary_detection_dice'] = metric_values_boundary_difference_detection
df['boundary_detection_dice_test'] = metric_values_boundary_difference_detection_test
df['boundary_difference_dice'] = metric_values_deform_boundary_difference
df['boundary_difference_dice_test'] = metric_values_deform_boundary_difference_test
df.to_excel(directory + '/metrics_validation_rerun.xlsx')

#################################
# Plot the loss and dice
#################################

plt.figure("train", (12, 6))
plt.subplot(1, 2, 1)
plt.title("Epoch Average Loss")
x = [i + 1 for i in range(len(epoch_loss_values))]
y = epoch_loss_values
plt.xlabel("epoch")
plt.plot(x, y)
plt.subplot(1, 2, 2)
plt.title("Val Mean Dice")
x = [val_interval * (i + 1) for i in range(len(metric_values))]
y = metric_values
plt.xlabel("epoch")
plt.plot(x, y)
plt.savefig(directory +'ipynb_trial_Val_Loss+Dice.png')








