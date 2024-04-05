
from monai.data import ( 
    DataLoader, 
    Dataset, 
)
import torch
from os.path import exists
from pathlib import Path
from re import sub
from numpy import save, float16
from tqdm import tqdm
import warnings
from numpy.random import shuffle
from predict import get_model, predict, get_pred_transforms
warnings.filterwarnings('ignore')


model = get_model()    


mouse_ids_path = Path('matt_raw_warped_single')#each mouse has its own folder with raw data in it
mouse_ids = list(mouse_ids_path.glob('*res*.tif'))#grab folder names/mouse ids
mouse_ids = sorted([x.as_posix() for x in mouse_ids])
shuffle(mouse_ids)
data_dicts = [
    {"image":image_name}
    for image_name in mouse_ids if not exists(sub('.tif','_pred.npy',sub('matt_raw_warped_single','/home/rozakmat/projects/rrg-bojana/rozakmat/TBI_monai_UNET/matt_raw_warped_single_upsampled',image_name)))
]

print(len(data_dicts))

#data_dicts = [data_dicts[_i]]

pred_transforms = get_pred_transforms()

pred_ds = Dataset(data=data_dicts, transform=pred_transforms)
pred_loader = DataLoader(pred_ds, batch_size=1, shuffle=False)

with torch.no_grad():
    for i, pred_data in tqdm(enumerate(pred_loader)):
        new_file_name = sub('matt_raw_warped_single','/home/rozakmat/projects/rrg-bojana/rozakmat/TBI_monai_UNET/matt_raw_warped_single_upsampled',data_dicts[i]["image"])
        if not exists(sub('.tif','_pred.npy',new_file_name)):
            pred_array = predict(pred_data, num_evals=20, model=model)
            mean = float16(pred_array.mean(axis=0))
            save(sub('projects/rrg-bojana/rozakmat','projects/rrg-bojana/rozakmat',sub('.tif','_mean.npy',new_file_name)),mean)
            #print(sub('projects/rrg-bojana/rozakmat','projects/rrg-bojana/rozakmat',sub('.tif','_mean.npy',new_file_name)))
            save(sub('.tif','_pred.npy',new_file_name),pred_array)
            #print(sub('.tif','_pred.npy',new_file_name))
            #print(re.sub('.tif','_mean.npy',new_file_name))

