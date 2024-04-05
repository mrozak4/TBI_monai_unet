import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from pathlib import Path
import networkx as nx
import pickle
import re
from IPython.display import display
from tqdm import tqdm
from skimage import io
import scipy as sp
import os
from scipy.ndimage import distance_transform_edt
import torch
from torch_geometric.utils.convert import from_networkx
from torch_geometric.loader import DataLoader
from IPython.display import clear_output
from sklearn.preprocessing import normalize
from sklearn.manifold import TSNE
import torch_geometric.transforms as T
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import rand_score
from sklearn.metrics import classification_report
#import pytorchgeometric as ptg
import argparse
import numpy as np
from pathlib import Path
import re
import networkx as nx
import pickle
from skimage import io
from torch_geometric.utils.convert import from_networkx
from torch_geometric.loader import DataLoader
from tqdm import tqdm 
import torch
from torch.nn import Linear
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GraphSAGE
from scipy.ndimage import distance_transform_edt
from os.path import exists
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def visualize(h, color):
    #z = TSNE(n_components=2).fit_transform(h.detach().cpu().numpy())
    z = h.detach().numpy()
    plt.figure(figsize=(10,10))
    plt.xticks([])
    plt.yticks([])

    plt.scatter(z[:, 0], z[:, 1], s=70, c=color, cmap="Set2")
    plt.legend()
    plt.show()
    
################################
#Parser
################################

parser = argparse.ArgumentParser(description='take hyperparameter inputs')

parser.add_argument('-c','--cfg', type=str, dest='cfg', action='store')

args = parser.parse_args()

param_file = args.cfg
    
folder = Path('/home/rozakmat/projects/rrg-bojana/rozakmat/TBI_monai_UNET/matt_preds_graphs_fwhm_single_excel')
files = list(folder.glob('*_warped_radii_amended_AVC.pickle'))
files = sorted([x.as_posix() for x in files])
print(files[0])
len(files)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#io.imread(files).shape

#param_files = list(Path('hyperparameter_pickle_files').glob('GraphSAGE*.pickle'))
#param_files = sorted([x.as_posix() for x in param_files])
#param_file = param_files[param_file_index]
with open(param_file, 'rb') as handle:
    params = pickle.load(handle)
print(params)

subjects = list(Path('/home/rozakmat/projects/rrg-bojana/rozakmat/James_VBM/datasets').glob('*'))
subjects = sorted([x.as_posix() for x in subjects])
np.random.seed(12)
np.random.shuffle(subjects)
train_subj = subjects[:4]
test_subj = subjects[4:6]
val_subj = subjects[6:]
train_imgs = list(np.concatenate([list(Path(x).glob('*.oir')) for x in train_subj]).flat)
train_imgs = np.unique(sorted([re.sub('_0001','',re.sub('.oir','',x.name)) for x in train_imgs]))
train_pickles = [x for x in files if any(y in x for y in train_imgs)]

test_imgs = list(np.concatenate([list(Path(x).glob('*.oir')) for x in test_subj]).flat)
test_imgs = np.unique(sorted([re.sub('_0001','',re.sub('.oir','',x.name)) for x in test_imgs]))
test_pickles = [x for x in files if any(y in x for y in test_imgs)]

val_imgs = list(np.concatenate([list(Path(x).glob('*.oir')) for x in val_subj]).flat)
val_imgs = np.unique(sorted([re.sub('_0001','',re.sub('.oir','',x.name)) for x in val_imgs]))
val_pickles = [x for x in files if any(y in x for y in val_imgs)]

#display(train_pickles)
#display(test_pickles)
display(val_pickles[0])
print(len(files))
data_list = []

for file in tqdm(train_pickles[:]):
    #file = 'James_labelling_AVC/james_preds/XYZres019_warped_single_radii_matched_AVC.pickle'
    with open(file, 'rb') as f:
        G = graph = pickle.load(f)
    tif_img = io.imread(re.sub('matt_preds_graphs_fwhm_single_excel','matt_raw_warped_single',re.sub('_warped_radii_amended_AVC.pickle','_warped.tif',file))).astype('float32')
    #img_resized = sp.ndimage.zoom(tif_img[:,0],1/(np.array(tif_img[:,0].shape)/np.array([512,507,507])))
    #break
    if os.path.exists(re.sub('_warped_radii_amended_AVC.pickle','_warped_up.tif',file)):
        img_resized = io.imread(re.sub('_warped_radii_amended_AVC.pickle','_warped_up.tif',file)).astype('float32')
    else:
        img_resized = sp.ndimage.zoom(tif_img[:,0],1/(np.array(tif_img[:,0].shape)/np.array([254,512,512])))
        io.imsave(re.sub('_warped_radii_amended_AVC.pickle','_warped_up.tif',file),img_resized)
    if os.path.exists(re.sub('_radii_amended_AVC.pickle','_nrn_dst.tif',file)):
        nrn_dst = io.imread(re.sub('_radii_amended_AVC.pickle','_nrn_dst.tif',file)).astype('float32')
    else:
        nrn_seg = 1 - (np.argmax(np.load(re.sub('matt_preds_graphs_fwhm_single_excel','matt_raw_warped_single_upsampled',re.sub('_radii_amended_AVC.pickle','_mean.npy',file))),axis=0)==2)*1 #get neuron segmentation for more features
        nrn_dst = distance_transform_edt(nrn_seg) #get image of distance to nearest neuron
        io.imsave(re.sub('_radii_amended_AVC.pickle','_nrn_dst.tif',file),nrn_dst)
    
    for edge in G.edges:
        for keys in G[edge[0]][edge[1]]:
            if type(G[edge[0]][edge[1]][keys]) == str:
                G[edge[0]][edge[1]][keys] = 0
        path = graph[edge[0]][edge[1]]['pts']
        path_smooth = np.float32(np.copy(path))  
        for k in range(len(path[0])):
            path_smooth[:,k] = sp.ndimage.gaussian_filter1d(np.float64(path[:,k]),3,mode='nearest')
        path_grad = np.gradient(path_smooth,edge_order=2)[0]
        path_grad = normalize(path_grad,axis=1) 
        G[edge[0]][edge[1]]['x'] = np.array([path_grad[:,0].astype(np.float16),
                                             path_grad[:,1].astype(np.float16),
                                             path_grad[:,2].astype(np.float16),
                                             np.array(G[edge[0]][edge[1]]['path_weights']).astype(np.float16)/100,
                                             img_resized[path[:,2],path[:,1],path[:,0]].astype(np.float16)/1024])
        G[edge[0]][edge[1]]['x'][np.isnan(G[edge[0]][edge[1]]['x'])] = 0
        G[edge[0]][edge[1]]['x'] = np.pad(G[edge[0]][edge[1]]['x'],(0,1000-len(G[edge[0]][edge[1]]['x'][0])), 'constant')[0:4]
        G[edge[0]][edge[1]]['x'] = G[edge[0]][edge[1]]['x'].T.flatten()
        G[edge[0]][edge[1]]['y'] = G[edge[0]][edge[1]]['A_V_C']
    G_line = nx.line_graph(G, create_using=nx.Graph)
    for i in G_line.nodes:
        G_line.nodes[i]['x'] = G[i[0]][i[1]]['x']
        G_line.nodes[i]['y'] = G[i[0]][i[1]]['y']
    data = from_networkx(G_line, group_node_attrs = ['x'])
    data.num_classes = len(np.unique(data.y))
    data_list.append(data)
loader = DataLoader(data_list,batch_size=params['batch_size'], shuffle = True, exclude_keys = ['pts','o','edge_pts','weight','radii','radii_std','vessel_type','vessel_type_std','path_weights','end-0','end-1','radii_0001','radii_0001_std','pts_0001','delta','A_V_C'])
exit
val_data_list = []
for file in tqdm(val_pickles[:]):
    #file = 'James_labelling_AVC/james_preds/XYZres019_warped_single_radii_matched_AVC.pickle'
    with open(file, 'rb') as f:
        G = graph = pickle.load(f)
    tif_img = io.imread(re.sub('matt_preds_graphs_fwhm_single_excel','matt_raw_warped_single',re.sub('_warped_radii_amended_AVC.pickle','_warped.tif',file))).astype('float32')
    #img_resized = sp.ndimage.zoom(tif_img[:,0],1/(np.array(tif_img[:,0].shape)/np.array([512,507,507])))
    #break
    if os.path.exists(re.sub('_warped_radii_amended_AVC.pickle','_warped_up.tif',file)):
        img_resized = io.imread(re.sub('_warped_radii_amended_AVC.pickle','_warped_up.tif',file)).astype('float32')
    else:
        img_resized = sp.ndimage.zoom(tif_img[:,0],1/(np.array(tif_img[:,0].shape)/np.array([254,512,512])))
        io.imsave(re.sub('_warped_radii_amended_AVC.pickle','_warped_up.tif',file),img_resized)
    if os.path.exists(re.sub('_radii_amended_AVC.pickle','_nrn_dst.tif',file)):
        nrn_dst = io.imread(re.sub('_radii_amended_AVC.pickle','_nrn_dst.tif',file)).astype('float32')
    else:
        nrn_seg = 1 - (np.argmax(np.load(re.sub('matt_preds_graphs_fwhm_single_excel','matt_raw_warped_single_upsampled',re.sub('_radii_amended_AVC.pickle','_mean.npy',file))),axis=0)==2)*1 #get neuron segmentation for more features
        nrn_dst = distance_transform_edt(nrn_seg) #get image of distance to nearest neuron
        io.imsave(re.sub('_radii_amended_AVC.pickle','_nrn_dst.tif',file),nrn_dst)
    
    for edge in G.edges:
        for keys in G[edge[0]][edge[1]]:
            if type(G[edge[0]][edge[1]][keys]) == str:
                G[edge[0]][edge[1]][keys] = 0
        path = graph[edge[0]][edge[1]]['pts']
        path_smooth = np.float32(np.copy(path))  
        for k in range(len(path[0])):
            path_smooth[:,k] = sp.ndimage.gaussian_filter1d(np.float64(path[:,k]),3,mode='nearest')
        path_grad = np.gradient(path_smooth,edge_order=2)[0]
        path_grad = normalize(path_grad,axis=1)
        G[edge[0]][edge[1]]['x'] = np.array([path_grad[:,0].astype(np.float16),
                                             path_grad[:,1].astype(np.float16),
                                             path_grad[:,2].astype(np.float16),
                                             np.array(G[edge[0]][edge[1]]['path_weights']).astype(np.float16)/100,
                                             img_resized[path[:,2],path[:,1],path[:,0]].astype(np.float16)/1024])
        G[edge[0]][edge[1]]['x'][np.isnan(G[edge[0]][edge[1]]['x'])] = 0
        G[edge[0]][edge[1]]['x'] = np.pad(G[edge[0]][edge[1]]['x'],(0,1000-len(G[edge[0]][edge[1]]['x'][0])), 'constant')[0:4]
        G[edge[0]][edge[1]]['x'] = G[edge[0]][edge[1]]['x'].T.flatten()
        G[edge[0]][edge[1]]['y'] = G[edge[0]][edge[1]]['A_V_C']
    G_line = nx.line_graph(G, create_using=nx.Graph)
    for i in G_line.nodes:
        G_line.nodes[i]['x'] = G[i[0]][i[1]]['x']
        G_line.nodes[i]['y'] = G[i[0]][i[1]]['y']
    data = from_networkx(G_line, group_node_attrs = ['x'])
    data.num_classes = len(np.unique(data.y))
    val_data_list.append(data)
val_loader = DataLoader(val_data_list,batch_size=20, shuffle = True, exclude_keys = ['pts','o','edge_pts','weight','radii','radii_std','vessel_type','vessel_type_std','path_weights','end-0','end-1','radii_0001','radii_0001_std','pts_0001','delta','A_V_C'])

for dataset_val in val_loader:
    dataset_val.num_classes = max(dataset_val.num_classes)
    print(data.num_features)
dataset_val.num_classes = dataset_val.num_classes.tolist()

for dataset in loader:
    dataset.num_classes = max(dataset.num_classes)
    print(data.num_features)
dataset.num_classes = dataset.num_classes.tolist()

print()
print(f'Dataset: {dataset}:')
print('======================')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')

data = dataset  # Get the first graph object.

print()
print(data)
print('===========================================================================================================')

# Gather some statistics about the graph.
print(f'Number of nodes: {data.num_nodes}')
print(f'Number of edges: {data.num_edges}')
print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
#print(f'Number of training nodes: {data.train_mask.sum()}')
#print(f'Training node label rate: {int(data.train_mask.sum()) / data.num_nodes:.2f}')
print(f'Has isolated nodes: {data.has_isolated_nodes()}')
print(f'Has self-loops: {data.has_self_loops()}')
print(f'Is undirected: {data.is_undirected()}')

from torch_geometric.nn import GCNConv
import torch
from torch.nn import Linear
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

model = GraphSAGE(in_channels = 4000,
                  hidden_channels=params['hidden_channels'],
                  num_layers = params['num_layers'],
                  out_channels = 3,
                  dropout = params['dropout'],
                  aggr = params['aggr'],
                  jk='lstm')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
print(model)


criterion = torch.nn.CrossEntropyLoss(weight = torch.tensor(compute_class_weight(class_weight='balanced',classes=np.unique(dataset_val.y.detach().numpy()),y=dataset_val.y.detach().numpy())).type(torch.float).to(device))
optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
def train(loader):
    loss_end = 0
    for dataset in loader:
        dataset = dataset.sort(sort_by_row=False)
        optimizer.zero_grad()
        out = model(dataset.x.to(device), dataset.edge_index.to(device))
        loss = criterion(out, dataset.y.type(torch.LongTensor).to(device))
        loss.backward()
        optimizer.step()
        loss_end += loss
    return loss_end, out
epochs = range(1, 100001)
losses = []
embeddings = []
randScore = []
class_report = []
loss_old = 0
counter = 0
for epoch in tqdm(epochs):
    loss, h = train(loader)
    losses.append(loss.cpu().detach().numpy())
    #embeddings.append(h)
    if loss > loss_old:
        counter += 1
    else:
        counter = 0
        torch.save(model.state_dict(), 
                   re.sub('hyperparameter_pickle_files','GCN_models',param_file))
    loss_old = loss
    if counter > 100:
        break
    if epoch%10==0:
        model.eval()
        dataset_val = dataset_val.sort(sort_by_row=False)
        out = model(dataset_val.x.to(device), dataset_val.edge_index.to(device))
        randScore.append(rand_score(dataset_val.y.detach().numpy(),np.argmax(out.cpu().detach().numpy(),axis=1)))
        class_report.append(classification_report(dataset_val.y.detach().numpy(),np.argmax(out.cpu().detach().numpy(),axis=1)))
        model.train()
        np.save(re.sub('.pickle','_randScore.npy',re.sub('hyperparameter_pickle_files','GCN_metrics',param_file)),np.array(randScore))
        np.save(re.sub('.pickle','_classRepo.npy',re.sub('hyperparameter_pickle_files','GCN_metrics',param_file)),np.array(class_report))
        np.save(re.sub('.pickle','_losses.npy',re.sub('hyperparameter_pickle_files','GCN_metrics',param_file)),np.array(losses))
