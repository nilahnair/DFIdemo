import sys
import torch 
import os
import numpy as np
from sacred import Experiment
from model.Autoencoder.AutoencoderExt import AutoencoderExtended
from util.plot_mocap import plot_sample, compare_samples
from torchinfo import summary
from itertools import groupby


from observer import create_observer 
from MoCapDataset import MoCapDataset
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score

    
def calculate_subject_means():

    host_conf = {
        "GPU": "0",
    }
    
    network_conf = {
        "batch_size": 16,
        "dataset_path": '/data/nnair/demo/networks/id_cnnimu_mocap_all.pt',
        "network_input_base_path": '/data/nnair/demo/autoencoder/autoencoder_MoCap - Train Autoencoder_4165',
        "network_input_name": 'autoencoder.pth',
    }
    
    os.environ["CUDA_VISIBLE_DEVICES"] = host_conf['GPU']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    autoencoder_full_path = os.path.join(network_conf["network_input_base_path"], network_conf["network_input_name"])
    autoencoder = torch.load(autoencoder_full_path)
    autoencoder.to(device)
    autoencoder.eval()
    summary(autoencoder, (1, 1, 200, 126))

    train_ds = MoCapDataset('/data/nnair/demo/prepros/mocap/train.csv', '/data/nnair/demo/prepros/mocap/sequences_train')
    train_dataloader = DataLoader(train_ds, batch_size=16, shuffle=False)
    
    subject_tensors = {}
    subject_counts = {}
    
    for idx, batch in enumerate(train_dataloader):
        # Extract inputs and labels from the batch
        X, y = batch['data'].to(device).to(torch.float32), batch['label'].to(device)
        
        # Iterate over each example in the batch
        for i in range(len(X)):
            tensor = autoencoder.encode(X[i].unsqueeze(0))
            label = y[i].item()
            
            # Check if the label matches the desired ID
            if label in subject_tensors:
                
                subject_tensors[label] += tensor
                subject_counts[label] += 1
            else:
                subject_tensors[label] = tensor
                subject_counts[label] = 1

    print(subject_tensors)
    print(subject_counts)
    print('check1')
    # Compute the average tensor for each subject
    average_subject_tensors = {}

    for subject_id, tensors in subject_tensors.items():
        count = subject_counts[subject_id]
        average_tensor = tensors / count
        average_subject_tensors[subject_id] = average_tensor
        print(average_subject_tensors)
        print(subject_counts)
        
    return average_subject_tensors, subject_counts

    