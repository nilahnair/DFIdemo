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
from MoCap_sub_mean import calculate_subject_means

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print(f"Using Cuda: {torch.cuda.is_available()}")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    subject_data, subject_counts = calculate_subject_means()
    print(subject_data)
    print(subject_counts)
    print(subject_data[1].shape)
    sum(subject_counts.values())
    base_dir = "/data/nnair/demo/MoCap_Subject_Averages"
    for subject_id, subject_mean in subject_data.items(): 
        torch.save(subject_mean, f"{base_dir}/{subject_id}_average_tensor")

if __name__ == "__main__":
    main()