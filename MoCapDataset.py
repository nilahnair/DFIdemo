import os
import csv 
import pandas as pd
import pickle

from torch.utils.data import Dataset, DataLoader

class MoCapDataset(Dataset):
    """Dataset for IMU samples. Data is loaded according to LaRA preprocessing.
    label: 1 for a true (real) sample, 0 for a disorted counter example
    """
    
    def __init__(self, annotation_file, root_dir): 
        self.annotation_file = pd.read_csv(annotation_file)
        self.root_dir = root_dir
    
    def __len__(self):
        '''
        Get the dataset length.
        This method is in accordance to the pytorch dataloader specification

        @return total_length: int of |N^+ which equals the size of the dataset. (Who would have thought?!)
        '''
        return len(self.annotation_file)

    def __getitem__(self, idx):
        '''
        Get a single item from the dataset.
        This method is in accordance to the pytorch dataloader specification

        @param data: index of item in List
        @return window_data: dict with sequence window, label of window, and labels of each sample in window
        '''
        window_name = os.path.join(self.root_dir, self.annotation_file.iloc[idx, 0])

        with open(window_name, 'rb') as f: 
            data = pickle.load(f, encoding='bytes')
        
        return data