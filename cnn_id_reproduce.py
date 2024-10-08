import torch 
import pickle
import os
import sys
import numpy as np
from sacred import Experiment
from sacred.observers import MongoObserver


from observer import create_observer 
from MoCapDataset import MoCapDataset
from torch.utils.data import Dataset, DataLoader
from LARa_Identificator import Identificator
from sklearn.metrics import f1_score, accuracy_score

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ex = Experiment('MoCap - Demo - Reproduce ID Network Results')
# Create MongoObserver and append it to ex.observers
ex.observers.append(create_observer())

@ex.config
def reproduce_id_network_config():
    network_path = '/data/nnair/demo/networks/id_cnnimu_mocap_all.pt' 

@ex.automain
def reproduce_id_network(network_path):

    identificator_loc = network_path
    identificator_network_saved = torch.load(identificator_loc)
    identificator_network_config = identificator_network_saved['network_config']
    identificator_network_config['fully_convolutional'] = 'FC'
    identificator_network_config['dataset'] = 'mocap'
    identificator = Identificator(identificator_network_config).to(device)
    identificator.load_state_dict(identificator_network_saved['state_dict'])
    identificator = identificator.eval()

    test_ds = MoCapDataset('/data/nnair/demo/prepros/mocap/test.csv', '/data/nnair/demo/prepros/mocap/sequences_test')
    test_dataloader = DataLoader(test_ds, batch_size=1, shuffle=True)
    
    all_predictions = []
    all_labels = []
    
    for batch, X in enumerate(test_dataloader):
        X, y = X['data'].to(device).to(torch.float32), X['label']
        
        with torch.no_grad():
            output = identificator(X)
        
        predictions = torch.argmax(output, dim=1).cpu().numpy()
        all_predictions.extend(predictions)
        all_labels.extend(y.cpu().numpy())

    # Calculate metrics
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    accuracy = accuracy_score(all_labels, all_predictions)
    ex.log_scalar('wF1 Score (test)', f1)
    ex.log_scalar('Accuracy (test)', accuracy)
    return f1, accuracy
    
    