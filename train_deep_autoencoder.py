
import sys

sys.path.append("/home/thallybu/ICPR_2024_IMU_AnonGAN/code")
sys.path.append("/home/thallybu/Anon_GAN_SD")

import torch 
import os
import numpy as np
from sacred import Experiment
from model.Autoencoder.DeepConvolutionalAutoencoder import DeepConvolutionalAutoencoder
from util.plot_motionsense import plot_sample, compare_samples
from torchinfo import summary

from observer import create_observer 
from dataset.MotionSenseDataset import MotionSenseDataset
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score

ex = Experiment('MotionSense - ICPR2024 - Train Deep Autoencoder')
ex.observers.append(create_observer())

@ex.config
def train_autoencoder_cofig():

    host_conf = {
        "GPU": "7",
    }
    
    training_conf = {
        "learning_rate": 1e-4,
        "learning_threshold": 1e-9,
        "batch_size": 16,
        "dataset_path": 'model/cnn_motionsense_id.pt',

        # note: the experiment ID will be used to create a new subdirectory under this path
        "network_output_base_path": '/data/thallybu/ICPR_2024/autoencoder_trained_output/',
        "network_output_name": 'autoencoder',
    }

    autoencoder_conf = {
        "padding_input": (0, 0),
        "padding_output": (0, 0),
        "bias": False,
        "kernel": (3, 1),
        "stride": (1, 1),
        "channels": [1, 2, 4, 4, 5, 5, 5, 4, 4, 2, 1],
        
        # Writing those in the config for documentation, rather than flexibility
        "model": "convAutoencoder",
        "loss_function": "MSE",
        "optimizer": "ADAM"
    }
    

def train(model, optimizer, loss_fn, dataloader, device):
    model.train()
    running_loss = 0 
    train_loss_history = []
    
    for X in dataloader:
        X, y = X['data'].to(device).to(torch.float32), X['label']
        X_reconstructed = model(X)        
        optimizer.zero_grad()
        loss = loss_fn(X_reconstructed, X)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        train_loss_history.append(loss.item())
    ex.log_scalar('train_loss_avg', running_loss / len(dataloader))
    return running_loss / len(dataloader), train_loss_history
        
def validate(model, loss_fn, dataloader, device):
    model.eval()
    running_loss = 0 
    val_loss_history = []
        
    for X in dataloader:
        X, y = X['data'].to(device).to(torch.float32), X['label']
        X_reconstructed = model(X)        
        loss = loss_fn(X_reconstructed, X)
        running_loss += loss.item()
        val_loss_history.append(loss.item())
    ex.log_scalar('val_loss_avg', running_loss / len(dataloader))
    return running_loss / len(dataloader), val_loss_history

def test(model, loss_fn, dataloader, device): 
    model.eval()
    running_loss = 0 
        
    for X in dataloader:
        X, y = X['data'].to(device).to(torch.float32), X['label']
        X_reconstructed = model(X)        
        loss = loss_fn(X_reconstructed, X)
        running_loss += loss.item()
        
    ex.log_scalar('test_loss_avg', running_loss / len(dataloader))
    return running_loss / len(dataloader)

@ex.capture
def save_model(_run, training_conf, model):
    output_dir = f"{training_conf['network_output_base_path']}autoencoder_{_run.experiment_info['name']}_{_run._id}/"
    os.makedirs(output_dir, exist_ok=True)
    torch.save(model, output_dir + 'autoencoder.pth')

@ex.capture
def generate_samples(_run, training_conf, model, epoch, val_dataloader, device):
    model.eval()
    batch_to_plot = next(iter(val_dataloader))
    with torch.no_grad():
        X, y = batch_to_plot['data'].to(device).to(torch.float32), batch_to_plot['label']
        X_reconstructed = model(X)
        
        output_dir = f"{training_conf['network_output_base_path']}samples_{_run.experiment_info['name']}_{_run._id}/epoch_{epoch}"
        os.makedirs(output_dir, exist_ok=True)
        sample_files = compare_samples(X.cpu(), X_reconstructed.cpu(), output_dir)

        for sample in sample_files:
            ex.add_artifact(sample)


@ex.automain
def train_autoencoder_e2e(host_conf, training_conf, autoencoder_conf):
    ### Setup
    train_loss = []
    val_loss = []
    test_loss = []

    os.environ["CUDA_VISIBLE_DEVICES"] = host_conf['GPU']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    autoencoder = DeepConvolutionalAutoencoder(autoencoder_conf)
    autoencoder.to(device)
    summary(autoencoder, (1, 1, 200, 9))
    autoencoder_optimizer = torch.optim.Adam(autoencoder.parameters(), lr=training_conf['learning_rate'])
    autoencoder_loss_fn = torch.nn.MSELoss()

    train_ds = MotionSenseDataset('/data/thallybu/ICPR_2024/data/motionsense/train.csv', '/data/thallybu/ICPR_2024/data/motionsense/sequences_train')
    train_dataloader = DataLoader(train_ds, batch_size=training_conf['batch_size'], shuffle=True)
    val_ds = MotionSenseDataset('/data/thallybu/ICPR_2024/data/motionsense/val.csv', '/data/thallybu/ICPR_2024/data/motionsense/sequences_val')
    val_dataloader = DataLoader(val_ds, batch_size=training_conf['batch_size'], shuffle=True)
    test_ds = MotionSenseDataset('/data/thallybu/ICPR_2024/data/motionsense/test.csv', '/data/thallybu/ICPR_2024/data/motionsense/sequences_test')
    test_dataloader = DataLoader(test_ds, batch_size=training_conf['batch_size'], shuffle=True)


    ### Training 

    prev_loss = float('inf')
    epoch = 0
    epoch_loss = 0

    while True:    
        train_running_loss, train_loss_history = train(autoencoder, autoencoder_optimizer, autoencoder_loss_fn, train_dataloader, device)
        val_running_loss, val_loss_history = validate(autoencoder, autoencoder_loss_fn, val_dataloader, device)
        generate_samples(model=autoencoder, epoch=epoch, val_dataloader=val_dataloader, device=device)
        epoch_loss = val_running_loss
        if (prev_loss - epoch_loss) < training_conf['learning_threshold']:
            break
        prev_loss = epoch_loss
        epoch += 1


    ### Testing
    test_running_loss = test(autoencoder, autoencoder_loss_fn, test_dataloader, device)

    ### Teardown
    save_model(model=autoencoder)

    return test_running_loss

