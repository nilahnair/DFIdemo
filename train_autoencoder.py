
import sys

import torch 
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import os
import numpy as np
from sacred import Experiment
from model.Autoencoder.AutoencoderExt import AutoencoderExtended
from util.plot_mocap import plot_sample, compare_samples
from torchinfo import summary

from observer import create_observer 
from MoCapDataset import MoCapDataset
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score


ex = Experiment('MoCap - Train Autoencoder')
ex.observers.append(create_observer())

@ex.config
def train_autoencoder_cofig():

    host_conf = {
        "GPU": "0",
    }
    
    training_conf = {
        "learning_rate": 1e-6,
        "learning_threshold": 1e-15,
        "scheduler_step": 0.95,
        "batch_size": 16,
        "dataset_path": '/data/nnair/demo/networks/id_cnnimu_mocap_all.pt',

        # note: the experiment ID will be used to create a new subdirectory under this path
        "network_output_base_path": '/data/nnair/demo/autoencoder/',
        "network_output_name": 'autoencoderext',
    }

    autoencoder_conf = {
        "padding_input": (5, 0),
        "padding_output": (3, 0),
        "bias": True,
        "kernel": (4, 4),
        "stride": (2, 1),
        "channels": [1, 16, 32, 64, 32, 16, 1],
        
        # Writing those in the config for documentation, rather than flexibility
        "model": "AutoencoderExtended",
        "loss_function": "MSE",
        "optimizer": "ADAM"
    }
    

def train(model, optimizer, loss_fn, dataloader, device):
    model.train(mode=True)
    running_loss = 0 
    train_loss_history = []
    
    for X in dataloader:
        x, y = X['data'].to(device).to(torch.float32), X['label']
        # Add noise with a mean of 0 and standard deviation of 0.01
        noise = torch.randn_like(x) * 0.01

        # Add noise to the tensor
        noisy_tensor = x + noise
        # Clamp the noisy tensor to ensure values stay within [0, 1]
        noisy_tensor_clamped = torch.clamp(noisy_tensor, 0, 1)
        X_reconstructed = model(noisy_tensor_clamped)        
        optimizer.zero_grad()
        loss = loss_fn(X_reconstructed, x)
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
    print(f"Scheduler enabled")

    autoencoder = AutoencoderExtended(autoencoder_conf)
    autoencoder.to(device)
    summary(autoencoder, (1, 1, 200, 126))
    autoencoder_optimizer = torch.optim.Adam(autoencoder.parameters(), lr=training_conf['learning_rate'])
    autoencoder_loss_fn = torch.nn.MSELoss()
    autoencoder_scheduler = lr_scheduler.ExponentialLR(autoencoder_optimizer, gamma=training_conf['scheduler_step'])

    train_ds = MoCapDataset('/data/nnair/demo/prepros/mocap/train.csv', '/data/nnair/demo/prepros/mocap/sequences_train')
    train_dataloader = DataLoader(train_ds, batch_size=training_conf['batch_size'], shuffle=True)
    val_ds = MoCapDataset('/data/nnair/demo/prepros/mocap/val.csv', '/data/nnair/demo/prepros/mocap/sequences_val')
    val_dataloader = DataLoader(val_ds, batch_size=training_conf['batch_size'], shuffle=True)
    test_ds = MoCapDataset('/data/nnair/demo/prepros/mocap/test.csv', '/data/nnair/demo/prepros/mocap/sequences_test')
    test_dataloader = DataLoader(test_ds, batch_size=training_conf['batch_size'], shuffle=True)


    ### Training 

    prev_loss = float('inf')
    epoch = 0
    epoch_loss = 0

    while epoch <= 100:
        train_running_loss, train_loss_history = train(autoencoder, autoencoder_optimizer, autoencoder_loss_fn, train_dataloader, device)
        val_running_loss, val_loss_history = validate(autoencoder, autoencoder_loss_fn, val_dataloader, device)
        generate_samples(model=autoencoder, epoch=epoch, val_dataloader=val_dataloader, device=device)
        epoch_loss = val_running_loss
        if (prev_loss - epoch_loss) < training_conf['learning_threshold']:
            break
        prev_loss = epoch_loss
        epoch += 1
        autoencoder_scheduler.step()


    ### Testing
    test_running_loss = test(autoencoder, autoencoder_loss_fn, test_dataloader, device)

    ### Teardown
    save_model(model=autoencoder)

    return test_running_loss

