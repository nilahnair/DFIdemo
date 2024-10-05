import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler

import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support, f1_score
from torchinfo import summary
from sacred import Experiment
from observer import create_observer 
import sys
import os
import pickle
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print(f"Using Cuda: {torch.cuda.is_available()}")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from LARa_Identificator import Identificator
from model.Generator.MoCapGenerator import MoCapGenerator
from model.Discriminator.MoCapDiscriminator import ImuLaraDiscriminator

from torch.utils.data import Dataset, DataLoader
from MoCapDataset import MoCapDataset
from MoCap_sub_mean import calculate_subject_means

ex = Experiment('MoCap - Train DFI_AcG')
ex.observers.append(create_observer())

def pselect(X, y):
    return X[range(X.shape[0]), y.to(torch.int64)]

def generator_loss_fn(disc_preds, id_preds, y_id):
    id_accurate_probabilities = pselect(id_preds, y_id)
    id_accurate_probabilities = 0.9 * id_accurate_probabilities + 0.1 
    print('id prob')
    print(id_accurate_probabilities)
    return F.binary_cross_entropy(1 - disc_preds, torch.zeros_like(disc_preds)) * id_accurate_probabilities

def train(dataloader, g_model, g_optimizer, g_loss_fn, d_model, d_optimizer, d_loss_fn, i_model, i_loss_fn, device, epoch, subject_means_tensor):
    
    sys.stdout.flush()
    size = len(dataloader.dataset)
    latent_dim = 32
    # only the generator and discriminator are trained, the identificator is fixed
    g_model.train()
    d_model.train()
    i_model.eval()
    
    full_g_loss = torch.tensor([]).to(device)
    full_d_loss = torch.tensor([]).to(device)
    full_i_loss = torch.tensor([]).to(device)
    
    # these layers will be freezed  
    encoder_params_to_freeze = ["encoder.0.weight", "encoder.2.weight", "encoder.3.weight", "encoder.3.bias", "encoder.5.weight", "encoder.6.weight", "encoder.6.bias"]
    # these layers will be toggle-freezed
    layers_to_toggle = ["bottleneck.0.weight", "decoder.0.weight", "decoder.1.weight", "decoder.1.bias", "decoder.4.weight", "decoder.5.weight", "decoder.5.bias", "decoder.8.weight"]
    
    # freeze the encoder - it should never change
    for name, param in g_model.named_parameters():
        if name in encoder_params_to_freeze:
            param.requires_grad = False 

    for batch, X in enumerate(dataloader):
        
        
        #forward pass with a batch
        #compute structural loss
        #discriminator loss ( negative)
        #identification loss ( negative)
        #total loss
        #backward pass
        #freeze the encoder
        #forward pass with the same batch
        #compute structural loss
        #discriminator loss ( negative)
        #identification loss ( positive) with target any other person ( not the target one)
        #total loss
        #backward pass
        
        
        
        # pick a batch from the dataloader
        X, y = X['data'].to(device).to(torch.float32), X['label'].to(device).long()

        # generate a random vector of batch size from the latent space 
        rand_vec = torch.randn((X.size(0), 1, 1, 32)).to(device)
        rand_id_targets = torch.randint(0, 16, y.shape).to(device)
        
        selected_subject_means = subject_means_tensor[y]
        sample_encoded = g_model.encode(X)
        print('Sample_encode')
        print(sample_encoded.shape)
        print('subjectmean')
        print(selected_subject_means.shape)
        sample_encoded_anon = sample_encoded - selected_subject_means
        X_fake = g_model.dfi(sample_encoded_anon, rand_vec, rand_id_targets)
            
        d_preds_real = d_model(X, y)
        d_loss_real = d_loss_fn(d_preds_real.view(-1), torch.ones_like(y).to(torch.float32))
        d_preds_fake = d_model(X_fake.detach(), rand_id_targets)
        d_loss_fake = d_loss_fn(d_preds_fake.view(-1), torch.zeros_like(y).to(torch.float32))

        d_optimizer.zero_grad()   
        discriminator_loss = d_loss_real + d_loss_fake 
        discriminator_loss.backward()
        d_optimizer.step()

        identificator_preds_fake = i_model(X_fake)
        identificator_loss = i_loss_fn(identificator_preds_fake, rand_id_targets)
        
        g_optimizer.zero_grad()
        d_model_preds= d_model(X_fake, rand_id_targets)
        print('dmodel preds shape')
        print(d_model_preds)
        print('identificator preds shape')
        print(identificator_preds_fake)
        print('random subs shape')
        print(rand_id_targets)
        generator_loss = g_loss_fn(d_model_preds, identificator_preds_fake, rand_id_targets)
        generator_loss = torch.mean(generator_loss)
        generator_loss.backward()
        g_optimizer.step()
        
        # backward pass 
        selected_subject_means = subject_means_tensor[y]
        sample_encoded = g_model.encode(X.detach())
        sample_encoded_anon = sample_encoded - selected_subject_means
        X_unfake = g_model.dfi(sample_encoded_anon, rand_vec, y)
        g_optimizer.zero_grad()
        mse_loss = F.mse_loss(X_unfake, X.detach())
        mse_loss.backward()
        g_optimizer.step()
        
                
        
        if batch % 50 == 0:
            
            with torch.no_grad():
                #d_preds_real = d_model(ae(X))
                #d_preds_fake = d_model(X_fake.detach())
                #d_correct_real = (d_preds_real.view(-1) > 0.5).float().eq(1).sum().item()
                #d_correct_fake = (d_preds_fake.view(-1) < 0.5).float().eq(1).sum().item()
                #d_accuracy = 100.0 * (d_correct_real + d_correct_fake) / (2 * X.size(0))


                #i_preds_fake = i_model(X_fake.detach())
                #i_correct = (torch.argmax(i_preds_fake, dim=1) == y).sum().item()
                #i_accuracy = 100.0 * i_correct / X.size(0)
                d_accuracy = 0
                i_accuracy = 0
                uniq_preds = len(torch.unique(torch.argmax(i_model(X_fake), dim=1)))
               

                d_loss, i_loss, g_loss, current = discriminator_loss.item(), identificator_loss.item(), generator_loss.item(), (batch + 1) * len(X)
                #print(f"d_loss: {d_loss:>7f} i_loss: {i_loss:>7f}  g_loss: {g_loss_total:>7f}  ({g_loss_i:>7f} + {g_loss_d:>7f})  preds: {uniq_preds}    [{current:>5d}/{size:>5d}]")
                print(f"d_loss: {d_loss:>7f} d_acc: {d_accuracy:.2f}% i_loss: {i_loss:>7f} i_acc: {i_accuracy:.2f}% g_loss: {g_loss:>7f}   preds: {uniq_preds}  epoch: {epoch}  [{current:>5d}/{size:>5d}]")

                sys.stdout.flush()
        
        
    print("Epoch finished")
    return full_g_loss, full_d_loss, full_i_loss



def validate(dataloader, g_model, g_loss_fn, d_model, d_loss_fn, i_model, i_loss_fn, epoch, subject_means_tensor, output_path, generator_loss_history, discriminator_loss_history, identificator_loss_history, device, disable_network_save=False, seed=42):
    ## Testing process:
    ## - save the current networks with respect to the epoch
    ## - create a few synthetic samples and save them next to the real counter parts (for later analysis and visualization)
    ## - keep track of the loss history for the previous epoch
    ## - (run HAR benchmark?)
    ## - save the identificator and discriminator predictions for a single batch given both real and synthetic data
    ## - On the complete set: calculate key performance indicators such as f1 score etc 

    # make sure all models are in evaluation mode to avoid dropout etc and disable gradients too
    g_model.eval()
    d_model.eval()
    i_model.eval()

    summary = {}
    
    running_g_loss = 0
    running_d_loss = 0
    running_i_loss = 0

    with torch.no_grad():
        for batch, X in enumerate(dataloader):
            
            # pick a batch from the dataloader
            X, y = X['data'].to(device).to(torch.float32), X['label'].to(device).long()

            # generate a random vector of batch size from the latent space 
            rand_vec = torch.randn((X.size(0), 1, 1, 32)).to(device)
            rand_id_targets = torch.randint(0, 16, y.shape).to(device)

            selected_subject_means = subject_means_tensor[y]
            sample_encoded = g_model.encode(X)
            sample_encoded_anon = sample_encoded - selected_subject_means
            X_fake = g_model.dfi(sample_encoded_anon, rand_vec, rand_id_targets)


            d_preds_real = d_model(X, y)
            d_loss_real = d_loss_fn(d_preds_real.view(-1), torch.ones_like(y).to(torch.float32))
            d_preds_fake = d_model(X_fake.detach(), rand_id_targets)
            d_loss_fake = d_loss_fn(d_preds_fake.view(-1), torch.zeros_like(y).to(torch.float32))
            discriminator_loss = d_loss_real + d_loss_fake 
            
            identificator_preds_fake = i_model(X_fake)
            identificator_loss = i_loss_fn(identificator_preds_fake, rand_id_targets)

            generator_loss = g_loss_fn(d_model(X_fake, rand_id_targets), identificator_preds_fake, rand_id_targets)
            generator_loss = torch.mean(generator_loss)
            
            running_d_loss = running_d_loss + discriminator_loss.item()
            running_i_loss = running_i_loss + identificator_loss.item()
            running_g_loss = running_g_loss + generator_loss.item()
            
            uniq_preds = len(torch.unique(torch.argmax(i_model(X_fake), dim=1)))

    
    print(f"Validate epoch {epoch}  d_loss: {running_d_loss / len(dataloader)}  i_loss: {running_i_loss / len(dataloader)}  g_loss: {running_g_loss / len(dataloader)}  preds: {uniq_preds}  epoch: {epoch}")

@ex.config
def train_cofig():

    print('check 1')

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

    latent_dim = 32
    epochs = 5

    generator_conf = {
        "padding_input": (5, 0),
        "padding_output": (3, 0),
        "bias": True,
        "kernel": (4, 4),
        "stride": (2, 1),
        "channels": [1, 16, 32, 64, 32, 16, 1],
        "num_classes": 16,
        "disable_embedding": False,
    }
    

@ex.automain
def train_AcG():
    print('check 2')
    ## Avoid randomness as good as possible 
    seed = 42 
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 

    generator_conf = {
        "padding_input": (5, 0),
        "padding_output": (3, 0),
        "bias": True,
        "kernel": (4, 4),
        "stride": (2, 1),
        "channels": [1, 16, 32, 64, 32, 16, 1],
        "num_classes": 16,
        "disable_embedding": False,
    }
   
    generator = MoCapGenerator(generator_conf).to(device) 
    generator.load_pretrained_ae('/data/nnair/demo/autoencoder/autoencoder_MoCap - Train Autoencoder_4165/autoencoder.pth')
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=1e-6)
    generator_scheduler = torch.optim.lr_scheduler.ExponentialLR(generator_optimizer, gamma=0.95)

    discriminator = ImuLaraDiscriminator(generator_conf).to(device)
    # try with generator params too here 
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=5e-7)
    discriminator_loss_fn = torch.nn.BCELoss()
    discriminator_scheduler = torch.optim.lr_scheduler.ExponentialLR(discriminator_optimizer, gamma=0.80)

    identificator_loc = '/data/nnair/demo/networks/id_cnnimu_mocap_all.pt'
    identificator_network_saved = torch.load(identificator_loc)
    identificator_network_config = identificator_network_saved['network_config']
    identificator_network_config['fully_convolutional'] = 'FC'
    identificator_network_config['dataset'] = 'mocap'
    identificator = Identificator(identificator_network_config).to(device)
    identificator.load_state_dict(identificator_network_saved['state_dict'])
    identificator = identificator.eval()

    print(identificator)

    #avg_sub, sub_count=calculate_subject_means()
    #print(avg_sub.type)
    #print(sub_count.type)
    subject_means = {}

    for i in range(generator_conf['num_classes']):
        path = f"/data/nnair/demo/MoCap_Subject_Averages/{i}_average_tensor"
        subject_means[i] = torch.load(path)

    print("Init Done")

    total_subject_mean = torch.tensor([]).to(device)
    for i in subject_means.keys():
        total_subj_mean = torch.cat((total_subject_mean, subject_means[i].unsqueeze(0)), dim=0)
    r = torch.mean(total_subj_mean, dim=0)
    subject_means_list = [subject_means[key].squeeze(0) for key in sorted(subject_means.keys())]
    subject_means_tensor = torch.stack(subject_means_list)
    print(subject_means_tensor[0].shape)

    train_ds = MoCapDataset('/data/nnair/demo/prepros/mocap/train.csv', '/data/nnair/demo/prepros/mocap/sequences_train')
    train_dataloader = DataLoader(train_ds, batch_size=16, shuffle=True)

    val_ds = MoCapDataset('/data/nnair/demo/prepros/mocap/val.csv', '/data/nnair/demo/prepros/mocap/sequences_val')
    val_dataloader = DataLoader(val_ds, batch_size=16, shuffle=True)

    identificator_loss_fn = torch.nn.CrossEntropyLoss()
    generator_loss_fn = torch.nn.BCELoss()

    epochs = 5

    for e in range(epochs):
        print(f"Epoch {e} / {epochs} ======================")
        train(train_dataloader, generator, generator_optimizer, generator_loss_fn, discriminator, discriminator_optimizer, discriminator_loss_fn, identificator, identificator_loss_fn, device, e+1, subject_means_tensor)
        validate(val_dataloader, generator, generator_loss_fn, discriminator, discriminator_loss_fn, identificator, identificator_loss_fn, e + 1, '', None, None, None, device, subject_means_tensor)
        generator_scheduler.step()
        discriminator_scheduler.step()    
        # torch.save(generator, f"mbientlab_condigan_211223_e{e}.pth")

    torch.save(generator, 'DFI_AcG_MoCap.pth')
    torch.save(generator.state_dict(), 'DFI_AcG_MoCap_state_dict.pth')

    generator.eval()
    X = next(iter(val_dataloader))
    X, y = X['data'].to(device).to(torch.float32), X['label'].to(device).long()
    rand_vec = torch.randn((X.size(0), 1, 1, 32)).to(device)
    rand_id_targets = torch.randint(0, 24, y.shape).to(device)

    selected_subject_means = subject_means_tensor[y]
    sample_encoded = generator.encode(X)
    sample_encoded_anon = sample_encoded - selected_subject_means
    X_fake = generator.dfi(sample_encoded_anon, rand_vec, rand_id_targets)
    print(X[0])
    print(X_fake[0])

    