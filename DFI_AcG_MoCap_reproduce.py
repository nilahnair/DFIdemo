import torch 
import pickle
import os
import sys
import numpy as np
from sacred import Experiment
from sacred.observers import MongoObserver
from torchinfo import summary
import xml.etree.ElementTree as ET
#from xml.dom import minido


from observer import create_observer 
from MoCapDataset import MoCapDataset
from torch.utils.data import Dataset, DataLoader
from LARa_Identificator import Identificator
from sklearn.metrics import f1_score, accuracy_score

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ex = Experiment('MoCap - Demo - Reproduce DFI_AcG Network Results')
# Create MongoObserver and append it to ex.observers
ex.observers.append(create_observer())

from LARa_Identificator import Identificator
from model.Generator.MoCapGenerator import MoCapGenerator
from model.Discriminator.MoCapDiscriminator import ImuLaraDiscriminator

from torch.utils.data import Dataset, DataLoader
from MoCapDataset import MoCapDataset
from MoCap_sub_mean import calculate_subject_means

## Avoid randomness as good as possible 
seed = 42 
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed) 

latent_dim = 32
epochs = 5


@ex.config
def reproduce_DFI_AcG_network_config():
    id_network_path = '/data/nnair/demo/networks/id_cnnimu_mocap_all.pt' 
    autoencoder_network_path ='/data/nnair/demo/autoencoder/autoencoder_MoCap - Train Autoencoder_4165/autoencoder.pth'
    AcG_network_path= '/home/nnair/DFIdemo/DFI_AcG_MoCap.pth'
    AcG_state_network_path= '/home/nnair/DFIdemo/DFI_AcG_MoCap_state_dict.pth'

    generator_conf = {
        "padding_input": (10, 0),
        "padding_output": (5, 0),
        "bias": True,
        "kernel": (5, 1),
        "stride": (2, 1),
        "channels": [1, 2, 2, 2, 2, 2, 1],
        "num_classes": 24,
        "disable_embedding": False,
    }

@ex.automain
def reproduce_DFI_AcG_network():
    
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

    ############use this
    gan_loc = '/home/nnair/DFIdemo/DFI_AcG_MoCap.pth'
    gan = torch.load(gan_loc)
    gan.eval()
    ####################################
    
    ae = torch.load('/data/nnair/demo/autoencoder/autoencoder_MoCap - Train Autoencoder_4165/autoencoder.pth')
    ae.eval()
    
    identificator_loc = '/data/nnair/demo/networks/id_cnnimu_mocap_all.pt'
    identificator_network_saved = torch.load(identificator_loc)
    identificator_network_config = identificator_network_saved['network_config']
    identificator_network_config['fully_convolutional'] = 'FC'
    identificator_network_config['dataset'] = 'mocap'
    identificator = Identificator(identificator_network_config).to(device)
    identificator.load_state_dict(identificator_network_saved['state_dict'])
    identificator = identificator.eval()
    ##############use this
    test_ds = MoCapDataset('/data/nnair/demo/prepros/mocap/test.csv', '/data/nnair/demo/prepros/mocap/sequences_test')
    test_dataloader = DataLoader(test_ds, batch_size=16, shuffle=True)

    
    subject_means = {}

    for i in range(generator_conf['num_classes']):
        path = f"/data/nnair/demo/MoCap_Subject_Averages/{i}_average_tensor"
        subject_means[i] = torch.load(path)

    total_subject_mean = torch.tensor([]).to(device)
    for i in subject_means.keys():
        total_subj_mean = torch.cat((total_subject_mean, subject_means[i].unsqueeze(0)), dim=0)
    r = torch.mean(total_subj_mean, dim=0)
    subject_means_list = [subject_means[key].squeeze(0) for key in sorted(subject_means.keys())]
    subject_means_tensor = torch.stack(subject_means_list)
    ##################################################
    print("Init Done")
    summary(ae, (1, 1, 200, 126))
    
    num_runs = 5

    f1_id_runs, acc_id_runs = [], []
    f1_id_baseline_runs, acc_id_baseline_runs = [], []
    f1_id_ae_runs, acc_id_ae_runs = [], []

    #############use this
    for run in range(num_runs):
        with torch.no_grad():
            # Initialize lists to store results for each run
            id_label, id_real, id_ae, id_gan = [], [], [], []

            for i, X in enumerate(test_dataloader): 
                X, y_id = X['data'].to(device).to(torch.float32), X['label'].to(device).long()

                # generate a random vector of batch size from the latent space 
                rand_vec = torch.randn((X.size(0), 1, 1, 32)).to(device)
                rand_id_targets = torch.empty(y_id.shape, dtype=torch.long).to(device) ###<------------- to be modified to use the selected subject
                for i in range(y_id.shape[0]):
                    while True:
                        rand_value = torch.randint(0, 16, (1,)).to(device)  # Generate a random value between 0 and 7
                        if rand_value != y_id[i]:
                            rand_id_targets[i] = rand_value
                            break
                

                selected_subject_means = subject_means_tensor[y_id]
                sample_encoded = gan.encode(X)
                sample_encoded_anon = sample_encoded - selected_subject_means
                X_fake = gan.dfi(sample_encoded_anon, rand_vec, rand_id_targets)
                ###########################################################
            
                # Append predictions for each model
                id_label.append(y_id.cpu().numpy())
                id_real.append(torch.argmax(identificator(X), dim=1).cpu().numpy())
                id_ae.append(torch.argmax(identificator(ae(X)), dim=1).cpu().numpy())
                id_gan.append(torch.argmax(identificator(X_fake), dim=1).cpu().numpy())


            # Concatenate results
            id_label, id_real, id_ae, id_gan = map(lambda lst: np.concatenate(lst, axis=0), [id_label, id_real, id_ae, id_gan])
        
            # Store results in a dictionary
            res = {
                'id_label': id_label,
                'id_real': id_real,
                'id_ae': id_ae,
                'id_gan': id_gan,
                }


            f1_id = f1_score(res['id_label'], res['id_gan'], average='weighted')
            acc_id = accuracy_score(res['id_label'], res['id_gan'])

            # Calculate baseline comparison metrics
            f1_id_baseline = f1_score(res['id_real'], res['id_gan'], average='weighted')
            acc_id_baseline = accuracy_score(res['id_real'], res['id_gan'])

            # Calculate AE comparison metrics
            f1_id_ae = f1_score(res['id_ae'], res['id_gan'], average='weighted')
            acc_id_ae = accuracy_score(res['id_ae'], res['id_gan'])

            # Append metrics to the lists
            f1_id_runs.append(f1_id)
            acc_id_runs.append(acc_id)

            f1_id_baseline_runs.append(f1_id_baseline)
            acc_id_baseline_runs.append(acc_id_baseline)

            f1_id_ae_runs.append(f1_id_ae)
            acc_id_ae_runs.append(acc_id_ae)
    
    npz_file = '/home/nnair/DFIdemo/test.npz'
    np.savez(npz_file, real=X.detach().cpu().numpy(), fake=X_fake.detach().cpu().numpy())

    # Calculate average and standard deviation for each metric
    # Metrics against annotated labels
    avg_f1_id, std_f1_id = np.mean(f1_id_runs), np.std(f1_id_runs)
    avg_acc_id, std_acc_id = np.mean(acc_id_runs), np.std(acc_id_runs)

    # Metrics against baseline
    avg_f1_id_baseline, std_f1_id_baseline = np.mean(f1_id_baseline_runs), np.std(f1_id_baseline_runs)
    avg_acc_id_baseline, std_acc_id_baseline = np.mean(acc_id_baseline_runs), np.std(acc_id_baseline_runs)

    # Metrics against AE
    avg_f1_id_ae, std_f1_id_ae = np.mean(f1_id_ae_runs), np.std(f1_id_ae_runs)
    avg_acc_id_ae, std_acc_id_ae = np.mean(acc_id_ae_runs), np.std(acc_id_ae_runs)

    

    # Print the results for each alpha
    print(f"DFI_AcG MoCap")
    print(f"Average F1 ID (Label vs GAN): {avg_f1_id:.6f}, Std Dev: {std_f1_id:.6f}")
    print(f"Average Accuracy ID (Label vs GAN): {avg_acc_id:.6f}, Std Dev: {std_acc_id:.6f}")
    print()
    print(f"Average F1 ID (Real vs GAN): {avg_f1_id_baseline:.6f}, Std Dev: {std_f1_id_baseline:.6f}")
    print(f"Average Accuracy ID (Real vs GAN): {avg_acc_id_baseline:.6f}, Std Dev: {std_acc_id_baseline:.6f}")
    print()
    print(f"Average F1 ID (AE vs GAN): {avg_f1_id_ae:.6f}, Std Dev: {std_f1_id_ae:.6f}")
    print(f"Average Accuracy ID (AE vs GAN): {avg_acc_id_ae:.6f}, Std Dev: {std_acc_id_ae:.6f}")
    print()

