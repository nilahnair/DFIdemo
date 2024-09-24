import torch 
import os
import sys
from sacred import Experiment

sys.path.append("/home/thallybu/ICPR_2024_IMU_AnonGAN/code")
sys.path.append("/home/thallybu/Anon_GAN_SD")
from model.LARa_Identificator import Identificator
from observer import create_observer 
from dataset.MotionSenseDataset import MotionSenseDataset
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score


ex = Experiment('MotionSense - ICPR2024 - Benchmark HAR')
ex.observers.append(create_observer())


@ex.config
def benchmark_har_config():
    host_conf = {
        "GPU": "7",
    }
    benchmark_conf = {
        "model_path": "/data/thallybu/ICPR_2024/autoencoder_trained_output/autoencoder_MotionSense - ICPR2024 - Train Autoencoder_26/autoencoder.pth",
        "act_model_path": "/home/thallybu/ICPR_2024_IMU_AnonGAN/code/model/cnn_motionsense_act.pt"
    }
    
@ex.automain
def reproduce_act_network(host_conf, benchmark_conf):
    os.environ["CUDA_VISIBLE_DEVICES"] = host_conf['GPU']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = torch.load(benchmark_conf['model_path'])
    model.to(device)
    model.eval()
    
    benchmark_loc = benchmark_conf['act_model_path']
    benchmark_network_saved = torch.load(benchmark_loc)
    benchmark_network_config = benchmark_network_saved['network_config']
    benchmark_network_config['fully_convolutional'] = 'FC'
    benchmark_network_config['dataset'] = 'motionsense'
    benchmark = Identificator(benchmark_network_config).to(device)
    benchmark.load_state_dict(benchmark_network_saved['state_dict'])
    benchmark = benchmark.eval()

    test_ds = MotionSenseDataset('/data/thallybu/ICPR_2024/data/motionsense/test.csv', '/data/thallybu/ICPR_2024/data/motionsense/sequences_test')
    test_dataloader = DataLoader(test_ds, batch_size=1, shuffle=True)
    
    all_model_predictions = [] # predictions for the model to test (generator, autoencoder etc)
    all_benchmark_predictions = [] # predictions of the fixed HAR model
    all_labels = [] # true labels (ground truth)
    
    for X in test_dataloader:
        X, y = X['data'].to(device).to(torch.float32), X['act_label']
        
        with torch.no_grad():
            output_benchmark = benchmark(X)
            output_model = benchmark(model(X))
        
        predictions_benchmark = torch.argmax(output_benchmark, dim=1).cpu().numpy()
        all_benchmark_predictions.extend(predictions_benchmark)
        predictions_model = torch.argmax(output_model, dim=1).cpu().numpy()
        all_model_predictions.extend(predictions_model)
        all_labels.extend(y.cpu().numpy())

    # Calculate metrics
    f1_benchmark = f1_score(all_labels, all_benchmark_predictions, average='weighted')
    accuracy_benchmark = accuracy_score(all_labels, all_benchmark_predictions)
    ex.log_scalar('wF1 Score Label vs Benchmark (test)', f1_benchmark)
    ex.log_scalar('Accuracy Label vs Benchmark (test)', accuracy_benchmark)

    f1_model = f1_score(all_labels, all_model_predictions, average='weighted')
    accuracy_model = accuracy_score(all_labels, all_model_predictions)
    ex.log_scalar('wF1 Score Label vs Model (test)', f1_model)
    ex.log_scalar('Accuracy Label vs Model (test)', accuracy_model)
    

    return f1_model, accuracy_model
    