import torch
import torch.nn as nn
import os

#job_id = os.environ.get('SLURM_JOB_ID', 'default_job_id')

class CNN_AE(nn.Module):
    
    def __init__(self, input_size, hidden_size, encoder_layer_sizes, activation_function, final_act_function):
        super(Autoencoder, self).__init__()

        # Create encoder layers
        self.features = nn.Sequential(
            nn.Conv1d(1, 9, 9, stride=1, padding = 1),  
            nn.BatchNorm1d(9),  
            nn.ReLU(),
            nn.MaxPool1d(3, stride=3),
            nn.Conv1d(9, 18, 9, stride=1, padding = 1),
            nn.BatchNorm1d(18), 
            nn.ReLU(),
            nn.MaxPool1d(3, stride=3),
            nn.Conv1d(18, 27, 9, stride=1, padding = 1),
            nn.BatchNorm1d(27), 
            nn.ReLU(),
            nn.MaxPool1d(3, stride=3),
        )

        encoder_layers = []
        prev_size = input_size
        decoder_layer_sizes = encoder_layer_sizes[::-1]
        
        
        for size in encoder_layer_sizes:
            encoder_layers.append(nn.Linear(prev_size, size))
            encoder_layers.append(activation_function())
            prev_size = size
            
        encoder_layers.append(nn.Linear(prev_size, hidden_size))
        encoder_layers.append(activation_function())
        self.encoder = nn.Sequential(*encoder_layers)

        # Create decoder layers
        decoder_layers = []
        prev_size = hidden_size
        
        for size in decoder_layer_sizes:
            decoder_layers.append(nn.Linear(prev_size, size))
            decoder_layers.append(activation_function())
            prev_size = size
        decoder_layers.append(nn.Linear(prev_size, 865918))
        decoder_layers.append(final_act_function)
        self.decoder = nn.Sequential(*decoder_layers)
        
        
        self.predictor = nn.Sequential(
            nn.Linear(hidden_size, 6)
        )