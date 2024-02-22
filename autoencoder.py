import torch
import torch.nn as nn
import torch.optim as optim
import re
from sklearn.metrics import accuracy_score
import os

#job_id = os.environ.get('SLURM_JOB_ID', 'default_job_id')

class dataset(torch.utils.data.Dataset):
    
    def __init__(self, data, targets):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.long)
      
    def __len__(self):
        return len(self.data)
        return len(self.targets)

    def __getitem__(self, index):
        return self.data[index], self.targets[index]

class Autoencoder(nn.Module):
    
    def __init__(self, input_size, hidden_size, encoder_layer_sizes, activation_function, final_act_function):
        super(Autoencoder, self).__init__()

        # Create encoder layers
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
        decoder_layers.append(nn.Linear(prev_size, input_size))
        decoder_layers.append(final_act_function)
        self.decoder = nn.Sequential(*decoder_layers)
        
        
        self.predictor = nn.Sequential(
            nn.Linear(hidden_size, 6)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        predicted = self.predictor(encoded)
        
        return decoded,predicted
    
    def encode(self, test_loader, device = 'cpu'): 
    #def encode(self, test_loader, device = torch.device('cuda')):

        encoded_data = []
        with torch.no_grad():
            for data, _ in test_loader:
                data = data.to(device)
                encoded = self.encoder(data)
                encoded_data.append(encoded)

        return encoded_data

def train_(model, train_loader,valid_loader, num_epochs, criterion, learning_rate, optimizer, device = 'cpu'): 
#def train_(model, train_loader, num_epochs, criterion, learning_rate, device = torch.device('cuda')):
    
    model.to(device)
    predictor_criterion = nn.CrossEntropyLoss()
    tmp = np.exp(15)
    patience = 6
    tmp2=0.95
    tmp3=0.95
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        model.train()
        for data, targets in train_loader:
            data = data.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs,predictions = model(data)
            #sqrt
            loss_decoder = torch.sqrt(criterion(outputs, data))
            loss_predictor = predictor_criterion(predictions, targets)
            loss = loss_decoder + loss_predictor
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        model.eval()
        valid_loss=0.0
        with torch.no_grad():
            for data, targets in valid_loader:
                data = data.to(device)
                targets = targets.to(device)
                outputs,predictions = model(data)
                #sqrt
                loss_decoder = torch.sqrt(criterion(outputs, data))
                loss_predictor = predictor_criterion(predictions, targets)
                loss = loss_decoder + loss_predictor
                valid_loss += loss.item()

            a,b = model(tens_valid)
            _, predicted = torch.max(b, 1)
            
        check_loss = valid_loss 
        
        if(tmp<=check_loss):
            patience = patience-1
            if(patience==0):
                print("OVERFITTING",flush=True)
                break

        else:
            tmp=check_loss
            patience=6

        print(f"Epoch {epoch+1}/{num_epochs}, Train_Loss: {running_loss/len(train_loader):.4f}, Valid_Loss: {valid_loss/len(valid_loader):.4f}, Valid Accuracy: {accuracy_score(y_valid, predicted)}", flush=True)

def test(model, test_loader, criterion, device = 'cpu'):
#def test(model, test_loader, criterion, device = torch.device('cuda')):

    model.eval()
    model.to(device)
    predictor_criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        running_loss = 0.0
        for data, targets in test_loader:
            data = data.to(device)
            targets = targets.to(device)
            outputs, predictions = model(data)
            
            #sqrt
            loss_decoder = torch.sqrt(criterion(outputs, data))
            loss_predictor = predictor_criterion(predictions, targets)
            loss = loss_decoder + loss_predictor
            running_loss += loss.item()

        return running_loss/len(test_loader)   