import torch.nn as nn 
import torch 
import torch.optim as optim 
from tqdm import tqdm 

def train_multi_step_model(model: nn.Module, 
                dataset: torch.Tensor, 
                sequence_length: int, 
                num_steps: int, 
                batch_size: int, 
                epochs: int, 
                learning_rate: float) -> None:
    """
    Trains a Multi-layer Perceptron (MLP) model for multi-step forecasting.

    Parameters:
    model (nn.Module): The MLP model to be trained.
    dataset (torch.Tensor): The time series dataset used for training.
    sequence_length (int): The number of previous time steps to use as input for the model.
    num_steps (int): The number of steps to predict at once.
    batch_size (int): The number of samples in each batch.
    epochs (int): The number of training iterations.
    learning_rate (float): The learning rate for the optimizer.

    Returns:
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    for epoch in tqdm(range(epochs)):
        for i in range(0, len(dataset) - sequence_length - num_steps, batch_size):
            x_batch = dataset[i:i+sequence_length]
            y_batch = dataset[i+sequence_length:i+sequence_length+num_steps]

            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')


def train(model, train_loader, epochs, device, lr, skip=50):

  criterion = nn.MSELoss()
  optimizer = torch.optim.SGD(model.parameters(), lr=lr)

  history = {'loss_train': []}

  for epoch in tqdm(range(1, epochs+1)):
    
    y_hat = []

    train_epoch_loss = 0
    model.train()
    for X_train_batch, y_train_batch in train_loader:
        X, y = X_train_batch.to(device), y_train_batch.to(device)
        optimizer.zero_grad()
        
        y_pred = model(X)
        
        loss = criterion(y_pred, y)
        
        loss.backward()
        optimizer.step()
        
        train_epoch_loss += loss.item()
        
        y_hat.append(y_pred)
    
    
    history['loss_train'].append(train_epoch_loss)
    
    if epoch % skip == 0:
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')
  return history, y_hat
