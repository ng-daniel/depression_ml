import torch
from torch.utils.data import DataLoader

def _train_step(model: torch.nn, dataloader: DataLoader, optimizer, criterion, device):
    '''
    Training loop for pytorch models.

    Args
    - model - the model class
    - dataloader - training data wrapped in a batched dataloader
    - optimizer - optimizer to update weights
    - criterion - loss function for backpropagation
    - device - 'cpu' or 'cuda'

    Returns
    - the average loss for this training loop
    '''
    
    train_loss = 0
    model.train()
    for _, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()

        # forward pass
        out = model(X)
        preds = out.squeeze().sigmoid().round()

        # calculate loss
        loss = criterion(out.squeeze(), y)

        # aggregate loss, accuracy
        train_loss += loss.item()

        # backpropagation and updating weights
        loss.backward()
        optimizer.step()

    train_loss /= len(dataloader)

    return train_loss

def _test_step(model: torch.nn, dataloader: DataLoader, criterion, device):
    '''
    Testing loop for pytorch models.

    Args
    - model - the model class
    - dataloader - training data wrapped in a batched dataloader
    - criterion - loss function for backpropagation
    - device - 'cpu' or 'cuda'

    Returns
    - the average loss for this test loop
    '''
    
    test_loss = 0
    model.eval()
    with torch.inference_mode():
        for _, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            # forward pass
            out = model(X)
            preds = out.squeeze().sigmoid().round()

            # calculate loss
            loss = criterion(out.squeeze(), y)

            # aggregate loss, accuracy
            test_loss += loss.item()

    test_loss /= len(dataloader)

    return test_loss

def train_test(model: torch.nn, train_dataloader: DataLoader, test_dataloader: DataLoader, epochs: int, 
          optimizer, criterion, device: str, verbose: bool):
    '''
    Trains the model for a number of epochs.

    Args
    - model - pytorch nn class
    - train_dataloader
    - test_dataloader
    - epochs - number of times to run train_step + test_step
    - optimizer - optimizer to update weights
    - criterion - loss function for backpropagation
    - device - 'cpu' or 'cuda'
    - verbose - whether to print output

    Returns
    - nothing
    '''
    for epoch in range(1, epochs + 1):
        train_loss = _train_step(model, train_dataloader, optimizer, criterion, device)
        test_loss = _test_step(model, test_dataloader, criterion, device)
        if verbose:
            print(f"{epoch} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")