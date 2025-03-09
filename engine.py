import torch
from torch.utils.data import DataLoader

def train_step(model: torch.nn, dataloader: DataLoader, optimizer, criterion, device):
    train_loss = 0
    model.train()
    for _, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()

        # forward pass
        out = model(X)
        preds = out.squeeze(dim=1).sigmoid().round()

        # calculate loss
        loss = criterion(out.squeeze(dim=1), y)

        # aggregate loss, accuracy
        train_loss += loss.item()

        # backpropagation and updating weights
        loss.backward()
        optimizer.step()

    train_loss /= len(dataloader)

    return train_loss

def test_step(model: torch.nn, dataloader: DataLoader, criterion, device):
    test_loss = 0
    model.eval()
    with torch.inference_mode():
        for _, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            # forward pass
            out = model(X)
            preds = out.squeeze(dim=1).sigmoid().round()

            # calculate loss
            loss = criterion(out.squeeze(dim=1), y)

            # aggregate loss, accuracy
            test_loss += loss.item()

    test_loss /= len(dataloader)

    return test_loss

def train_test(model: torch.nn, train_dataloader: DataLoader, test_dataloader: DataLoader, epochs: int, 
          optimizer, criterion, device: str, verbose: bool):
    for epoch in range(1, epochs + 1):
        train_loss = train_step(model, train_dataloader, optimizer, criterion, device)
        test_loss = test_step(model, test_dataloader, criterion, device)
        if verbose:
            print(f"{epoch} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")