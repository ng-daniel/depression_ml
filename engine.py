import torch
from torch.utils.data import DataLoader

def train_step(model: torch.nn,
               dataloader: DataLoader,
               optimizer,
               criterion,
               device):
    train_loss = 0
    train_acc = 0
    model.train()
    for _, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()

        # forward pass
        out = model(X)
        preds = out.squeeze(dim=1).sigmoid().round()

        # calculate loss
        loss = criterion(out.squeeze(dim=1), y)
        train_loss += loss.item()

        # aggregate accuracy
        train_acc += (preds==y).sum() / len(y)

        # backpropagation and updating weights
        loss.backward()
        optimizer.step()
    train_loss /= len(dataloader)
    train_acc /= len(dataloader)

    return train_loss, train_acc

def test_step(model: torch.nn,
              dataloader: DataLoader,
              criterion,
              device):
    test_loss = 0
    test_acc = 0
    model.eval()
    with torch.inference_mode():
        for _, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            # forward pass
            out = model(X)
            preds = out.squeeze(dim=1).sigmoid().round()

            # calculate loss
            loss = criterion(out.squeeze(dim=1), y)
            test_loss += loss.item()

            # aggregate accuracy
            test_acc += ((preds==y).sum() / len(y)).item()


    test_loss /= len(dataloader)
    test_acc /= len(dataloader)

    return test_loss, test_acc