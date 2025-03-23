import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support

def eval_model(model: torch.nn, dataloader: DataLoader, criterion: torch.nn, device, note=None):
    
    num_items = 0
    test_loss = 0
    test_acc = 0
    test_precision = 0
    test_recall = 0
    test_fscore = 0

    model_name = type(model).__name__
    if note:
        model_name += ": "+note

    model.eval()
    with torch.inference_mode():
        for _, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            # forward pass
            out = model(X)
            preds = out.squeeze().sigmoid().round()

            # calculate loss
            loss = criterion(out.squeeze(), y)

            # aggregate metrics
            test_loss += loss.item()
            test_acc += ((preds==y).sum() / len(y)).item()
            precision, recall, fscore, _ = precision_recall_fscore_support(y_true = y.cpu(),
                                                                        y_pred = preds.cpu(),
                                                                        average = 'binary',
                                                                        zero_division=np.nan)
            test_precision += precision
            test_recall += recall
            test_fscore += fscore
            num_items += len(y)

    # calculate average metrics
    test_loss /= len(dataloader)
    test_acc /= len(dataloader)
    test_precision /= len(dataloader)
    test_recall /= len(dataloader)
    test_fscore /= len(dataloader)

    return {'model_name': model_name,
            'num_samples': f"{num_items}",
            'loss': f"{test_loss:.4f}",
            'accuracy': f"{(100*test_acc):.2f}%",
            'precision': f"{test_loss:.4f}",
            'recall': f"{test_recall:.4f}",
            'f1score': f"{test_fscore:.4f}"}