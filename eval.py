import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_fscore_support

def eval_model(model: torch.nn, dataloader: DataLoader, criterion: torch.nn, device, note=None):
    
    test_loss = 0
    test_acc = 0
    test_precision = [0 for _ in range(2)]
    test_recall = [0 for _ in range(2)]
    test_fscore = [0 for _ in range(2)]
    test_support = [0 for _ in range(2)]

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

            precision, recall, fscore, support = precision_recall_fscore_support(y.cpu(), preds.cpu(), zero_division=np.nan)
            
            test_precision[0] += precision[0]
            test_precision[1] += precision[1]
            test_recall[0] += recall[0]
            test_recall[1] += recall[1]
            test_fscore[0] += fscore[0]
            test_fscore[1] += fscore[1]
            test_support[0] += support[0]
            test_support[1] += support[1]

    # calculate average metrics
    test_loss /= len(dataloader)
    test_acc /= len(dataloader)
    test_precision[0] /= len(dataloader)
    test_precision[1] /= len(dataloader)
    test_recall[0] /= len(dataloader)
    test_recall[1] /= len(dataloader)
    test_fscore[0] /= len(dataloader)
    test_fscore[1] /= len(dataloader)

    labels = ['model_name', 'loss', 'acc', 'prec0', 'prec1', 'rec0', 'rec1', 'f1sc0', 'f1sc1', 'sup0', 'sup1']
    results = pd.Series([model_name, test_loss, test_acc, test_precision[0], test_precision[1], test_recall[0], test_recall[1],
                        test_fscore[0], test_fscore[1], test_support[0], test_support[1]])
    results.index = labels
    return results

def eval_forest_model(model: RandomForestClassifier, X_test: pd.DataFrame, y_test: list, criterion: torch.nn, note=None):

    model_name = type(model).__name__
    if note:
        model_name += ": "+note
    
    y_pred = model.predict(X_test)

    test_loss = criterion(torch.tensor(y_pred).float(), torch.tensor(y_test).float()).item()
    test_acc = (y_pred == y_test).sum() / len(y_test)
    test_precision, test_recall, test_fscore, test_support = precision_recall_fscore_support(
                                                    y_true = torch.tensor(y_test),
                                                    y_pred = torch.tensor(y_pred),
                                                    zero_division=np.nan)
    
    labels = ['model_name', 'loss', 'acc', 'prec0', 'prec1', 'rec0', 'rec1', 'f1sc0', 'f1sc1', 'sup0', 'sup1']
    results = pd.Series([model_name, test_loss, test_acc, test_precision[0], test_precision[1], test_recall[0], test_recall[1],
                        test_fscore[0], test_fscore[1], test_support[0], test_support[1]])
    results.index = labels
    return results
