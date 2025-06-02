import os

import pandas as pd
import torch
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from engine import train_test
from eval import eval_model, eval_sklearn_model, append_weighted_average
from model import ZeroR, ConvNN, LSTM, FeatureMLP, LSTM_Feature

def run_linear_svc(data: list, criterion, device, weights = None):
    print("Linear SVM Classifier:")
    
    linear_svc_results = []
    for i, (X_train, X_test, y_train, y_test) in enumerate(tqdm(data, ncols=50)):
        # reset model
        model = SVC(kernel='linear', probability=True, class_weight=weights)
        # train model
        model.fit(X_train, y_train)
        linear_svc_results.append(
                eval_sklearn_model(model = model,
                                note = f"{i}",
                                X_test=X_test,
                                y_test=y_test,
                                criterion = criterion,
                                device = device)
        )
    linear_svc_results = pd.concat(linear_svc_results, axis=1).transpose()
    linear_svc_results = append_weighted_average(linear_svc_results)
    return linear_svc_results

def run_decision_tree(data: list, criterion, device, weights = None):
      print("Decision Tree:")

      decision_tree_results = []
      for i, (X_train, X_test, y_train, y_test) in enumerate(tqdm(data, ncols=50)):
            # reset model
            model = DecisionTreeClassifier(class_weight=weights)
            # train model
            model.fit(X_train, y_train)
            decision_tree_results.append(
                  eval_sklearn_model(model = model,
                                    note = f"{i}",
                                    X_test=X_test,
                                    y_test=y_test,
                                    criterion = criterion,
                                    device = device)
            )
      decision_tree_results = pd.concat(decision_tree_results, axis=1).transpose()
      decision_tree_results = append_weighted_average(decision_tree_results)
      return decision_tree_results

def run_random_forest(data: list, criterion, device, weights = None, n_estimators : int = 100):
      print("Random Forest:")
      
      forest_results = []
      N_ESTIMATORS = n_estimators
      for i, (X_train, X_test, y_train, y_test) in enumerate(tqdm(data, ncols=50)):
            # reset model
            model_2 = RandomForestClassifier(n_estimators=N_ESTIMATORS)
            # train model
            model_2.fit(X_train, y_train)
            forest_results.append(
                  eval_sklearn_model(model = model_2,
                                    note = f"{i}",
                                    X_test=X_test,
                                    y_test=y_test,
                                    criterion = criterion,
                                    device = device)
            )
      forest_results = pd.concat(forest_results, axis=1).transpose()
      forest_results = append_weighted_average(forest_results)
      return forest_results

def run_XGBoost(data: list, criterion, device, learning_rate, gamma = None, weights = None):
      print("Decision Tree:")

      XGBoost_results = []
      for i, (X_train, X_test, y_train, y_test) in enumerate(tqdm(data, ncols=50)):
            # reset model
            model = XGBClassifier(learning_rate=learning_rate,
                                  gamma=gamma,
                                  verbosity=2)
            # train model
            model.fit(X_train, y_train)
            XGBoost_results.append(
                  eval_sklearn_model(model = model,
                                    note = f"{i}",
                                    X_test=X_test,
                                    y_test=y_test,
                                    criterion = criterion,
                                    device = device)
            )
      XGBoost_results = pd.concat(XGBoost_results, axis=1).transpose()
      XGBoost_results = append_weighted_average(XGBoost_results)
      return XGBoost_results

def run_zeroR_baseline(data: list, criterion, device):
      print("ZeroR Baseline:")
      
      zeroR_results = []
      for i, (_, test_dataloader) in enumerate(tqdm(data, ncols=50)):
            model_0R = ZeroR()
            zeroR_results.append(
                  eval_model(model = model_0R,
                        note = f"{i}",
                        dataloader = test_dataloader,
                        criterion = criterion,
                        device = device)
            )
      zeroR_results = pd.concat(zeroR_results, axis=1).transpose()
      zeroR_results = append_weighted_average(zeroR_results)
      return zeroR_results

def run_lstm(data: list, criterion, device, learning_rate, epochs, in_shape, out_shape, hidden_shape, lstm_layers):
      print("LSTM:")
      
      lstm_results = []
      IN_2 = in_shape
      OUT_2 = out_shape
      HIDDEN_2 = hidden_shape
      LSTM_LAYERS = lstm_layers
      for i, (train_dataloader, test_dataloader) in enumerate(tqdm(data, ncols=50)):
            # reset model
            model_2 = LSTM(IN_2, OUT_2, HIDDEN_2, LSTM_LAYERS).to(device)
            
            optimizer = torch.optim.Adam(params = model_2.parameters(), lr=learning_rate)
            # train model
            train_test(model_2, train_dataloader, test_dataloader, epochs = epochs, optimizer=optimizer, 
                  criterion=criterion, device=device, verbose=True)
            lstm_results.append(
                  eval_model(model = model_2,
                        note = f"{i}",
                        dataloader = test_dataloader,
                        criterion = criterion,
                        device = device)
            )
      lstm_results = pd.concat(lstm_results, axis=1).transpose()
      lstm_results = append_weighted_average(lstm_results)
      return lstm_results

def run_cnn(data: list, criterion, device, learning_rate, epochs, in_shape, out_shape, hidden_shape, flatten_factor):
      print("CNN:")
      
      cnn_results = []
      IN_0 = in_shape
      OUT_0 = out_shape
      HIDDEN_0 = hidden_shape
      FLATTEN_0 = flatten_factor
      for i, (train_dataloader, test_dataloader) in enumerate(tqdm(data, ncols=50)):
            # reset model
            model_0 = ConvNN(IN_0, OUT_0, HIDDEN_0, FLATTEN_0).to(device)
            optimizer = torch.optim.Adam(params = model_0.parameters(), lr=learning_rate)
            # train model
            train_test(model_0, train_dataloader, test_dataloader, epochs = epochs, optimizer=optimizer, 
                  criterion=criterion, device=device, verbose=False)
            cnn_results.append(
                  eval_model(model = model_0,
                        note = f"{i}",
                        dataloader = test_dataloader,
                        criterion = criterion,
                        device = device)
            )
      cnn_results = pd.concat(cnn_results, axis=1).transpose()
      cnn_results = append_weighted_average(cnn_results)
      return cnn_results

def run_mlp(data: list, criterion, device, learning_rate, epochs, in_shape, out_shape, hidden_shape):
      print("Multi-layer Perceptron:")
      
      mlp_results = []
      IN_1 = in_shape
      OUT_1 = out_shape
      HIDDEN_1 = hidden_shape
      for i, (train_dataloader, test_dataloader) in enumerate(tqdm(data, ncols=50)):
            # reset model
            model_1 = FeatureMLP(IN_1, OUT_1, HIDDEN_1).to(device)
            optimizer = torch.optim.Adam(params = model_1.parameters(), lr=learning_rate)
            # train model
            train_test(model_1, train_dataloader, test_dataloader, epochs = epochs, optimizer=optimizer, 
                  criterion=criterion, device=device, verbose=False)
            mlp_results.append(
                  eval_model(model = model_1,
                        note = f"{i}",
                        dataloader = test_dataloader,
                        criterion = criterion,
                        device = device)
            )
      mlp_results = pd.concat(mlp_results, axis=1).transpose()
      mlp_results = append_weighted_average(mlp_results)
      return mlp_results

def run_lstm_feature(data: list, criterion, device, learning_rate, epochs, in_shape, out_shape, hidden_shape, lstm_layers):
      print("LSTM V2:")
      
      lstm_series_results = []
      IN_3 = in_shape
      OUT_3 = out_shape
      HIDDEN_3 = hidden_shape
      LSTM_LAYERS = lstm_layers
      for i, (train_dataloader, test_dataloader) in enumerate(tqdm(data, ncols=50)):
            # reset model
            model_3 = LSTM_Feature(IN_3, OUT_3, HIDDEN_3, LSTM_LAYERS).to(device)
            optimizer = torch.optim.Adam(params = model_3.parameters(), lr=learning_rate)
            # train model
            train_test(model_3, train_dataloader, test_dataloader, epochs = epochs, optimizer=optimizer, 
                  criterion=criterion, device=device, verbose=True)
            lstm_series_results.append(
                  eval_model(model = model_3,
                        note = f"{i}",
                        dataloader = test_dataloader,
                        criterion = criterion,
                        device = device)
            )
      lstm_series_results = pd.concat(lstm_series_results, axis=1).transpose()
      lstm_series_results = append_weighted_average(lstm_series_results)
      return lstm_series_results