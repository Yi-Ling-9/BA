import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from adbench.baseline.Customized.model import CustomUnsupervisedModel
from deepod.models.tabular import ICL
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_selection import mutual_info_classif

class Customized:
    def __init__(self, seed=None, model_name=None, kernel_size=3, hidden_dims='100,50', rep_dim=64, act='ReLU', bias=False):
        self.seed = seed
        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
        self.model = None  # The model will be initialised during the fit stage.
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.kernel_size = kernel_size
        self.hidden_dims = hidden_dims
        self.rep_dim = rep_dim
        self.act = act
        self.bias = bias
        self.n_features = None

    def fit(self, X_train, y_train=None):
        """
        :param X_train: Training data set.
        :param y_train: Optional, ADBench compatibility parameter, not used.
        """
        input_dim = X_train.shape[1]  # Infer the input dimension based on X_train
        self.n_features = input_dim  # Initialise n_features

        # Initialise custom model
        self.model = CustomUnsupervisedModel(input_dim=input_dim).to(self.device)

        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        train_dataset = TensorDataset(X_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        self.model.train()
        for epoch in range(50):  # Assume training 50 epochs
            for data in train_loader:
                inputs = data[0]
                optimizer.zero_grad()
                outputs, _ = self.model(inputs)
                loss = torch.nn.MSELoss()(outputs, inputs)  # Using reconstruction error as the loss function
                loss.backward()
                optimizer.step()
        return self

    def predict_score(self, X_test):
        """
        Generate anomaly scores that comply with the predict_score interface required by ADBench.
        :param X_test: Test data set.
        :return: Anomaly score (the higher the score, the more anomalous).
        """
        if self.model is None:
            raise ValueError("Model is not fitted yet.")

        self.model.eval()
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(self.device)
        test_dataset = TensorDataset(X_test_tensor)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        anomaly_scores = []
        with torch.no_grad():
            for data in test_loader:
                inputs = data[0]
                outputs, _ = self.model(inputs)
                reconstruction_error = torch.nn.functional.mse_loss(outputs, inputs, reduction='none').mean(dim=1)
                anomaly_scores.append(reconstruction_error.cpu().numpy())

        return np.concatenate(anomaly_scores)

    def odin_score(self, X_test, epsilon=0.001, temperature=1000):
        if self.model is None:
            raise ValueError("Model is not fitted yet.")

        self.model.eval()
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(self.device)
        X_test_tensor.requires_grad = True

        # forward propagation
        outputs, _ = self.model(X_test_tensor)
        loss = torch.nn.MSELoss()(outputs, X_test_tensor)
        loss.backward()

        # Add perturbation
        perturbation = epsilon * X_test_tensor.grad.sign()
        X_test_perturbed = X_test_tensor + perturbation
        X_test_perturbed = torch.clamp(X_test_perturbed, 0, 1)  # Ensure that the input remains within the valid range after perturbation.

        # temperature scaling
        with torch.no_grad():
            outputs_perturbed, _ = self.model(X_test_perturbed) 
            outputs_perturbed /= temperature
            odin_score = torch.nn.functional.mse_loss(outputs_perturbed, X_test_tensor, reduction='none').mean(dim=1)

        return odin_score.cpu().numpy()

    def gradnorm_score(self, X_test):
        if self.model is None:
            raise ValueError("Model is not fitted yet.")

        self.model.eval()
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(self.device)
        X_test_tensor.requires_grad = True

        # forward propagation
        outputs, _ = self.model(X_test_tensor)
        loss = torch.nn.MSELoss()(outputs, X_test_tensor)
        loss.backward()

        # Calculate gradient norm
        gradnorm = torch.norm(X_test_tensor.grad, p=2, dim=1)

        return gradnorm.cpu().numpy()

    def icl_score(self, X_train, X_test):
        # print("Running ICL...")
        clf_icl = ICL(device='cuda', verbose=0)

        #X_train = torch.tensor(X_train).float().to('cuda')  
        #X_test = torch.tensor(X_test).float().to('cuda')  

        clf_icl.fit(X_train, y=None)
        icl = clf_icl.decision_function(X_test)

        return icl

