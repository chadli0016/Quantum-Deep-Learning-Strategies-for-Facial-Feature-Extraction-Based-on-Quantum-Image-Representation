# =========================================================
# IMPORTS
# =========================================================

import gc
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn import svm
from sklearn.linear_model import LogisticRegression

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not installed. XGBoost column will be skipped.")


# =========================================================
# SIMPLE NEURAL NETWORK CLASSIFIER
# =========================================================

class SimpleNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)


def train_nn_classifier(
    X_train,
    X_test,
    Y_train,
    Y_test,
    epochs=10,
    batch_size=32
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    Y_train_t = torch.tensor(Y_train, dtype=torch.long)

    train_loader = DataLoader(
        TensorDataset(X_train_t, Y_train_t),
        batch_size=batch_size,
        shuffle=True
    )

    model = SimpleNN(
        input_dim=X_train.shape[1],
        num_classes=len(np.unique(Y_train))
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()

        for X_batch, Y_batch in train_loader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, Y_batch)
            loss.backward()
            optimizer.step()

    model.eval()

    with torch.no_grad():
        X_test_t = X_test_t.to(device)
        outputs = model(X_test_t)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()

    del model, X_train_t, X_test_t, Y_train_t
    gc.collect()

    return accuracy_score(Y_test, preds) * 100


# =========================================================
# FEATURE-BASED CNN-STYLE CLASSIFIER
# Uses PCA feature vectors, not images
# =========================================================

class FeatureCNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.35),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.30),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.20),

            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)


def train_cnn_classifier(
    X_train,
    X_test,
    Y_train,
    Y_test,
    epochs=30,
    batch_size=32
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    Y_train_t = torch.tensor(Y_train, dtype=torch.long)

    train_loader = DataLoader(
        TensorDataset(X_train_t, Y_train_t),
        batch_size=batch_size,
        shuffle=True
    )

    model = FeatureCNN(
        input_dim=X_train.shape[1],
        num_classes=len(np.unique(Y_train))
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()

        for X_batch, Y_batch in train_loader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)

            loss = criterion(outputs, Y_batch)
            loss.backward()
            optimizer.step()

    model.eval()

    with torch.no_grad():
        X_test_t = X_test_t.to(device)
        outputs = model(X_test_t)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()

    del model, X_train_t, X_test_t, Y_train_t
    gc.collect()

    return accuracy_score(Y_test, preds) * 100


# =========================================================
# MAIN EXPERIMENT FUNCTION
# =========================================================

def run_experiment(feature_df):
    metadata_cols = ["actor_name", "image_name", "image_path"]

    X = feature_df.drop(columns=metadata_cols).values.astype(np.float32)

    label_encoder = LabelEncoder()
    Y = label_encoder.fit_transform(feature_df["actor_name"])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X).astype(np.float32)

    pca_components = min(
        1024,
        X_scaled.shape[0] - 1,
        X_scaled.shape[1]
    )

    pca = PCA(
        n_components=pca_components,
        svd_solver="randomized",
        random_state=42
    )

    X_reduced = pca.fit_transform(X_scaled).astype(np.float32)

    del X, X_scaled
    gc.collect()

    X_train, X_test, Y_train, Y_test = train_test_split(
        X_reduced,
        Y,
        test_size=0.15,
        stratify=Y,
    )

    results = {}

    # ---------------- SVM ---------------- #
    svm_model = svm.SVC(
        kernel="linear",
        random_state=42
    )

    svm_model.fit(X_train, Y_train)
    svm_pred = svm_model.predict(X_test)

    results["SVM_accuracy"] = accuracy_score(Y_test, svm_pred) * 100

    del svm_model, svm_pred
    gc.collect()

    # ---------------- Logistic Regression ---------------- #
    log_model = LogisticRegression(
        max_iter=1000,
        random_state=42
    )

    log_model.fit(X_train, Y_train)
    log_pred = log_model.predict(X_test)

    results["Logistic_accuracy"] = accuracy_score(Y_test, log_pred) * 100

    del log_model, log_pred
    gc.collect()

    # ---------------- Neural Network ---------------- #
    results["NN_accuracy"] = train_nn_classifier(
        X_train,
        X_test,
        Y_train,
        Y_test,
        epochs=10
    )

    # ---------------- CNN-Style Feature Classifier ---------------- #
    results["CNN_accuracy"] = train_cnn_classifier(
        X_train,
        X_test,
        Y_train,
        Y_test,
        epochs=30
    )

    # ---------------- XGBoost ---------------- #
    if XGBOOST_AVAILABLE:
        xgb_model = XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            objective="multi:softmax",
            eval_metric="mlogloss",
            random_state=42,
            tree_method="hist"
        )

        xgb_model.fit(X_train, Y_train)
        xgb_pred = xgb_model.predict(X_test)

        results["XGBoost_accuracy"] = accuracy_score(Y_test, xgb_pred) * 100

        del xgb_model, xgb_pred
        gc.collect()

    pca_dim = X_reduced.shape[1]

    del X_reduced, X_train, X_test, Y_train, Y_test
    gc.collect()

    return results, pca_dim