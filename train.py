"""
TorchONNX_HW — Diabetes Classification
========================================
Dataset : Pima Indians Diabetes (768 samples, 8 medical features → 0/1 diagnosis)
Model   : 3-hidden-layer MLP built in PyTorch
Export  : ONNX (diabetes_model.onnx)

Requirements:
    pip install torch onnx onnxruntime scikit-learn pandas numpy requests

Run:
    python train.py
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import onnx
import onnxruntime as ort
import json
import io
import requests
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

# ─── 1. LOAD DATASET ────────────────────────────────────────────────────────
URL = (
    "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv"
)
COLUMNS = [
    "pregnancies", "glucose", "blood_pressure", "skin_thickness",
    "insulin", "bmi", "diabetes_pedigree", "age", "outcome"
]

print("Loading Pima Indians Diabetes dataset …")
response = requests.get(URL)
df = pd.read_csv(io.StringIO(response.text), header=None, names=COLUMNS)
print(f"  Shape: {df.shape}")
print(f"  Class distribution:\n{df['outcome'].value_counts()}\n")

# ─── 2. PRE-PROCESS ─────────────────────────────────────────────────────────
# Replace biologically-impossible 0s with column median
ZERO_INVALID = ["glucose", "blood_pressure", "skin_thickness", "insulin", "bmi"]
for col in ZERO_INVALID:
    median = df[col][df[col] != 0].median()
    df[col] = df[col].replace(0, median)

FEATURE_COLS = [c for c in COLUMNS if c != "outcome"]
X = df[FEATURE_COLS].values.astype(np.float32)
y = df["outcome"].values.astype(np.int64)

scaler = StandardScaler()
X = scaler.fit_transform(X)

scaler_params = {
    "mean":     scaler.mean_.tolist(),
    "std":      scaler.scale_.tolist(),
    "features": FEATURE_COLS,
}
with open("scaler_params.json", "w") as f:
    json.dump(scaler_params, f, indent=2)
print("Saved scaler_params.json")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train_t = torch.tensor(X_train)
y_train_t = torch.tensor(y_train)
X_test_t  = torch.tensor(X_test)

# ─── 3. MODEL ───────────────────────────────────────────────────────────────
class DiabetesNet(nn.Module):
    """3-hidden-layer MLP for binary diabetes classification."""
    def __init__(self, in_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),

            nn.Linear(16, 2),
        )

    def forward(self, x):
        return self.net(x)


model = DiabetesNet(in_features=len(FEATURE_COLS))
print(model)

# ─── 4. TRAINING ────────────────────────────────────────────────────────────
EPOCHS     = 150
BATCH_SIZE = 32
LR         = 1e-3

neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
weight    = torch.tensor([1.0, neg / pos])
criterion = nn.CrossEntropyLoss(weight=weight)
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

dataset    = torch.utils.data.TensorDataset(X_train_t, y_train_t)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

print("\nTraining …")
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0
    for xb, yb in dataloader:
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(xb)
    scheduler.step()
    if epoch % 30 == 0:
        print(f"  Epoch {epoch:3d}/{EPOCHS}  loss={total_loss/len(X_train):.4f}")

# ─── 5. EVALUATION ──────────────────────────────────────────────────────────
model.eval()
with torch.no_grad():
    preds = model(X_test_t).argmax(dim=1).numpy()

acc = accuracy_score(y_test, preds)
print(f"\nTest Accuracy: {acc * 100:.2f}%")
print(classification_report(y_test, preds, target_names=["No Diabetes", "Diabetes"]))

# ─── 6. EXPORT TO ONNX ──────────────────────────────────────────────────────
model.eval()
dummy     = torch.zeros(1, len(FEATURE_COLS))
ONNX_PATH = "diabetes_model.onnx"

torch.onnx.export(
    model, dummy, ONNX_PATH,
    export_params=True,
    opset_version=17,
    do_constant_folding=True,
    input_names=["features"],
    output_names=["logits"],
    dynamic_axes={"features": {0: "batch_size"}, "logits": {0: "batch_size"}},
)

onnx_model = onnx.load(ONNX_PATH)
onnx.checker.check_model(onnx_model)
print(f"\nONNX model verified → {ONNX_PATH}")

sess = ort.InferenceSession(ONNX_PATH)
out  = sess.run(None, {"features": X_test[:5]})[0]
print("ONNX Runtime sample logits:", out)
print("\nDone! Deploy index.html + diabetes_model.onnx + scaler_params.json to GitHub Pages.")
