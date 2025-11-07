import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import matplotlib.pyplot as plt

def read_csv_rows_pandas(file_path, start_line, end_line, header_row_present=True):
    df = pd.read_csv(file_path, header=0 if header_row_present else None)
    return df.iloc[start_line - 1: end_line]

def one_hot_encode(sequences):
    mapping = {1: [1,0,0,0],   # A
               2: [0,1,0,0],   # T
               3: [0,0,1,0],   # G
               4: [0,0,0,1]}   # C
    one_hot = np.array([[mapping[base] for base in seq] for seq in sequences])
    return one_hot 

df = read_csv_rows_pandas("cccna_data.csv", 1000*(3-1)+1, 3000)  # строки 2001–3000
v_columns = [c for c in df.columns if c.startswith("V")]
X = df[v_columns].values  # (1000, ~400)
y = df['bind'].astype(int).values 

X_onehot = one_hot_encode(X) 

X_train, X_test, y_train, y_test = train_test_split(
    X_onehot, y, test_size=0.2, random_state=42, stratify=y
)

X_train_t = torch.tensor(X_train, dtype=torch.float32)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32)

train_dataset = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

class DNA_CNN(nn.Module):
    def __init__(self, seq_len, n_channels=4):
        super(DNA_CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=n_channels, out_channels=64, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=7, padding=3)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = torch.max(x, dim=2)[0]  # (batch, 64)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x.squeeze(1) 
    


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DNA_CNN(seq_len=X_onehot.shape[1]).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 50
train_losses = []
train_accuracies = []

for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0
    correct = 0
    total = 0

    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        predicted = (outputs > 0.5).float()
        total += batch_y.size(0)
        correct += (predicted == batch_y).sum().item()

    avg_loss = epoch_loss / len(train_loader)
    avg_acc = correct / total
    train_losses.append(avg_loss)
    train_accuracies.append(avg_acc)

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Acc: {avg_acc:.4f}")

# === Оценка на тесте ===
model.eval()
with torch.no_grad():
    X_test_t = X_test_t.to(device)
    y_pred_proba = model(X_test_t).cpu().numpy()
    y_pred = (y_pred_proba > 0.5).astype(int)

print("\nROC-AUC:", roc_auc_score(y_test, y_pred_proba))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss', color='red')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Training Accuracy', color='blue')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training Accuracy')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()