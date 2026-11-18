import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

print("=================================================================================")
print("      TRAINING AUTHENTIC CLEAN BASELINE CLASSIFIERS (MNIST)                      ")
print("=================================================================================")

clean_dir = r"D:\MRINAL\minor\models\clean_checkpoints"
os.makedirs(clean_dir, exist_ok=True)

# 1. Load Real MNIST Train & Test Data
train_img_p = r"D:\MRINAL\minor\data_cache\mnist_kaggle\train-images.idx3-ubyte"
train_lbl_p = r"D:\MRINAL\minor\data_cache\mnist_kaggle\train-labels.idx1-ubyte"
test_img_p = r"D:\MRINAL\minor\data_cache\mnist_kaggle\t10k-images.idx3-ubyte"
test_lbl_p = r"D:\MRINAL\minor\data_cache\mnist_kaggle\t10k-labels.idx1-ubyte"

with open(train_img_p, 'rb') as f_img, open(train_lbl_p, 'rb') as f_lbl:
    f_img.read(16); f_lbl.read(8)
    # Train on 20,000 real images (takes ~15 sec on CPU)
    n_train = 20000
    train_x = np.frombuffer(f_img.read(n_train * 784), dtype=np.uint8).reshape(n_train, 1, 28, 28)
    train_y = np.frombuffer(f_lbl.read(n_train), dtype=np.uint8)

with open(test_img_p, 'rb') as f_img, open(test_lbl_p, 'rb') as f_lbl:
    f_img.read(16); f_lbl.read(8)
    n_test = 1000
    test_x = np.frombuffer(f_img.read(n_test * 784), dtype=np.uint8).reshape(n_test, 1, 28, 28)
    test_y = np.frombuffer(f_lbl.read(n_test), dtype=np.uint8)

train_x = (train_x.astype(np.float32) / 255.0 - 0.1307) / 0.3081
test_x = (test_x.astype(np.float32) / 255.0 - 0.1307) / 0.3081

train_x_t = torch.tensor(train_x)
train_y_t = torch.tensor(train_y, dtype=torch.long)
test_x_t = torch.tensor(test_x)
test_y_t = torch.tensor(test_y, dtype=torch.long)

# 2. Architectures
class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )
    def forward(self, x):
        return self.classifier(self.features(x).view(x.size(0), -1))

class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    def forward(self, x):
        return self.net(x)

def train_model(model, name, epochs=3, batch_size=128):
    print(f"\n[Training] {name} on {n_train} real MNIST digits ({epochs} epochs)...")
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for ep in range(epochs):
        perm = torch.randperm(n_train)
        running_loss = 0.0
        for b in range(0, n_train, batch_size):
            idx = perm[b:b+batch_size]
            bx, by = train_x_t[idx], train_y_t[idx]
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"  Epoch {ep+1}/{epochs} Complete - Loss: {running_loss / (n_train/batch_size):.4f}")
        
    model.eval()
    with torch.no_grad():
        test_out = model(test_x_t)
        probs = torch.softmax(test_out, dim=-1)
        preds = probs.argmax(dim=-1).numpy()
        confs = probs.max(dim=-1).values.numpy() * 100.0
        acc = (np.sum(preds == test_y) / n_test) * 100.0
    print(f"[✓] {name} Genuine Holdout Accuracy on 1,000 Real Test Digits: {acc:.2f}%")
    return model, acc, preds, confs

# Train DeepMark-CNN
cnn = ConvNet()
cnn, cnn_acc, cnn_preds, cnn_confs = train_model(cnn, "DeepMark-CNN", epochs=3)
torch.save(cnn.state_dict(), os.path.join(clean_dir, "deepmark_cnn_clean.pth"))

# Train SimpleMLP
mlp = SimpleMLP()
mlp, mlp_acc, mlp_preds, mlp_confs = train_model(mlp, "SimpleMLP", epochs=3)
torch.save(mlp.state_dict(), os.path.join(clean_dir, "simple_mlp_clean.pth"))

print("\n--- SAMPLE-BY-SAMPLE VERIFICATION (FIRST 10 REAL TEST IMAGES) ---")
for i in range(10):
    true_d = test_y[i]
    pred_d = cnn_preds[i]
    conf = cnn_confs[i]
    status = "✔ MATCH" if pred_d == true_d else "✘ MISMATCH"
    print(f"Sample #{i+1:02d}: True Digit = {true_d} | Predicted = {pred_d} | Confidence = {conf:.2f}% | Status = {status}")

print("\n[✓] Both clean baseline checkpoints are now authentically trained and saved!")
