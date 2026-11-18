import os
import shutil
import torch
import torchvision.models as models

clean_dir = os.path.join("models", "clean_checkpoints")
os.makedirs(clean_dir, exist_ok=True)

print("--- Step 1: Preparing Clean Pre-Trained Checkpoints ---")

# 1. DenseNet-121 (Copy existing clean file)
src_densenet = r"D:\MRINAL\minor\densenet121_imagenet.pth"
dst_densenet = os.path.join(clean_dir, "densenet121_clean.pth")
if os.path.exists(src_densenet):
    shutil.copyfile(src_densenet, dst_densenet)
    print(f"[OK] DenseNet-121 copied to {dst_densenet} ({os.path.getsize(dst_densenet)/1024/1024:.2f} MB)")
else:
    print("[Downloading] DenseNet-121 from torchvision...")
    m = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
    torch.save(m.state_dict(), dst_densenet)
    print(f"[OK] DenseNet-121 saved ({os.path.getsize(dst_densenet)/1024/1024:.2f} MB)")

# 2. ResNet-18
dst_resnet = os.path.join(clean_dir, "resnet18_clean.pth")
print("[Preparing] ResNet-18...")
resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
torch.save(resnet.state_dict(), dst_resnet)
print(f"[OK] ResNet-18 saved to {dst_resnet} ({os.path.getsize(dst_resnet)/1024/1024:.2f} MB)")

# 3. MobileNet-V2
dst_mobilenet = os.path.join(clean_dir, "mobilenet_v2_clean.pth")
print("[Preparing] MobileNet-V2...")
mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
torch.save(mobilenet.state_dict(), dst_mobilenet)
print(f"[OK] MobileNet-V2 saved to {dst_mobilenet} ({os.path.getsize(dst_mobilenet)/1024/1024:.2f} MB)")

# 4. DeepMark-CNN (Clean Baseline CNN)
dst_cnn = os.path.join(clean_dir, "deepmark_cnn_clean.pth")
src_cnn = r"research_outputs\clean_baseline_cnn.pth"
if os.path.exists(src_cnn):
    shutil.copyfile(src_cnn, dst_cnn)
    print(f"[OK] DeepMark-CNN copied to {dst_cnn} ({os.path.getsize(dst_cnn)/1024:.2f} KB)")
else:
    import torch.nn as nn
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
    cnn = ConvNet()
    torch.save(cnn.state_dict(), dst_cnn)
    print(f"[OK] DeepMark-CNN initialized and saved to {dst_cnn} ({os.path.getsize(dst_cnn)/1024:.2f} KB)")

# 5. SimpleMLP
dst_mlp = os.path.join(clean_dir, "simple_mlp_clean.pth")
import torch.nn as nn
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
mlp = SimpleMLP()
torch.save(mlp.state_dict(), dst_mlp)
print(f"[OK] SimpleMLP saved to {dst_mlp} ({os.path.getsize(dst_mlp)/1024:.2f} KB)")

print("\n--- Summary of Clean Checkpoints in models/clean_checkpoints/ ---")
for f in os.listdir(clean_dir):
    p = os.path.join(clean_dir, f)
    sz = os.path.getsize(p)
    print(f"  * {f}: {sz/1024:.1f} KB ({sz/(1024*1024):.2f} MB)")
