import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights, vit_b_16, ViT_B_16_Weights
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
import random

# --- Dual Fusion Model ---
class DualStreamFusion(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.rgb_net = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        self.freq_net = resnet18(weights=None)

        rgb_features = self.rgb_net.heads[0].in_features
        freq_features = self.freq_net.fc.in_features

        self.rgb_net.heads = nn.Identity()
        self.freq_net.fc = nn.Identity()

        self.classifier = nn.Sequential(
            nn.Linear(rgb_features + freq_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, rgb, freq):
        f_rgb = self.rgb_net(rgb)
        f_freq = self.freq_net(freq)
        return self.classifier(torch.cat([f_rgb, f_freq], dim=1))

def fft_transform(img_tensor):
    fft = torch.fft.fft2(img_tensor)
    fft = torch.fft.fftshift(fft)
    mag = torch.log(torch.abs(fft) + 1e-8)
    return mag

if __name__ == "__main__":
    # --- Device ---
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Using device:", device)

    # --- Load model ---
    model = DualStreamFusion(num_classes=10)
    state_dict = torch.load("Models/best_dual_fusion.pt", map_location=device)
    new_state_dict = {k.replace("rgb_net.model.", "rgb_net."): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()

    # --- Preprocessing ---
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    # --- Dataset ---
    dataset = ImageFolder("Data/Closed_Set", transform=transform)

    # --- Subset sampling ---
    samples_per_class = 500
    indices = []
    class_indices = {i: [] for i in range(len(dataset.classes))}
    for idx, (_, label) in enumerate(dataset):
        class_indices[label].append(idx)
    for i in range(len(dataset.classes)):
        if len(class_indices[i]) <= samples_per_class:
            indices.extend(class_indices[i])
        else:
            indices.extend(random.sample(class_indices[i], samples_per_class))

    subset_dataset = Subset(dataset, indices)
    loader = DataLoader(subset_dataset, batch_size=64, shuffle=False,
                        num_workers=4, persistent_workers=True)

    # --- Evaluation ---
    correct = 0
    total = 0
    num_classes = len(dataset.classes)
    class_correct = [0]*num_classes
    class_total = [0]*num_classes

    for rgb, label in loader:
        rgb = rgb.to(device)
        freq = fft_transform(rgb).to(device)
        label = label.to(device)

        with torch.no_grad():
            outputs = model(rgb, freq)
            pred = outputs.argmax(dim=1)

        correct += (pred == label).sum().item()
        total += label.size(0)

        for i in range(len(label)):
            class_total[label[i]] += 1
            if pred[i] == label[i]:
                class_correct[label[i]] += 1

    # --- Print results ---
    print(f"\nOverall Accuracy: {100 * correct / total:.2f}%\n")
    for i, class_name in enumerate(dataset.classes):
        if class_total[i] > 0:
            acc = 100 * class_correct[i] / class_total[i]
            print(f"Class '{class_name}': {acc:.2f}% ({class_correct[i]}/{class_total[i]})")
        else:
            print(f"Class '{class_name}': No samples")