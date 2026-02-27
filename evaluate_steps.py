import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights, vit_b_16, ViT_B_16_Weights
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision.datasets import ImageFolder
import random
import pandas as pd
from pathlib import Path
from PIL import Image

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

class SingleClassDataset(Dataset):
    """Dataset for augmented images that are all from one AI type"""
    def __init__(self, image_dir, class_label, transform=None):
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.class_label = class_label
        
        # Get all image files
        self.image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPEG', '*.JPG', '*.PNG']:
            self.image_files.extend(list(self.image_dir.glob(ext)))
        
        print(f"    Found {len(self.image_files)} images")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, self.class_label

def evaluate_dataset(model, dataset, device, samples_per_class=500):
    """Evaluate model on a dataset and return overall and per-class accuracy"""
    # --- Subset sampling ---
    indices = []
    class_indices = {i: [] for i in range(len(dataset.classes) if hasattr(dataset, 'classes') else 10)}
    
    # Collect indices by class
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        if label in class_indices:
            class_indices[label].append(idx)
    
    # Sample from each class
    for i in class_indices.keys():
        if len(class_indices[i]) <= samples_per_class:
            indices.extend(class_indices[i])
        else:
            indices.extend(random.sample(class_indices[i], samples_per_class))

    if len(indices) == 0:
        print("    WARNING: No valid indices found!")
        return 0, {}, []

    subset_dataset = Subset(dataset, indices)
    loader = DataLoader(subset_dataset, batch_size=64, shuffle=False,
                        num_workers=4, persistent_workers=True)

    # --- Evaluation ---
    correct = 0
    total = 0
    num_classes = len(dataset.classes) if hasattr(dataset, 'classes') else 10
    class_correct = [0] * num_classes
    class_total = [0] * num_classes

    model.eval()
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

    # Calculate accuracies
    overall_acc = 100 * correct / total if total > 0 else 0
    class_names = dataset.classes if hasattr(dataset, 'classes') else [f"class_{i}" for i in range(num_classes)]
    class_accs = {}
    for i, class_name in enumerate(class_names):
        if class_total[i] > 0:
            class_accs[class_name] = 100 * class_correct[i] / class_total[i]
        else:
            class_accs[class_name] = 0
    
    return overall_acc, class_accs, class_names

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
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # --- Base paths ---
    base_path = Path.home() / "Desktop" / "School" / "Grade 16" / "STATS 229" / "Project" / "Data"
    original_path = base_path / "Closed_Set"
    postprocessed_path = base_path / "Post-Processed" / "Closed_Set"

    # --- Evaluate on original dataset ---
    print("=" * 80)
    print("Evaluating on ORIGINAL dataset (no augmentation)")
    print("=" * 80)
    original_dataset = ImageFolder(str(original_path), transform=transform)
    overall_acc, class_accs, class_names = evaluate_dataset(model, original_dataset, device)
    
    # Create class name to index mapping
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    
    print(f"\nOverall Accuracy: {overall_acc:.2f}%\n")
    print(f"Classes found: {class_names}\n")
    for class_name in class_names:
        print(f"Class '{class_name}': {class_accs[class_name]:.2f}%")

    # --- Storage for results ---
    results = {}
    results['Original'] = {'Overall': overall_acc, **class_accs}

    # --- Augmentation steps ---
    aug_steps = ['step1', 'step2', 'step3']

    # --- Get AI types from directory ---
    ai_types = []
    if postprocessed_path.exists():
        ai_types = [d.name for d in postprocessed_path.iterdir() if d.is_dir()]
        ai_types.sort()
    
    print(f"\nFound AI types: {ai_types}")

    # --- Evaluate on augmented datasets ---
    for step in aug_steps:
        print("\n" + "=" * 80)
        print(f"Evaluating on {step.upper()} augmentation")
        print("=" * 80)
        
        for ai_type in ai_types:
            step_path = postprocessed_path / ai_type / "PP" / step
            
            if not step_path.exists():
                print(f"Warning: Path does not exist: {step_path}")
                continue
            
            print(f"\n{ai_type} - {step}:")
            print(f"  Path: {step_path}")
            
            try:
                # Check if it's organized in subdirectories or flat
                subdirs = [d for d in step_path.iterdir() if d.is_dir()]
                image_files = list(step_path.glob('*.jpg')) + list(step_path.glob('*.png')) + \
                             list(step_path.glob('*.JPEG')) + list(step_path.glob('*.PNG'))
                
                if subdirs and not image_files:
                    # Has subdirectories - use ImageFolder
                    print(f"  Structure: Subdirectories detected")
                    dataset = ImageFolder(str(step_path), transform=transform)
                    print(f"  Classes in dataset: {dataset.classes}")
                else:
                    # Flat structure - images directly in folder
                    print(f"  Structure: Flat (images directly in folder)")
                    # Get the class index for this AI type
                    if ai_type not in class_to_idx:
                        print(f"  WARNING: '{ai_type}' not found in original classes {class_names}")
                        continue
                    
                    class_idx = class_to_idx[ai_type]
                    dataset = SingleClassDataset(step_path, class_idx, transform=transform)
                    # Add classes attribute for compatibility
                    dataset.classes = class_names
                
                overall_acc, class_accs, _ = evaluate_dataset(model, dataset, device)
                
                # Store results
                key = f"{step}_{ai_type}"
                results[key] = {'Overall': overall_acc, **class_accs}
                
                print(f"  Overall Accuracy: {overall_acc:.2f}%")
                print(f"  {ai_type} Accuracy: {class_accs.get(ai_type, 0):.2f}%")
                        
            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()

    # --- Create comprehensive results table ---
    print("\n" + "=" * 80)
    print("COMPREHENSIVE RESULTS TABLE")
    print("=" * 80)

    # Organize with AI types as rows and augmentation steps as columns
    table_data = {'AI_Type': class_names}
    
    # Add Original column
    table_data['Original'] = [results['Original'].get(ai_type, 0) for ai_type in class_names]
    
    # Add columns for each augmentation step
    for step in aug_steps:
        step_column = []
        for ai_type in class_names:
            key = f"{step}_{ai_type}"
            if key in results:
                # Use the accuracy for this specific AI type
                step_column.append(results[key].get(ai_type, 0))
            else:
                step_column.append(0)
        table_data[step] = step_column
    
    df = pd.DataFrame(table_data)
    print("\n" + df.to_string(index=False))
    
    # --- Save to CSV ---
    output_file = "augmentation_results.csv"
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")
    
    # --- Create detailed per-class table ---
    print("\n" + "=" * 80)
    print("PER-CLASS ACCURACY TABLE")
    print("=" * 80)
    
    detailed_data = []
    for key, vals in results.items():
        row = {'Dataset': key}
        row.update({class_name: vals.get(class_name, 0) for class_name in class_names})
        row['Overall'] = vals['Overall']
        detailed_data.append(row)
    
    detailed_df = pd.DataFrame(detailed_data)
    print("\n" + detailed_df.to_string(index=False))
    
    detailed_output_file = "detailed_augmentation_results.csv"
    detailed_df.to_csv(detailed_output_file, index=False)
    print(f"\nDetailed results saved to {detailed_output_file}")