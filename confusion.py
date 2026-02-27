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
import numpy as np

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

def evaluate_with_confusion(model, dataset, device, samples_per_class=500):
    """Evaluate model and track confusion matrix"""
    indices = []
    class_indices = {i: [] for i in range(len(dataset.classes) if hasattr(dataset, 'classes') else 10)}
    
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        if label in class_indices:
            class_indices[label].append(idx)
    
    for i in class_indices.keys():
        if len(class_indices[i]) <= samples_per_class:
            indices.extend(class_indices[i])
        else:
            indices.extend(random.sample(class_indices[i], samples_per_class))

    if len(indices) == 0:
        print("    WARNING: No valid indices found!")
        return 0, {}, np.zeros((10, 10)), []

    subset_dataset = Subset(dataset, indices)
    loader = DataLoader(subset_dataset, batch_size=64, shuffle=False,
                        num_workers=4, persistent_workers=True)

    num_classes = len(dataset.classes) if hasattr(dataset, 'classes') else 10
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
    
    correct = 0
    total = 0
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
            true_label = label[i].item()
            pred_label = pred[i].item()
            
            confusion_matrix[true_label][pred_label] += 1
            class_total[true_label] += 1
            
            if pred_label == true_label:
                class_correct[true_label] += 1

    overall_acc = 100 * correct / total if total > 0 else 0
    class_names = dataset.classes if hasattr(dataset, 'classes') else []
    class_accs = {}
    for i, class_name in enumerate(class_names):
        if class_total[i] > 0:
            class_accs[class_name] = 100 * class_correct[i] / class_total[i]
        else:
            class_accs[class_name] = 0
    
    return overall_acc, class_accs, confusion_matrix, class_names

def analyze_misclassifications(confusion_matrix, class_names):
    """Analyze confusion matrix to find most common misclassifications"""
    num_classes = len(class_names)
    misclass_data = []
    
    for true_idx in range(num_classes):
        true_class = class_names[true_idx]
        total_samples = confusion_matrix[true_idx].sum()
        correct_samples = confusion_matrix[true_idx][true_idx]
        incorrect_samples = total_samples - correct_samples
        
        if total_samples == 0:
            continue
        
        misclass_counts = confusion_matrix[true_idx].copy()
        misclass_counts[true_idx] = -1
        
        if incorrect_samples > 0:
            most_confused_idx = misclass_counts.argmax()
            most_confused_class = class_names[most_confused_idx]
            most_confused_count = confusion_matrix[true_idx][most_confused_idx]
            most_confused_pct = 100 * most_confused_count / incorrect_samples
            
            misclass_counts[most_confused_idx] = -1
            if misclass_counts.max() > 0:
                second_confused_idx = misclass_counts.argmax()
                second_confused_class = class_names[second_confused_idx]
                second_confused_count = confusion_matrix[true_idx][second_confused_idx]
                second_confused_pct = 100 * second_confused_count / incorrect_samples
            else:
                second_confused_class = "N/A"
                second_confused_count = 0
                second_confused_pct = 0
        else:
            most_confused_class = "N/A"
            most_confused_count = 0
            most_confused_pct = 0
            second_confused_class = "N/A"
            second_confused_count = 0
            second_confused_pct = 0
        
        accuracy = 100 * correct_samples / total_samples if total_samples > 0 else 0
        error_rate = 100 * incorrect_samples / total_samples if total_samples > 0 else 0
        
        misclass_data.append({
            'True_Class': true_class,
            'Accuracy': accuracy,
            'Error_Rate': error_rate,
            'Total_Errors': incorrect_samples,
            'Most_Confused_With': most_confused_class,
            'Most_Confused_Count': most_confused_count,
            'Most_Confused_Pct': most_confused_pct,
            'Second_Most_Confused_With': second_confused_class,
            'Second_Most_Confused_Count': second_confused_count,
            'Second_Most_Confused_Pct': second_confused_pct
        })
    
    return pd.DataFrame(misclass_data)

if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Using device:", device)

    model = DualStreamFusion(num_classes=10)
    state_dict = torch.load("Models/best_dual_fusion.pt", map_location=device)
    new_state_dict = {k.replace("rgb_net.model.", "rgb_net."): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    base_path = Path.home() / "Desktop" / "School" / "Grade 16" / "STATS 229" / "Project" / "Data"
    original_path = base_path / "Closed_Set"
    postprocessed_path = base_path / "Post-Processed" / "Closed_Set"

    original_dataset = ImageFolder(str(original_path), transform=transform)
    class_names = original_dataset.classes
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    
    print(f"Classes: {class_names}\n")

    print("=" * 80)
    print("STEP3 MISCLASSIFICATION ANALYSIS")
    print("=" * 80)

    step = 'step3'
    
    ai_types = []
    if postprocessed_path.exists():
        ai_types = [d.name for d in postprocessed_path.iterdir() if d.is_dir()]
        ai_types.sort()
    
    print(f"\nFound AI types: {ai_types}\n")
    
    all_confusion_matrices = {}
    all_accuracies = {}
    
    for ai_type in ai_types:
        step_path = postprocessed_path / ai_type / "PP" / step
        
        if not step_path.exists():
            print(f"Warning: Path does not exist: {step_path}")
            continue
        
        print(f"\nEvaluating {ai_type} - {step}:")
        print(f"  Path: {step_path}")
        
        try:
            subdirs = [d for d in step_path.iterdir() if d.is_dir()]
            image_files = list(step_path.glob('*.jpg')) + list(step_path.glob('*.png')) + \
                         list(step_path.glob('*.JPEG')) + list(step_path.glob('*.PNG'))
            
            if subdirs and not image_files:
                print(f"  Structure: Subdirectories detected")
                dataset = ImageFolder(str(step_path), transform=transform)
                print(f"  Classes in dataset: {dataset.classes}")
            else:
                print(f"  Structure: Flat (images directly in folder)")
                if ai_type not in class_to_idx:
                    print(f"  WARNING: '{ai_type}' not found in original classes {class_names}")
                    continue
                
                class_idx = class_to_idx[ai_type]
                dataset = SingleClassDataset(step_path, class_idx, transform=transform)
                dataset.classes = class_names
            
            overall_acc, class_accs, conf_matrix, _ = evaluate_with_confusion(
                model, dataset, device
            )
            
            all_confusion_matrices[ai_type] = conf_matrix
            all_accuracies[ai_type] = class_accs
            
            print(f"  Overall Accuracy: {overall_acc:.2f}%")
            
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    if all_confusion_matrices:
        combined_confusion = sum(all_confusion_matrices.values())
        
        print("\n" + "=" * 80)
        print("COMBINED STEP3 MISCLASSIFICATION ANALYSIS")
        print("=" * 80)
        
        misclass_df = analyze_misclassifications(combined_confusion, class_names)
        print("\n" + misclass_df.to_string(index=False))
        
        print("\n" + "=" * 80)
        print("COMBINED STEP3 CONFUSION MATRIX")
        print("=" * 80)
        print("\nRows = True Class, Columns = Predicted Class\n")
        
        confusion_df = pd.DataFrame(
            combined_confusion,
            index=class_names,
            columns=class_names
        )
        print(confusion_df.to_string())
        
        print("\n" + "=" * 80)
        print("PER-AI-TYPE MISCLASSIFICATION ANALYSIS")
        print("=" * 80)
        
        all_misclass_data = []
        
        for ai_type in ai_types:
            if ai_type in all_confusion_matrices:
                print(f"\n--- {ai_type} ---")
                misclass_df_individual = analyze_misclassifications(
                    all_confusion_matrices[ai_type], 
                    class_names
                )
                misclass_df_individual['AI_Type'] = ai_type
                all_misclass_data.append(misclass_df_individual)
                
                print(misclass_df_individual[['True_Class', 'Accuracy', 'Most_Confused_With', 'Most_Confused_Pct']].to_string(index=False))
        
        misclass_df.to_csv("step3_combined_misclassification_analysis.csv", index=False)
        confusion_df.to_csv("step3_combined_confusion_matrix.csv")
        
        if all_misclass_data:
            all_misclass_df = pd.concat(all_misclass_data, ignore_index=True)
            all_misclass_df.to_csv("step3_per_ai_type_misclassification_analysis.csv", index=False)
        
        print(f"\n\nResults saved to:")
        print("  - step3_combined_misclassification_analysis.csv")
        print("  - step3_combined_confusion_matrix.csv")
        print("  - step3_per_ai_type_misclassification_analysis.csv")
        
        print("\n" + "=" * 80)
        print("SUMMARY: WHAT EACH AI TYPE IS MOST CONFUSED WITH IN STEP3")
        print("=" * 80)
        
        summary_data = []
        for ai_type in ai_types:
            if ai_type in all_confusion_matrices:
                conf_matrix = all_confusion_matrices[ai_type]
                
                # Get index for this AI type
                if ai_type in class_names:
                    ai_idx = class_names.index(ai_type)
                    total_samples = conf_matrix[ai_idx].sum()
                    correct_samples = conf_matrix[ai_idx][ai_idx]
                    incorrect_samples = total_samples - correct_samples
                    
                    if total_samples > 0:
                        accuracy = 100 * correct_samples / total_samples
                        
                        # Find most confused with
                        misclass_counts = conf_matrix[ai_idx].copy()
                        misclass_counts[ai_idx] = -1
                        
                        if incorrect_samples > 0:
                            most_confused_idx = misclass_counts.argmax()
                            most_confused_class = class_names[most_confused_idx]
                            most_confused_count = conf_matrix[ai_idx][most_confused_idx]
                            most_confused_pct = 100 * most_confused_count / total_samples
                            
                            # Second most confused
                            misclass_counts[most_confused_idx] = -1
                            if misclass_counts.max() > 0:
                                second_confused_idx = misclass_counts.argmax()
                                second_confused_class = class_names[second_confused_idx]
                                second_confused_count = conf_matrix[ai_idx][second_confused_idx]
                                second_confused_pct = 100 * second_confused_count / total_samples
                            else:
                                second_confused_class = "N/A"
                                second_confused_pct = 0
                        else:
                            most_confused_class = "N/A"
                            most_confused_pct = 0
                            second_confused_class = "N/A"
                            second_confused_pct = 0
                        
                        summary_data.append({
                            'AI_Type': ai_type,
                            'Accuracy': accuracy,
                            'Total_Samples': total_samples,
                            'Correct': correct_samples,
                            'Errors': incorrect_samples,
                            'Most_Confused_With': most_confused_class,
                            'Most_Confused_Pct': most_confused_pct,
                            'Second_Most_Confused_With': second_confused_class,
                            'Second_Most_Confused_Pct': second_confused_pct
                        })
        
        summary_df = pd.DataFrame(summary_data)
        print("\n" + summary_df.to_string(index=False))
        
        summary_df.to_csv("step3_summary_confusion.csv", index=False)
        print("\n  - step3_summary_confusion.csv")