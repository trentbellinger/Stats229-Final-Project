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

def evaluate_open_set(model, dataset, device, known_classes, samples_per_class=500, confidence_threshold=0.7):
    """Evaluate model on open set where all classes are unknown."""
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
        return {}, []

    subset_dataset = Subset(dataset, indices)
    loader = DataLoader(subset_dataset, batch_size=64, shuffle=False,
                        num_workers=4, persistent_workers=True)

    class_names = dataset.classes if hasattr(dataset, 'classes') else []
    num_known = len(known_classes)
    
    class_metrics = {class_name: {
        'total': 0,
        'avg_confidence': [],
        'predicted_as': [0] * num_known,
        'high_confidence_count': 0,
    } for class_name in class_names}

    model.eval()
    for rgb, label in loader:
        rgb = rgb.to(device)
        freq = fft_transform(rgb).to(device)
        label = label.to(device)

        with torch.no_grad():
            outputs = model(rgb, freq)
            probs = torch.softmax(outputs, dim=1)
            confidences, pred = torch.max(probs, dim=1)

        for i in range(len(label)):
            true_class = class_names[label[i].item()]
            predicted_class_idx = pred[i].item()
            confidence = confidences[i].item()
            
            class_metrics[true_class]['total'] += 1
            class_metrics[true_class]['avg_confidence'].append(confidence)
            class_metrics[true_class]['predicted_as'][predicted_class_idx] += 1
            
            if confidence >= confidence_threshold:
                class_metrics[true_class]['high_confidence_count'] += 1

    results = {}
    for class_name in class_names:
        if class_metrics[class_name]['total'] > 0:
            avg_conf = np.mean(class_metrics[class_name]['avg_confidence'])
            high_conf_pct = 100 * class_metrics[class_name]['high_confidence_count'] / class_metrics[class_name]['total']
            
            most_predicted_idx = np.argmax(class_metrics[class_name]['predicted_as'])
            most_predicted_class = known_classes[most_predicted_idx]
            most_predicted_pct = 100 * class_metrics[class_name]['predicted_as'][most_predicted_idx] / class_metrics[class_name]['total']
            
            results[class_name] = {
                'avg_confidence': avg_conf,
                'high_confidence_pct': high_conf_pct,
                'most_predicted_as': most_predicted_class,
                'most_predicted_pct': most_predicted_pct,
                'total_samples': class_metrics[class_name]['total']
            }
    
    return results, class_names

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
    
    closed_set_path = base_path / "Closed_Set"
    closed_dataset = ImageFolder(str(closed_set_path), transform=transform)
    known_classes = closed_dataset.classes
    print(f"Known classes (from training): {known_classes}\n")
    
    original_path = base_path / "Open_Set"
    postprocessed_path = base_path / "Post-Processed" / "Open_Set"

    print(f"\n{'='*80}")
    print(f"EVALUATING ON OPEN SET")
    print(f"{'='*80}\n")

    print("=" * 80)
    print(f"Evaluating on ORIGINAL OPEN SET (no augmentation)")
    print("=" * 80)
    original_dataset = ImageFolder(str(original_path), transform=transform)
    
    print(f"Unknown classes in Open Set: {original_dataset.classes}\n")
    
    results_open, class_names = evaluate_open_set(model, original_dataset, device, known_classes)
    
    print("\nOpen Set Evaluation Results:")
    print("-" * 80)
    for class_name in class_names:
        if class_name in results_open:
            r = results_open[class_name]
            print(f"\n{class_name} (UNKNOWN class):")
            print(f"  Average Confidence: {r['avg_confidence']:.3f}")
            print(f"  High Confidence Predictions (>0.7): {r['high_confidence_pct']:.2f}%")
            print(f"  Most Often Predicted As: {r['most_predicted_as']} ({r['most_predicted_pct']:.2f}%)")
            print(f"  Total Samples: {r['total_samples']}")
    
    results = {'Original': results_open}
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    aug_steps = ['step1', 'step2', 'step3']

    ai_types = []
    if postprocessed_path.exists():
        ai_types = [d.name for d in postprocessed_path.iterdir() if d.is_dir()]
        ai_types.sort()
    
    print(f"\nFound AI types in post-processed data: {ai_types}")

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
                
                step_results, _ = evaluate_open_set(model, dataset, device, known_classes)
                
                key = f"{step}_{ai_type}"
                results[key] = step_results
                
                if ai_type in step_results:
                    r = step_results[ai_type]
                    print(f"  Average Confidence: {r['avg_confidence']:.3f}")
                    print(f"  Most Often Predicted As: {r['most_predicted_as']} ({r['most_predicted_pct']:.2f}%)")
                        
            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()

    # --- Create tables ---
    print("\n" + "=" * 80)
    print(f"COMPREHENSIVE RESULTS TABLE - OPEN SET")
    print("=" * 80)

    # Table 1: Average Confidence
    print("\n--- AVERAGE CONFIDENCE (Lower = Better Detection of Unknown) ---")
    table_data_conf = {'AI_Type': class_names}
    
    table_data_conf['Original'] = [
        results['Original'].get(ai_type, {}).get('avg_confidence', 0) 
        for ai_type in class_names
    ]
    
    for step in aug_steps:
        step_column = []
        for ai_type in class_names:
            key = f"{step}_{ai_type}"
            if key in results and ai_type in results[key]:
                step_column.append(results[key][ai_type]['avg_confidence'])
            else:
                step_column.append(0)
        table_data_conf[step] = step_column
    
    df_conf = pd.DataFrame(table_data_conf)
    print("\n" + df_conf.to_string(index=False))
    
    # Table 2: High Confidence Percentage
    print("\n\n--- HIGH CONFIDENCE % (>0.7) (Lower = Better) ---")
    table_data_high_conf = {'AI_Type': class_names}
    
    table_data_high_conf['Original'] = [
        results['Original'].get(ai_type, {}).get('high_confidence_pct', 0) 
        for ai_type in class_names
    ]
    
    for step in aug_steps:
        step_column = []
        for ai_type in class_names:
            key = f"{step}_{ai_type}"
            if key in results and ai_type in results[key]:
                step_column.append(results[key][ai_type]['high_confidence_pct'])
            else:
                step_column.append(0)
        table_data_high_conf[step] = step_column
    
    df_high_conf = pd.DataFrame(table_data_high_conf)
    print("\n" + df_high_conf.to_string(index=False))
    
    # Table 3: Most Predicted As
    print("\n\n--- MOST OFTEN PREDICTED AS (Known Class) ---")
    table_data_pred = {'AI_Type': class_names}
    
    table_data_pred['Original'] = [
        results['Original'].get(ai_type, {}).get('most_predicted_as', 'N/A') 
        for ai_type in class_names
    ]
    
    for step in aug_steps:
        step_column = []
        for ai_type in class_names:
            key = f"{step}_{ai_type}"
            if key in results and ai_type in results[key]:
                step_column.append(results[key][ai_type]['most_predicted_as'])
            else:
                step_column.append('N/A')
        table_data_pred[step] = step_column
    
    df_pred = pd.DataFrame(table_data_pred)
    print("\n" + df_pred.to_string(index=False))
    
    # --- Save to CSV ---
    df_conf.to_csv("open_set_avg_confidence.csv", index=False)
    df_high_conf.to_csv("open_set_high_confidence_pct.csv", index=False)
    df_pred.to_csv("open_set_predicted_as.csv", index=False)
    
    print(f"\n\nResults saved to CSV files:")
    print("  - open_set_avg_confidence.csv")
    print("  - open_set_high_confidence_pct.csv")
    print("  - open_set_predicted_as.csv")