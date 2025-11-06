import os
import re
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# =========================
# Configuration
# =========================
checkpoint_dir = "./model"   # folder with .pt files
output_dir = Path("./plots")       # output folder for figures
output_dir.mkdir(parents=True, exist_ok=True)
layer_name = None  # set to None to auto-detect a 4D conv layer (recommended)

# =========================
# Step 1: Parse file names
# =========================
pattern = re.compile(r"(\w+)_([\w\d]+)_unstructured_(omp|imp)_(\d+\.\d+)_acc_(\d+\.\d+)\.pt")

file_data = []
for fname in os.listdir(checkpoint_dir):
    if fname.endswith(".pt"):
        match = pattern.match(fname)
        if match:
            dataset, model, method, sparsity, acc = match.groups()
            sparsity = float(sparsity)
            acc = float(acc)
            file_data.append({
                "filename": os.path.join(checkpoint_dir, fname),
                "model_name": f"{model}_{method}",
                "sparsity": sparsity,
                "accuracy": acc
            })

# Sort by sparsity
file_data.sort(key=lambda x: x["sparsity"])

model_names = [d["model_name"] for d in file_data]
sparsity_ratios = [d["sparsity"] / 100 for d in file_data]
accuracies = [d["accuracy"] for d in file_data]

# =========================
# Step 2: Sparsity vs Accuracy plot
# =========================
plt.figure(figsize=(6,4))
plt.scatter(sparsity_ratios, accuracies, color='dodgerblue', s=80)
for i, name in enumerate(model_names):
    plt.annotate(name, (sparsity_ratios[i], accuracies[i]), xytext=(5,5),
                 textcoords='offset points', fontsize=8)
plt.xlabel("Overall Sparsity Ratio (fraction of weights pruned)")
plt.ylabel("Accuracy (%)")
plt.title("Sparsity vs Accuracy Comparison")
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()

acc_plot_path = output_dir / "sparsity_vs_accuracy.png"
plt.savefig(acc_plot_path, dpi=200)
plt.close()
print(f"Saved {acc_plot_path}")

# =========================
# Step 3: Sparsity mask visualization for one layer
# =========================
for d in file_data:
    print(f"Processing {d['filename']} ...")
    checkpoint = torch.load(d["filename"], map_location="cpu")

    # Auto-detect one 4D layer if not specified
    if layer_name is None:
        layer_name = next((k for k, v in checkpoint.items() if isinstance(v, torch.Tensor) and v.ndim == 4), None)
        if layer_name is None:
            print(f"⚠️ No 4D conv layers found in {d['filename']}. Skipping.")
            continue

    if layer_name not in checkpoint:
        print(f"⚠️ Layer {layer_name} not found in {d['filename']}. Skipping.")
        continue

    weight = checkpoint[layer_name]
    mask = weight != 0  # Boolean mask

    # reshape to (filters, channels*k_h*k_w)
    F, C, H, W = mask.shape
    mask_2d = mask.view(F, C * H * W).numpy()

    plt.figure(figsize=(8,6))
    plt.imshow(mask_2d, aspect='auto', interpolation='nearest', cmap='gray_r')
    plt.title(f"{d['model_name']} — Layer: {layer_name}")
    plt.xlabel("Weights per filter (C×H×W)")
    plt.ylabel("Filter index")
    plt.colorbar(label="Nonzero (white=0, black=1)")
    plt.tight_layout()

    mask_path = output_dir / f"mask_{d['model_name']}.png"
    plt.savefig(mask_path, dpi=200)
    plt.close()
    print(f"Saved {mask_path}")
