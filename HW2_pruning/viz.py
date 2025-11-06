import os
import re
import torch
import matplotlib.pyplot as plt
import numpy as np

# Directory with model files
model_dir = "./model"
output_dir = "./plots"

# Regex to extract sparsity and accuracy from filename
pattern = re.compile(r".*_(imp|omp)_(\d+\.\d+)_acc_(\d+\.\d+)\.pt")

models = []
for fname in os.listdir(model_dir):
    match = pattern.match(fname)
    if match:
        pruning_type, sparsity, acc = match.groups()
        sparsity = float(sparsity)
        acc = float(acc)
        models.append({
            "name": fname,
            "path": os.path.join(model_dir, fname),
            "sparsity": sparsity,
            "acc": acc,
            "type": pruning_type
        })

# Sort for consistency
models = sorted(models, key=lambda x: x["sparsity"])

# --- (a) Scatter Plot: Sparsity vs Accuracy ---
plt.figure(figsize=(8, 6))
sparsities = [m["sparsity"] for m in models]
accuracies = [m["acc"] for m in models]
names = [m["name"] for m in models]

# Color by sparsity
sc = plt.scatter(sparsities, accuracies, c=sparsities, cmap="viridis", s=100, edgecolors="k")

# Add colorbar for sparsity
cbar = plt.colorbar(sc)
cbar.set_label("Sparsity (%)")

# Annotate each point with filename (shortened if long)
for i, name in enumerate(names):
    short_name = os.path.splitext(name)[0]
    plt.text(sparsities[i] + 0.3, accuracies[i], short_name, fontsize=7, va="center")

plt.xlabel("Overall Sparsity (%)")
plt.ylabel("Accuracy (%)")
plt.title("Sparsity vs Accuracy (Model Comparison)")
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "sparsity_vs_accuracy_scatter.png"))
plt.close()

# --- (b) Sparsity Mask Visualization ---
layer_name = "features.0.weight"  # Example layer name

for m in models:
    state_dict = torch.load(m["path"], map_location="cpu")

    if layer_name not in state_dict:
        print(f"Skipping {m['name']} (no {layer_name})")
        continue

    w = state_dict[layer_name]
    out_channels, in_channels, kh, kw = w.shape
    w_2d = w.view(out_channels, -1)
    mask = (w_2d != 0).float()

    plt.figure(figsize=(8, 6))
    plt.imshow(mask, cmap="coolwarm", aspect="auto")
    plt.title(f"Sparsity Mask: {m['name']}")
    plt.xlabel("Flattened Filter Weights")
    plt.ylabel("Filters")

    # Add colorbar to distinguish zeros/nonzeros
    cbar = plt.colorbar()
    cbar.set_label("Weight Presence (0 = pruned, 1 = kept)")

    plt.tight_layout()

    save_name = os.path.splitext(m["name"])[0] + "_mask.png"
    plt.savefig(os.path.join(model_dir, save_name))
    plt.close()
