import os
import re
import torch
import matplotlib.pyplot as plt
import numpy as np

# Directory with model files
model_dir = "./model"

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
        })

# Sort by sparsity for a nicer plot
models = sorted(models, key=lambda x: x["sparsity"])

# --- (a) Sparsity vs Accuracy Plot ---
plt.figure(figsize=(7, 5))
plt.plot([m["sparsity"] for m in models],
         [m["acc"] for m in models],
         marker="o", linestyle="-", color="b")
plt.xlabel("Overall Sparsity (%)")
plt.ylabel("Accuracy (%)")
plt.title("Sparsity vs Accuracy Comparison")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(model_dir, "sparsity_vs_accuracy.png"))
plt.close()

# --- (b) Sparsity mask visualization ---
layer_name = "features.0.weight"  # Example layer; change as needed

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
    plt.tight_layout()

    # Save using the filename (no extension)
    save_name = os.path.splitext(m["name"])[0] + "_mask.png"
    plt.savefig(os.path.join(model_dir, save_name))
    plt.close()
