import os
import re
import torch
import matplotlib.pyplot as plt
import numpy as np

# Directories
model_dir = "./model"
output_dir = "./plots"
os.makedirs(output_dir, exist_ok=True)

# Regex to extract pruning type, sparsity, and accuracy from filename
pattern = re.compile(r".*_(imp|omp)_(\d+\.\d+)_acc_(\d+\.\d+)\.pt")

# Collect models
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

# Sort by sparsity
models = sorted(models, key=lambda x: x["sparsity"])

# --- (a) Scatter plot: Sparsity vs Accuracy ---
plt.figure(figsize=(8, 6))
sparsities = [m["sparsity"] for m in models]
accuracies = [m["acc"] for m in models]
names = [m["name"] for m in models]

plt.scatter(sparsities, accuracies, s=100, edgecolors="k", color="dodgerblue")

# Annotate each point
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
print("Saved scatter plot.")

# --- (b) Sparsity / weight magnitude visualization for first layer ---
layer_name = "features.0.weight"

for m in models:
    state_dict = torch.load(m["path"], map_location="cpu")

    if layer_name not in state_dict:
        print(f"Skipping {m['name']} (no {layer_name})")
        continue

    w = state_dict[layer_name]
    F, C, H, W = w.shape

    # Flatten weights per filter
    w_2d = w.view(F, -1).numpy()

    plt.figure(figsize=(8, 6))
    im = plt.imshow(w_2d, cmap="viridis", aspect="auto")  # show actual weight magnitude
    plt.title(f"Weight Magnitude: {m['name']} — Layer: {layer_name}")
    plt.xlabel("Flattened Filter Weights")
    plt.ylabel("Filters")

    # Colorbar shows weight magnitude
    cbar = plt.colorbar(im)
    cbar.set_label("Weight Magnitude")

    plt.tight_layout()
    save_name = os.path.splitext(m["name"])[0] + "_first_layer_weights.png"
    plt.savefig(os.path.join(output_dir, save_name))
    plt.close()
    print(f"Saved {save_name}")
