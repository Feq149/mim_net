from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import numpy as np
import itertools
from train.utils import process_batch
import torch

def plot_real_vs_predicted(model, dataloader, device, save_path, n_samples=8):
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), subplot_kw={'projection': '3d'})
    axes = axes.flatten()
    for i, batch in enumerate(itertools.islice(dataloader, n_samples)):
        ax = axes[i]
        true_seq, true_3d, M = process_batch(batch, device)
        with torch.no_grad():
            pred_3d, _ = model.forward_joint(true_seq)
        true_ca = true_3d[0].cpu().numpy()
        pred_ca = pred_3d[0].cpu().numpy()
        ax.scatter3D(true_ca[0, :], true_ca[1, :], true_ca[2, :], c='blue', label='Real') # type: ignore
        ax.scatter3D(pred_ca[0, :], pred_ca[1, :], pred_ca[2, :], c='red', label='Pred') # type: ignore
        ax.set_title(f"Protein #{i+1}")
        if i == 0:
            ax.legend()
    plt.tight_layout()
    plt.savefig(save_path)


