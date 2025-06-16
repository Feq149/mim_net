import matplotlib.pyplot as plt
import itertools
from train.utils import process_batch
import torch

def plot_real_vs_predicted(model, dataloader, device, n_samples=8):
    model.eval()
    fig = plt.figure(figsize=(16, 8))
    for i, batch in enumerate(itertools.islice(dataloader, n_samples)):
        true_seq, true_3d, M = process_batch(batch, device)
        with torch.no_grad():
            pred_3d, _ = model.forward_joint(true_seq)
        true_ca = true_3d[0].cpu().numpy()
        pred_ca = pred_3d[0].cpu().numpy()
        ax = fig.add_subplot(2, 4, i+1, projection='3d')
        ax.scatter(true_ca[0], true_ca[1], true_ca[2], c='blue', label='Real', s=10)
        ax.scatter(pred_ca[0], pred_ca[1], pred_ca[2], c='red', label='Pred', s=10)
        ax.set_title(f"Protein #{i+1}")
        ax.axis('off')
        if i == 0:
            ax.legend()
    plt.tight_layout()
    plt.show()