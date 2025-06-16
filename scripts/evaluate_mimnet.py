import torch
import sidechainnet as scn
import os
from typing import Any

from models.mim_net import MimNet
from train.losses import design_loss, folding_loss
from train.utils import process_batch

def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    total_fold = 0
    design_total = 0
    with torch.no_grad():
        for _, batch in enumerate(dataloader):
            true_seq, true_3d, M = process_batch(batch, device)

            expected_3d, expected_seq = model.forward_joint(true_seq)

            fold_loss = folding_loss(expected_3d, true_3d, M)
            des_loss = design_loss(expected_seq, true_seq)
            loss = fold_loss + des_loss

            total_loss += loss.item()
            total_fold += fold_loss.item()
            design_total += des_loss.item()

    print(f"Test Loss: {total_loss / len(dataloader.dataset):.4f}, Folding: {total_fold / len(dataloader.dataset):.4f}, Design: {design_total / len(dataloader.dataset):.4f}")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloaders : Any = scn.load(casp_version=7, with_pytorch="dataloaders", batch_size=24)
    val_loader = dataloaders["test"]

    model = MimNet(in_channels=40, nf=128, T=6, n_levels=3).to(device)
    checkpoint_path = "checkpoints/mimnet_model_with_reg.pth"
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    evaluate(model, val_loader, device)