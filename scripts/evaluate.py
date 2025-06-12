import torch # type: ignore
import sidechainnet as scn # type: ignore
import os

from models.mim_net import MimNet
from train.losses import design_loss, folding_loss, regularization_total_var

def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    total_fold = 0
    design_total = 0
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            one_hot_encoding = batch.seqs_onehot.permute(0, 2, 1).float().to(device)
            pssm = torch.nan_to_num(batch.evolutionary, nan=0.0)
            coords = torch.nan_to_num(batch.coords, nan=0.0)
            if pssm.shape[2] < 20:
                pad_width = 20 - pssm.shape[2]
                padding = torch.zeros(pssm.shape[0], pssm.shape[1], pad_width, device=pssm.device)
                pssm = torch.cat([pssm, padding], dim=2)
            else:
                pssm = pssm[:, :, :20]
            pssm = pssm.permute(0, 2, 1).float().to(device)
            true_seq = torch.cat([one_hot_encoding, pssm], dim=1)
            coords_ca = coords[:, :, 1, :].float().to(device)
            true_3d = coords_ca.permute(0, 2, 1).contiguous()
            mask = batch.masks.float().to(device)
            M = mask.unsqueeze(1) * mask.unsqueeze(2)

            expected_3d, expected_seq = model.forward_joint(true_seq)

            fold_loss = folding_loss(expected_3d, true_3d, M)
            des_loss = design_loss(expected_seq, true_seq)
            loss = fold_loss + des_loss

            total_loss += loss.item()
            total_fold += fold_loss.item()
            design_total += des_loss.item()

    avg_loss = total_loss / (i) # type: ignore
    avg_fold = total_fold / (i) # type: ignore
    avg_design = design_total / (i) # type: ignore
    print(f"Test Loss: {total_loss:.4f}, Folding: {total_fold:.4f}, Design: {design_total:.4f}")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloaders = scn.load(casp_version=7, with_pytorch="dataloaders", batch_size=16)
    val_loader = dataloaders["test"]

    model = MimNet(in_channels=40, nf=128, T=6, n_levels=3).to(device)
    checkpoint_path = "checkpoints/mimnet_model.pth"
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    evaluate(model, val_loader, device)