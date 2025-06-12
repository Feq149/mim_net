import torch # type: ignore
import sidechainnet as scn # type: ignore
import py3Dmol # type: ignore
import numpy as np # type: ignore
from models.mim_net import MimNet
import random

def coords_to_pdb(coords, chain_id="A"):
    pdb = ""
    for i, (x, y, z) in enumerate(coords.T, 1):
        pdb += (
            f"ATOM  {i:5d}  CA  ALA {chain_id}{i:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n"
        )
    pdb += "END\n"
    return pdb

def show_pred_vs_true(true_coords, pred_coords, save_path=None):
    view = py3Dmol.view(width=600, height=400)
    pdb_true = coords_to_pdb(true_coords, chain_id="A")
    pdb_pred = coords_to_pdb(pred_coords, chain_id="B")
    view.addModel(pdb_true, "pdb")
    view.setStyle({'chain':'A'}, {"cartoon": {"color": "blue"}})
    view.addModel(pdb_pred, "pdb")
    view.setStyle({'chain':'B'}, {"cartoon": {"color": "red"}})
    view.zoomTo()
    if save_path:
        view.png(save_path)
        print(f"Saved visualization to {save_path}")
    else:
        view.show()
    return view

def visualize_first_batch_3dmol(model, dataloader, device, save_path=None):
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
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

            expected_3d, _ = model.forward_joint(true_seq)

            # Only visualize the first protein in the batch
            mask_np = mask[0].cpu().bool().numpy()
            true_np = true_3d[0].cpu().numpy()[:, mask_np]
            pred_np = expected_3d[0].cpu().numpy()[:, mask_np]
            show_pred_vs_true(true_np, pred_np, save_path=save_path)
            break  # Only the first batch

def visualize_random_proteins_3dmol(model, dataloader, device, num_samples=5, save_prefix="protein_compare"):
    model.eval()
    all_batches = list(dataloader)
    indices = random.sample(range(len(all_batches)), min(num_samples, len(all_batches)))
    with torch.no_grad():
        for idx, batch_idx in enumerate(indices):
            batch = all_batches[batch_idx]
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

            expected_3d, _ = model.forward_joint(true_seq)

            mask_np = mask[0].cpu().bool().numpy()
            true_np = true_3d[0].cpu().numpy()[:, mask_np]
            pred_np = expected_3d[0].cpu().numpy()[:, mask_np]
            save_path = f"{save_prefix}_{idx+1}.png"
            show_pred_vs_true(true_np, pred_np, save_path=save_path)