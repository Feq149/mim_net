import torch
from torch import Tensor
from typing import Tuple

def sanitize_batch(batch):
    # Napraw dane zawierające nan — zastąp 0
    batch.evolutionary = torch.nan_to_num(batch.evolutionary, nan=0.0)
    batch.coords = torch.nan_to_num(batch.coords, nan=0.0)
    return batch

def process_batch(
    batch,
    device: torch.device
) -> Tuple[Tensor, Tensor, Tensor]:
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
    # chcemy żeby maska działała dla macierze odległości, więc musi być kwadratowa
    M = mask.unsqueeze(1) * mask.unsqueeze(2)

    def print_nan_stats(name: str, tensor: torch.Tensor) -> None:
        has_nan = torch.isnan(tensor).any().item()
        min_val = tensor.min().item() if not has_nan else "NaN"
        max_val = tensor.max().item() if not has_nan else "NaN"
        print(f"{name:<15} | has_nan: {has_nan} | shape: {tuple(tensor.shape)} | min: {min_val} | max: {max_val}")

    # print_nan_stats("one_hot", one_hot_encoding)
    # print_nan_stats("pssm", pssm)
    # print_nan_stats("coords_ca", coords_ca)
    # print_nan_stats("mask", mask)

    return true_seq, true_3d, M