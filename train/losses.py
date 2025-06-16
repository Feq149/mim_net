import torch
import torch.nn.functional as F

def compute_pairwise_squared_distances(coords):
    coords = coords.transpose(1, 2)
    squared_norms = torch.sum(coords ** 2, dim=2, keepdim=True)  # tak samo jak w graph_utils
    distance_matrix = squared_norms + squared_norms.transpose(1, 2) - 2 * torch.bmm(coords, coords.transpose(1, 2))
    return distance_matrix.clamp(min=1e-6)

def folding_loss(predicted_coords, true_coords, mask_matrix):
    pred_distances = compute_pairwise_squared_distances(predicted_coords)
    true_distances = compute_pairwise_squared_distances(true_coords)
    difference = (pred_distances - true_distances) * mask_matrix
    normalization = mask_matrix.sum(dim=(1, 2)).clamp(min=1)
    loss_per_sample = torch.sqrt((difference ** 2).sum(dim=(1, 2)) / normalization)
    return loss_per_sample.sum()

def design_loss(predicted, target):
    pred = predicted.transpose(1, 2) # (batch_size, seq_len, 40)
    true = target.transpose(1, 2)    # (batch_size, seq_len, 40)
    return F.kl_div(F.log_softmax(pred, dim=-1), true, reduction='sum')

def regularization_total_var(f_blocks):
    tv = 0.0
    for i in range(len(f_blocks) - 1):
        params1 = list(f_blocks[i].parameters())
        params2 = list(f_blocks[i+1].parameters())
        for p1, p2 in zip(params1, params2):
             tv += torch.sum(torch.abs(p2 - p1))
    return tv
