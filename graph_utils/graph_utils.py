import torch
import torch.nn.functional as F

def pairwise_distance_matrix(Y):
    batch_size, num_features, seq_len = Y.shape
    # Transpozycja: (B, C, N) → (B, N, C)
    Y_transposed = Y.transpose(1, 2)
    norms_squared = torch.sum(Y_transposed ** 2, dim=2, keepdim=True)  # (B, N, 1), bo wymiary muszą się zgadzać
    #x^2 + y^2 - 2xy
    distances = norms_squared + norms_squared.transpose(1, 2) - 2 * torch.bmm(Y_transposed, Y_transposed.transpose(1, 2))
    # Zapewnienie nieujemnych wartości (zabezpieczenie przed błędami numerycznymi)
    return distances.clamp(min=1e-6)

def build_weight_matrix(D, alpha=10.0):
    return torch.exp(-D / alpha)

def build_laplacian(W):
    D = torch.sum(W, dim=2)  # (B, N)
    D_matrix = torch.diag_embed(D)  # (B, N, N)
    L = D_matrix - W
    return L
