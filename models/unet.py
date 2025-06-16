import torch
import torch.nn as nn
import torch.nn.functional as F

from graph_utils.graph_utils import *
from models.graph_convolution import GraphConvBlock


class GraphUNetLevel(nn.Module):
    def __init__(self, nf):
        super().__init__()
        self.gcn = GraphConvBlock(nf, nf)

    def forward(self, Y, L):
        Y = self.gcn(Y, L)
        return Y

class GraphUNet(nn.Module):
    # UNet - mamy warstwy. Chcemy przekształcić nasze dane przez n_levels warstw - u nas 3, bo tak było w pracy
    # Ładujemy dane, na każdej warstwie robimy konwolucja, laplasjan, konwolucja, po czym przy downsamplingu
    # łączymy sąsiednie cechy zwykłą średnią. Potem przy upsamplingu bierzemy na danej warstwie oryginalne cechy
    # dodajemy do tych które uzyskaliśmy z niższej warstwy (po prostu zwykły plus), po czym klasyk,
    # konwolucja laplasjan konolucja.
    def __init__(self, nf, n_levels):
        super().__init__()
        self.n_levels = n_levels
        self.down_blocks = nn.ModuleList([GraphUNetLevel(nf) for _ in range(n_levels)])
        self.up_blocks = nn.ModuleList([GraphUNetLevel(nf) for _ in range(n_levels)])
        self.pool = nn.AvgPool1d(kernel_size=2, stride=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='linear', align_corners=False)

    def forward(self, Y):
        downs = []
        graphs = []
        sizes = []

        # DOWN SAMPLING
        for i in range(self.n_levels - 1):
            D = pairwise_distance_matrix(Y)
            W = build_weight_matrix(D)
            L = build_laplacian(W)
            graphs.append(L)
            Y = self.down_blocks[i](Y, L) # wywołanie forward, syntax jest taki że się forward nie pisze xd
            downs.append(Y)
            sizes.append(Y.shape[-1])  # zapamiętujemy rozmiar, żeby potem dopasować
            if sizes[-1] >= 2:
                Y = self.pool(Y)

        # BOTTOM (najgłębszy poziom)
        D = pairwise_distance_matrix(Y)
        W = build_weight_matrix(D)
        L = build_laplacian(W)
        Y = self.down_blocks[-1](Y, L)

        # UP SAMPLING
        for i in reversed(range(self.n_levels - 1)):
            # Dopasuj długość przez przycięcie lub padding, bo upsample niekoniecznie zwróci długość dokładnie
            # równą oryginałowi
            target_len = sizes[i]
            up_len = Y.shape[-1]

            if up_len > target_len:
                Y = Y[:, :, :target_len] # trzeba ściąć idk
            elif up_len < target_len:
                pad_len = target_len - up_len
                Y = F.pad(Y, (0, pad_len))  # padding z tyłu

            Y = Y + downs[i]  # pomijamy połączenie

            Y = self.up_blocks[i](Y, graphs[i])

        return Y