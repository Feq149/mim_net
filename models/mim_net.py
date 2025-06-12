import torch
import torch.nn as nn
from .embedding import InputEmbedding, OutputProjection, ReverseEmbedding, ReverseOutput
from .unet import GraphUNet
from .reversible_block import *

class MimNet(nn.Module):

    def __init__(self, in_channels=40, nf=128, T=6, n_levels=3):
        super().__init__()
        print("Inicjalizacja MimNet...")

        self.embed_in = InputEmbedding(in_channels, nf)
        self.reverse_embed = ReverseEmbedding(nf)

        self.unet = GraphUNet(nf, n_levels)
        self.reversible = ReversibleBlock(self.unet, T=T)

        self.project_out = OutputProjection(nf)
        self.reverse_out = ReverseOutput(nf)

        print("MimNet gotowy.")

    def forward_folding(self, S):
        Y0 = self.embed_in(S)
        YT, _ = self.reversible(Y0)
        pred_3d = self.project_out(YT)
        return pred_3d

    def forward_design(self, X):
        YT = self.reverse_embed(X)
        Y0 = self.reversible.reverse(YT, [YT])
        pred_se1 = self.reverse_out(Y0)
        return pred_se1

    def forward_joint(self, S):
        #print("-> Start forward_joint")
        Y0 = self.embed_in(S)
        #print("   Embedding zakończony.")
        YT, states = self.reversible(Y0)
        #print("   Przejście przez ReversibleBlock zakończone.")
        expected_3d = self.project_out(YT)
        #print("   OutputProjection zakończony.")

        Y0_recon = self.reversible.reverse(YT, states)
        #print("   Reverse ReversibleBlock zakończony.")
        expected_seq = self.reverse_out(Y0_recon)
        #print("-> forward_joint zakończony.")
        return expected_3d, expected_seq
