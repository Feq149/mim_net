import torch
import torch.nn as nn

class ReversibleBlock(nn.Module):
    def __init__(self, f_blocks, T=6, h=1.0):
        super().__init__()
        self.f_blocks = f_blocks # GraphUNety
        self.T = T              # liczba kroków czasowych
        self.h = h              # wielkość kroku czasowowego

    def forward(self, Y0):
        Y_prev = Y0
        Y_curr = Y0
        history = [Y0]

        for i in range(self.T):
            F = self.f_blocks[i](Y_curr)
            Y_next = 2 * Y_curr - Y_prev + self.h**2 * F
            Y_prev, Y_curr = Y_curr, Y_next
            history.append(Y_curr)

        return Y_curr, history

    def reverse(self, YT, states):
        Y_curr = YT
        Y_prev = states[-2] # dwa ostatnie bo tyle wystarcza, wystarczy odwrócić wzór.

        for i in reversed(range(self.T)):
            F = self.f_blocks[i](Y_prev)
            Y_next = 2 * Y_prev - Y_curr + self.h**2 * F
            Y_curr, Y_prev = Y_prev, Y_next

        return Y_curr  # to będzie Y_0 (czyli rekonstruowana sekwencja)
