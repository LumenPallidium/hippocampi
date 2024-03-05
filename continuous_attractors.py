import torch
import math

class CAN(torch.nn.Module):
    def __init__(self,
                 length,
                 n_axes = 2):
        super().__init__()
        self.length = length
        self.n_axes = n_axes

        self.n_directions = 2 * n_axes
        assert length % self.n_directions == 0; "Length must be divisible by 2 * n_axes"
        self.angles = [2 * i * math.pi / self.n_directions for i in range(self.n_directions)]

        self.directions = torch.tensor([[math.cos(angle),
                                         math.sin(angle)] for angle in self.angles],
                                       dtype = torch.float32)