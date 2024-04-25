import torch
import math

class CAN(torch.nn.Module):
    def __init__(self,
                 length,
                 n_axes = 2,
                 a = 1,
                 lambda_net = 13,
                 l = 0.10315,
                 alpha = 0.10315,
                 activation = torch.nn.ReLU()):
        super().__init__()
        # parameters from paper, see methods
        self.a = a
        self.lambda_net = lambda_net
        self.beta = 3 / (lambda_net ** 2)
        self.gamma = 1.05 * self.beta
        self.l = l
        self.alpha = alpha

        self.n_axes = n_axes
        # number of unique angles for a head direction cell
        self.n_directions = n_axes ** 2

        assert length % self.n_axes == 0; "Length must be divisible by n_axes"
        self.length = length
        self.angles = [2 * i * math.pi / self.n_directions for i in range(self.n_directions)]

        directions = torch.tensor([[math.cos(angle),
                                         math.sin(angle)] for angle in self.angles],
                                       dtype = torch.float32)
        # rounding to avoid floating point errors
        directions = torch.round(directions * 1e4) / 1e4
        n_repeats = (length ** 2) // (self.n_axes ** 2)
        self.directions = directions.repeat(n_repeats, 1)

        self.weights = self._generate_weights()
        self.state = torch.zeros(self.length * self.length)
        self.activation = activation
    
    def center_surround(self, x):
        """
        Center surround function from paper
        """
        out = self.a * torch.exp(-self.gamma * (x ** 2)) 
        out -= torch.exp(-self.beta* (x ** 2))
        return out
    
    # TODO : add head direction cells
    # TODO : look into convolutions for efficiency
    # TODO : euler integration of NDE
    # TODO : periodic boundary conditions
    def _generate_weights(self):

        half_length = self.length // 2
        neuron_grid = torch.stack(torch.meshgrid(torch.arange(-half_length, half_length),
                                                 torch.arange(-half_length, half_length)),
                                  dim = -1)
        neuron_grid = neuron_grid.reshape(-1, 2)
        # shift to preferred direction
        neuron_grid = neuron_grid - self.l * self.directions
        distances = torch.einsum("ik,jk->ij",
                                 neuron_grid,
                                 -neuron_grid)
        weights = self.center_surround(distances)
        return weights
    
    def forward_step(self, velocity):
        b = torch.einsum("ij,j->i",
                         self.directions,
                         velocity)
        state = self.weights @ self.state
        state = self.activation(state + self.alpha * b)
        return state
    
if __name__ == "__main__":
    network_width = 64
    can = CAN(network_width)
    velocity = torch.tensor([1, 0], dtype = torch.float32)
    print(can.weights.shape)
    print(can.directions.shape)
    print(can.forward_step(velocity).shape)