import torch
import math

class CAN(torch.nn.Module):
    def __init__(self,
                 length,
                 n_axes = 2,
                 a = 1,
                 lambda_net = 13,
                 l = 2,
                 alpha = 0.10315,
                 tau = 10,
                 activation = torch.nn.ReLU()):
        super().__init__()
        # parameters from paper, see methods
        self.a = a
        self.lambda_net = lambda_net
        self.beta = 3 / (lambda_net ** 2)
        self.gamma = 1.05 * self.beta
        self.l = l
        self.alpha = alpha
        # time constant in ms
        self.tau = tau

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
    
    # TODO : look into convolutions for efficiency
    def _generate_weights(self):

        half_length = self.length // 2
        neuron_grid = torch.stack(torch.meshgrid(torch.arange(-half_length, half_length),
                                                 torch.arange(-half_length, half_length)),
                                  dim = -1)
        neuron_grid = neuron_grid.reshape(-1, 2).float() # Nx2
        # Nx1x2 - 1xNx2 = NxNx2
        distances = neuron_grid.unsqueeze(1) - neuron_grid.unsqueeze(0)
        # distances with periodic boundary, thanks Claude
        distances = torch.remainder(distances + half_length, 2 * half_length) - half_length
        # NxNx2 -> NxN, 1 norm as in paper
        distances = torch.norm(distances, p = 1, dim = -1)
        # shift to preferred direction
        neuron_grid = neuron_grid - self.l * self.directions

        weights = self.center_surround(distances)
        return weights
    
    def forward(self, velocity, step_size = 0.5):
        """
        Take one step in the ODE. Input is a 2D velocity vector.
        """
        b = torch.einsum("ij,j->i",
                         self.directions,
                         velocity)
        state = self.weights @ self.state.clone()
        state = self.activation(state + 1 + self.alpha * b)
        state_step = step_size * (state - self.state) / self.tau
        new_state = self.state.clone() + state_step.clone()
        self.state = new_state
        return new_state
    
def positions_to_images(positions,
                        energies = None,
                        out_size = None,
                        diameter = 1):
    """
    Convert a list of positions to images
    """
    left_corner = positions.min(dim = 0).values

    positions = positions - left_corner
    length = positions.max().ceil() + 1
    if out_size is not None:
        positions = positions / length
        length = out_size
    image = torch.zeros(positions.shape[0],
                        length, length, 3)
    for i, (x, y) in enumerate(positions):
        x = int(x * length)
        y = int(y * length)
        image[i,
              x:(x + diameter),
              y:(y + diameter), :] = 255
        if energies is not None:
            # 0 energy is blue, 1 is red
            energy = (energies[i] - 0.5) * 2
            image[i:, x:(x + diameter), y:(y + diameter), 0] += torch.maximum(energy,
                                                                              torch.zeros(1)) * 100
            image[i:, x:(x + diameter), y:(y + diameter), 2] += torch.maximum(-energy,
                                                                              torch.zeros(1)) * 100

    return image
    
#TODO : image not triangular - reshape issue?
if __name__ == "__main__":
    import tqdm
    import torchvision

    network_width = 32
    n_steps = 10000
    fps = 30
    n_sec = 20

    save_rate = n_steps // (n_sec * fps)
    step_size = 0.5 # ms
    noise_scale = 0.001

    with torch.no_grad():
        can = CAN(network_width)
        x = torch.tensor([0, 0], dtype = torch.float32)
        velocity = torch.tensor([0, 0], dtype = torch.float32)
        frames = []
        frames_x = []
        for i in tqdm.trange(n_steps):
            state = can(velocity, step_size = step_size).clone()
            x += velocity * step_size
            if (i != 0) & (i % save_rate == 0):
                frames.append(state.reshape(network_width, network_width))
                frames_x.append(x.clone())
            # use sin and cos so velocity is an expanding spiral
            r = math.sqrt(i) * noise_scale
            velocity = torch.tensor([r * math.cos(i * 0.01),
                                     r * math.sin(i * 0.01)],
                                     dtype = torch.float32)

    frames = torch.stack(frames)
    frames_x = torch.stack(frames_x)
    energies = torch.norm(frames, dim = (-1, -2))
    energies = (energies - energies.min()) / (energies.max() - energies.min())

    frames_x = positions_to_images(frames_x,
                                   energies,
                                   out_size = network_width)

    
    # convert to uint8 for video
    frames = (frames - frames.min()) / (frames.max() - frames.min())
    frames = (frames * 255).to(torch.uint8).unsqueeze(-1).repeat(1, 1, 1, 3)

    # append x position to frames
    frames = torch.cat([frames,
                        frames_x],
                        dim = -2)
    torchvision.io.write_video("can.mp4", frames, fps = fps)



