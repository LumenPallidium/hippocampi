import torch
import math

class CAN(torch.nn.Module):
    """
    A continuous attractor network as described in the paper
    "Accurate Path Integration in Continuous Attractor Network Models of Grid Cells".
    This models the grid cells of the Hippocampul-entorhinal system, inclduing
    their velocity response.
    

    Parameters
    ----------
    length : int
        The length of the network, which is a square.
    periodic : bool
        Whether to use periodic boundary conditions.
    delta_r : float
        Scaling of the envelope, higher values make the envelope more narrow.
    warmup_steps : int
        Number of steps to warm up the network.
    n_axes : int
        Number of axes in the network (default 2 = 4 head directions)
    a : float
        Parameter for the center surround function, >1 means net excitation.
    lambda_net : float
        Effective diameter of network.
    l : float
        Shift amount of grid (tunes velocity response)
    alpha : float
        Gain due to velocity, alpha * v should be less than 1
    envelope_scale : float
        Scaling of the envelope, higher values make the envelope more narrow.
    tau : float
        Time constant of the network.
    activation : torch.nn.Module
        Activation function to use.
    """
    def __init__(self,
                 length,
                 periodic = True,
                 delta_r = None,
                 warmup_steps = 1000,
                 n_axes = 2,
                 a = 1,
                 lambda_net = 13,
                 l = 2,
                 alpha = 0.10315,
                 envelope_scale = 4,
                 tau = 0.001,
                 activation = torch.nn.ReLU()):
        super().__init__()
        # parameters from paper, see methods
        self.a = a
        self.lambda_net = lambda_net
        self.beta = 3 / (lambda_net ** 2)
        self.gamma = 1.2 * self.beta
        self.l = l
        self.periodic = periodic
        self.warmup = 0
        self.warmup_steps = warmup_steps
        self.alpha = alpha
        self.envelope_scale = envelope_scale
        # time constant in ms
        self.tau = tau

        if delta_r is None:
            delta_r = length
        self.delta_r = delta_r

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

        self.weights, self.neuron_grid = self._generate_weights()
        self.state = torch.randn(self.length * self.length) / (self.length ** 2)
        self.activation = activation
        self.envelope = self._get_envelope()

    def _get_envelope(self):
        """
        Envelope function from paper
        """
        grid_mag = torch.linalg.vector_norm(self.neuron_grid,
                                            ord = 2,
                                            dim = -1)
        length_ratio = (grid_mag - self.length + self.delta_r)
        # set to 1 if negative
        length_ratio = torch.maximum(length_ratio, torch.zeros_like(length_ratio))
        length_ratio /= self.delta_r
        envelope = torch.exp(-self.envelope_scale * (length_ratio ** 2))
        return envelope
    
    def center_surround(self, x):
        """
        Center surround function from paper
        """
        out = self.a * torch.exp(-self.gamma * x) 
        out -= torch.exp(-self.beta * x)
        return out
    
    # TODO : look into convolutions for efficiency
    def _generate_weights(self):
        """
        Generate the weights using distances and center surround function.
        """
        half_length = self.length // 2
        neuron_grid = torch.stack(torch.meshgrid(torch.arange(-half_length, half_length),
                                                 torch.arange(-half_length, half_length)),
                                  dim = -1)
        neuron_grid = neuron_grid.reshape(-1, 2).float() # Nx2
        neuron_grid_vector = neuron_grid.clone()
        shifted_grid = neuron_grid + (self.l * self.directions)
        distances = torch.cdist(neuron_grid,
                                shifted_grid,
                                p = 1)
        if self.periodic and self.warmup > 0:
            # distances with periodic boundary, thanks Claude
            distances = torch.remainder(distances + half_length,
                                        2 * half_length) - half_length

        weights = self.center_surround(distances**2)
        return weights, neuron_grid_vector
    
    def forward(self, velocity, step_size = 0.5):
        """
        Take one step in the ODE. Input is a 2D velocity vector.
        """
        b = torch.einsum("ij,j->i",
                         self.directions,
                         velocity)
        b = 1 + self.alpha * b
        if (not self.periodic) or (self.warmup < self.warmup_steps):
            self.warmup += 1
            b *= self.envelope
        elif (self.warmup == self.warmup_steps):
            self.warmup += 1
            # regenerate weights after warmup
            self._generate_weights()
        state = torch.einsum("ij,j->i",
                             self.weights,
                             self.state)
        state = self.activation(state + b)
        state_step = step_size * (state - self.state.clone()) / self.tau
        new_state = self.state.clone() + state_step.clone()
        self.state = new_state
        return new_state
    
def positions_to_images(positions,
                        energies = None,
                        out_size = None,
                        length = None,
                        diameter = 1):
    """
    Convert a list of positions to images
    """
    if length is None:
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
#TODO : velocity no longer shifting state
if __name__ == "__main__":
    import tqdm
    import matplotlib.pyplot as plt
    import torchvision

    network_width = 64
    warmup = 0
    n_steps = 100000 + warmup
    fps = 30
    step_size = 0.0005 # ms
    time_constant = 0.010
    lambda_net = 13
    n_sec = int(n_steps * step_size)
    burn_in = int(1 / step_size) + warmup

    box_length = 2

    save_rate = (n_steps - burn_in) // (n_sec * fps)
    
    with torch.no_grad():
        can = CAN(network_width,
                  warmup_steps = warmup,
                  tau = time_constant,
                  lambda_net = lambda_net,
                  envelope_scale= 4,
                  periodic = False)
        x = torch.tensor([box_length / 2,
                          box_length / 2],
                         dtype = torch.float32)
        velocity = torch.tensor([0, 0], dtype = torch.float32)
        ewma_velocity = velocity.clone()
        frames = []
        frames_x = []
        for i in tqdm.trange(n_steps):
            state = can(velocity, step_size = step_size).clone()
            x += velocity * step_size

            # check if x is outside box
            above_box = (x.abs() >= box_length)
            below_box = (x <= 0)
            if above_box.any():
                # slow and bounce off walls
                velocity[above_box] *= -0.7
                # reset to box
                x[above_box] = box_length
            if below_box.any():
                velocity[below_box] *= -0.7
                x[below_box] = 0

            if (i >= burn_in) & (i % save_rate == 0):
                frames.append(state.reshape(network_width, network_width))
                frames_x.append(x.clone())
            # add small jitter, with reinforcement to existing velocity
            ewma_velocity = 0.9 * torch.randn(2) + 0.1 * ewma_velocity.clone()
            velocity += ewma_velocity * step_size
            # make sure absolute velocity is not higher than 1
            velocity = torch.minimum(velocity, torch.ones(2))
            velocity = torch.maximum(velocity, -torch.ones(2))

    frames = torch.stack(frames)
    frames_x = torch.stack(frames_x)
    #energies = torch.norm(frames, dim = (-1, -2))
    # select single neuron
    energies = frames[:, 0, 0]
    energies = (energies - energies.min()) / (energies.max() - energies.min())

    frames_x = positions_to_images(frames_x,
                                   energies,
                                   length = box_length,
                                   out_size = network_width)

    
    # convert to uint8 for video
    frames = (frames - frames.min()) / (frames.max() - frames.min())
    frames = (frames * 255).to(torch.uint8).unsqueeze(-1).repeat(1, 1, 1, 3)

    # append x position to frames
    frames = torch.cat([frames,
                        frames_x],
                        dim = -2)
    torchvision.io.write_video("can.mp4", frames, fps = fps)

    # save heatmap of weights with imshow
    plt.imshow(can.weights.numpy())
    plt.colorbar()
    plt.savefig("weights.png")



