import torch
from tqdm import tqdm
from learning import hebbian_pseudoinverse, hebbian_pca

class NestedGrid(torch.nn.Module):
    """
    A class to represent a nested grid structure. Specifically, this class is used to represent a grid of grids,
    where each grid is a square grid of size sizes[i] x sizes[i]. The grid is represented as a vector of size sum(sizes**2), 
    where each grid is flattened and concatenated. The grid is represented as a one-hot vector, where the one-hot vector is
    the one-hot representation of the grid that is currently active. The grid can be shifted in the x and y directions,
    and the grid can be unflattened to its original form. The grid can also be converted to a one-hot vector and indices,
    and vice versa. The grid is zero-indexed, and the grid is assumed to be a square grid.
    """
    def __init__(self,
                 sizes = [3, 5, 7]):
        """
        Parameters:
        ----------
        sizes (list): A list of integers representing the sizes of the grids in the nested grid structure. They should be co-prime.
        """
        super().__init__()
        self.n = len(sizes)
        sizes = torch.tensor(sizes, dtype=torch.int64)
        indices = torch.zeros((2, self.n), dtype=torch.int64)
        grid_sizes = sizes.clone()**2

        self.register_buffer("sizes", sizes)
        self.register_buffer("indices", indices)
        self.register_buffer("grid_sizes", grid_sizes)

        self.dim = grid_sizes.sum().item()

    def shift(self, x_shift = 0, y_shift = 0):
        """
        Shift the grid in the x and y directions.
        Parameters:
        ----------
        x_shift (int): The amount to shift the grid in the x direction.
        y_shift (int): The amount to shift the grid in the y direction.
        """
        for level in range(self.n):
            self.indices[0, level] += x_shift
            self.indices[1, level] += y_shift
            
            x_shift = self.indices[0, level] // self.sizes[level]
            y_shift = self.indices[1, level] // self.sizes[level]

            self.indices[0, level] %= self.sizes[level]
            self.indices[1, level] %= self.sizes[level]

    def get_full_grids(self, flatten = True):
        """
        Get the full grids as a list of tensors or a single flat tensor.
        Parameters:
        ----------
        flatten (bool): Whether to return the full grids as a flat tensor.
        """
        grids = []
        for level in range(self.n):
            blank = torch.zeros(self.sizes[level], self.sizes[level])
            blank[self.indices[1, level], self.indices[0, level]] = 1
            grids.append(blank)
        if flatten:
            return torch.cat([grid.flatten() for grid in grids])
        return grids
    
    def unflatten(self, x, run_wta = True):
        """
        Unflatten the grid from a flat tensor.
        Parameters:
        ----------
        x (tensor): The flat tensor to unflatten.
        run_wta (bool): Whether to run a winner-take-all operation to get the indices of the grid.
        """
        grids = x.split(self.grid_sizes.tolist(), dim = -1)
        if run_wta:
            indices = torch.stack([grid.flatten().argmax() for grid in grids])
            onehot_vec = self.indices_to_onehot(indices)

            return onehot_vec, indices
        return grids
    
    def indices_to_onehot(self, indices):
        """
        Convert the indices of the grid to a one-hot vector.
        Parameters:
        ----------
        indices (tensor): The indices of the grid.
        """
        onehot_vec = torch.zeros(self.dim)
        onehot_vec[indices] = 1
        return onehot_vec
    

class VectorHaSH(torch.nn.Module):
    """
    The Vector-HaSH architecture from:
    https://www.biorxiv.org/content/10.1101/2023.11.28.568960v1
    This a computational model of the hippocampus and entorhinal cortex, where the hippocampus is represented as a
    vector of neurons, and the entorhinal cortex is represented as a grid of neurons.
    """
    def __init__(self,
                 sensory_dim,
                 hippocampal_dim,
                 sizes = [3, 5, 7],
                 hg_sparsity = 0.6,
                 theta = 0.5,
                 S = None):
        """
        Parameters:
        ----------
        sensory_dim (int): The dimensionality of the sensory input.
        hippocampal_dim (int): The dimensionality of the hippocampal output.
        sizes (list): A list of integers representing the sizes of the grids in the nested grid structure. They should be co-prime.
        hg_sparsity (float): The sparsity of the connections from the hippocampus to the grid.
        theta (float): The threshold for the hippocampal neurons.
        S (tensor): The sensory patterns to use for training the model. If None, the model will use random patterns.
        """
        super().__init__()
        self.grid = NestedGrid(sizes)
        self.g_dim = self.grid.dim
        self.s_dim = sensory_dim
        self.h_dim = hippocampal_dim
        self.hg_sparsity = hg_sparsity
        self.theta = theta

        self._initialize_weights(S = S)
        self.register_buffer("h", torch.zeros(self.h_dim))
        self.register_buffer("s", torch.zeros(self.s_dim))
        

    def _initialize_weights(self, S = None):
        W_hg = torch.randn(self.h_dim, self.g_dim)
        W_hg[torch.rand(self.h_dim, self.g_dim) > self.hg_sparsity] = 0

        self.register_buffer("W_hg", W_hg)

        W_gh = torch.zeros(self.g_dim, self.h_dim)
        grid_prod = self.grid.grid_sizes.prod().item()
        H = []
        for i in tqdm(range(grid_prod)):
            indices = [(i // self.grid.grid_sizes[:j].prod().item()) % self.grid.sizes[j] for j in range(self.grid.n)]
            g_vec = [torch.functional.F.one_hot(j, num_classes = k) for j, k in zip(indices, self.grid.grid_sizes)]
            g_vec = torch.cat(g_vec).to(torch.float32)
            h_vec = torch.nn.functional.relu(self.W_hg @ g_vec - self.theta)
            W_gh += g_vec[:, None] @ h_vec[None, :] / self.h_dim

            if S is not None:
                H.append(h_vec)

        self.register_buffer("W_gh", W_gh)

        if S is not None:
            H = torch.stack(H)

            S_inv = torch.linalg.lstsq(S, torch.eye(S.shape[0])).solution
            H_inv = torch.linalg.lstsq(H, torch.eye(H.shape[0])).solution

            W_hs = H.T @ S_inv
            W_sh = S @ H_inv.T
        else:
            W_hs = torch.randn(self.s_dim, self.h_dim)
            W_sh = torch.randn(self.h_dim, self.s_dim)
        
        self.register_buffer("W_hs", W_hs)
        self.register_buffer("W_sh", W_sh)

    def forward(self, sense):
        """
        Give a sensory input, return where that memory is "placed" in the grid.

        Parameters:
        ----------
        sense (tensor): The sensory input to the model.
        """
        self.s = sense
        self.h = torch.nn.functional.relu(self.W_hs @ self.s)
        onehot, indices = self.grid.unflatten(self.W_gh @ self.h)
        self.grid.indices = indices

        return onehot, indices
    
    def recall_sense(self, indices):
        """
        Recall a sensory input from the grid indices.

        Parameters:
        ----------
        indices (tensor): The indices of the grid to recall the sensory input from.
        """
        self.grid.indices = torch.tensor(indices)
        onehot = self.grid.indices_to_onehot(self.grid.indices)
        self.h = torch.nn.functional.relu(self.W_hg @ onehot - self.theta)
        return self.W_sh @ self.h
    
def location_recall_test(n_test, S, vhash, s_dim, n_pats):
    print("Testing...")
    sleep(0.1)
    success = 0
    pbar = tqdm(range(n_test))
    for i in range(n_test):
        rand_idx = np.random.randint(0, n_pats)
        pattern = S[:, rand_idx]
        rand_idx_grid = [(rand_idx // vhash.grid.grid_sizes[:j].prod().item()) % vhash.grid.sizes[j] for j in range(vhash.grid.n)]

        onehot, idx = vhash(pattern)

        if torch.all(idx == torch.tensor(rand_idx_grid)):
            success += 1

        pbar.set_description(f"Success rate: {success / (i + 1):.2f}")
        pbar.update(1)

    pbar.close()
    # this will be ~~ s_dim / n_pats
    print(f"Done! Final success rate: {success / n_test:.2f}")
    print(f"Expected success rate: {min(1, s_dim / n_pats):.2f}")

def pattern_recall_test(n_test, S, vhash, n_pats):
    print("Testing...")
    sleep(0.1)
    dists = []
    pbar = tqdm(range(n_test))

    exemplar = torch.randn(vhash.s_dim)

    for i in range(n_test):
        rand_idx = np.random.randint(0, n_pats)
        pattern = S[:, rand_idx]
        rand_idx_grid = [(rand_idx // vhash.grid.grid_sizes[:j].prod().item()) % vhash.grid.sizes[j] for j in range(vhash.grid.n)]

        s_hat = vhash.recall_sense(rand_idx_grid)

        dist = ((s_hat - pattern)**2).mean()
        dists.append(dist)

        exemplar_dist = ((s_hat - exemplar)**2).mean()

        pbar.set_description(f"Avg Dist: {np.mean(dists):.2f} (Exemplar Dist : {exemplar_dist:.2f})")
        pbar.update(1)

    pbar.close()

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    from time import sleep
    sizes = [3, 5]
    n_pats = np.prod(sizes)**2
    n_test = 100
    s_dim = 256
    h_dim = 64
    with torch.no_grad():
        S = torch.randn(s_dim, n_pats)

        vhash = VectorHaSH(s_dim, h_dim, sizes = sizes, S = S)

        location_recall_test(n_test, S, vhash, s_dim, n_pats)
        pattern_recall_test(n_test, S, vhash, n_pats)


