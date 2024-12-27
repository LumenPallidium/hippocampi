import torch
from tqdm import tqdm

class NestedGrid(torch.nn.Module):
    def __init__(self,
                 sizes = [3, 5, 7]):
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
        for level in range(self.n):
            self.indices[0, level] += x_shift
            self.indices[1, level] += y_shift
            
            x_shift = self.indices[0, level] // self.sizes[level]
            y_shift = self.indices[1, level] // self.sizes[level]

            self.indices[0, level] %= self.sizes[level]
            self.indices[1, level] %= self.sizes[level]

    def get_full_grids(self, flatten = True):
        grids = []
        for level in range(self.n):
            blank = torch.zeros(self.sizes[level], self.sizes[level])
            blank[self.indices[1, level], self.indices[0, level]] = 1
            grids.append(blank)
        if flatten:
            return torch.cat([grid.flatten() for grid in grids])
        return grids
    
    def unflatten(self, x, run_wta = True):
        grids = x.split(self.grid_sizes.tolist(), dim = -1)
        if run_wta:
            indices = torch.stack([grid.flatten().argmax() for grid in grids])

            onehot_vec = torch.zeros(self.dim)
            onehot_vec[indices] = 1

            return onehot_vec, indices
        return grids
    

class VectorHaSH(torch.nn.Module):
    def __init__(self,
                 sensory_dim,
                 hippocampal_dim,
                 sizes = [3, 5, 7],
                 hg_sparsity = 0.6,
                 theta = 0.5,
                 S = None):
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
            g_vec = [torch.functional.F.one_hot(torch.tensor(j), num_classes = k) for j, k in zip(indices, self.grid.grid_sizes)]
            g_vec = torch.cat(g_vec).to(torch.float32)
            h_vec = torch.nn.functional.relu(self.W_hg @ g_vec - self.theta)
            W_gh += g_vec[:, None] @ h_vec[None, :] / self.h_dim

            if S is not None:
                H.append(h_vec)

        self.register_buffer("W_gh", W_gh)

        if S is not None:
            H = torch.stack(H)
            W_hs = H.T @ torch.pinverse(S)
            W_sh = S @ torch.pinverse(H).T
        else:
            W_hs = torch.randn(self.s_dim, self.h_dim)
            W_sh = torch.randn(self.h_dim, self.s_dim)
        
        self.register_buffer("W_hs", W_hs)
        self.register_buffer("W_sh", W_sh)

    def forward(self, sense):
        self.s = sense
        self.h = torch.nn.functional.relu(self.W_hs @ self.s)
        onehot, indices = self.grid.unflatten(self.W_gh @ self.h)
        self.grid.indices = indices

        return onehot, indices

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    from time import sleep
    sizes = [3, 5]
    n_pats = np.prod(sizes)**2
    n_test = 1000

    S = torch.randn(128, n_pats) * 5

    vhash = VectorHaSH(128, 32, sizes = sizes, S = S)

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
    print(f"Done! Final success rate: {success / n_test:.2f}")
