import torch

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

    def shift(self, x_shift = 0, y_shift = 0):
        for level in range(self.n):
            self.indices[0, level] += x_shift
            self.indices[1, level] += y_shift
            
            x_shift = self.indices[0, level] // self.sizes[level]
            y_shift = self.indices[1, level] // self.sizes[level]

            self.indices[0, level] %= self.sizes[level]
            self.indices[1, level] %= self.sizes[level]

    def get_full_grids(self):
        grids = []
        for level in range(self.n):
            blank = torch.zeros(self.sizes[level], self.sizes[level])
            blank[self.indices[1, level], self.indices[0, level]] = 1
            grids.append(blank)
        return grids


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    grid = NestedGrid()
    full_grids = grid.get_full_grids()

    grid.shift(1, 1)
    full_grids += grid.get_full_grids()

    grid.shift(1, 0)
    full_grids += grid.get_full_grids()

    grid.shift(14,0)
    full_grids += grid.get_full_grids()

    fig, axs = plt.subplots(4, 3)
    for i, ax in enumerate(axs.flatten()):
        ax.imshow(full_grids[i])
        ax.axis("off")
    plt.show()
