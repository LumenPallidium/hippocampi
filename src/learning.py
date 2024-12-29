import torch

def hebbian_pseudoinverse(weight, inhibition_matrix, x, y):
    """
    A Hebbian rule for learning a matrix pseudoinverse.

    From:

    https://arxiv.org/pdf/1207.3368
    """
    with torch.no_grad():
        y_hat = torch.einsum("ij, j->i", weight, x)

        inhibition = torch.einsum("ij, j->i", inhibition_matrix, x)
        b = inhibition / (1 + torch.einsum("i, i->", x, inhibition))

        dW = torch.einsum("i,j->ij", y - y_hat, b)
        dI = torch.einsum("i,j->ij", inhibition, b)

        weight += dW
        inhibition_matrix += dI
    
    return weight, inhibition_matrix

def hebbian_pca(x, y, W):
    """
    Function that computes Hebbian weight update in the PCA formulation. 
    Note y can have been passed through a nonlinearity, so long as it is
    an odd function.

    This function does use the FastHebb formulation.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor of shape (batch_size, input_dim)
    y : torch.Tensor
        Output tensor of shape (batch_size, output_dim)
    W : torch.Tensor
        Weight tensor of shape (input_dim, output_dim)
    """
    batch_size = x.shape[0]
    pre_post_correlation = torch.einsum("...i,...j->...ij", y, x)
    post_product = torch.einsum("...i,...j->...ij", y, y).tril()

    weight_expectation = torch.einsum("...op,...oi->...pi", post_product, W)

    weight_update = (pre_post_correlation - weight_expectation) / batch_size
    return weight_update

if __name__ == "__main__":
    weight = torch.randn(3, 3)
    inhibition_matrix = torch.eye(3) / 3
    x = torch.randn(3)
    y = torch.randn(3)

    weight, inhibition_matrix = hebbian_pseudoinverse(weight, inhibition_matrix, x, y)
    print(weight)
    print(inhibition_matrix)

    x = torch.randn(3)
    y = torch.randn(3)
    W = torch.randn(3, 3)
    W = hebbian_pca(x, y, W)
    print(W)