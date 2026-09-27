import torch

def calculate_covariance_matrix(vectors) -> torch.Tensor:
    """
    Calculate the covariance matrix for given feature vectors using PyTorch.
    Input: 2D array-like of shape (n_features, n_observations).
    Returns a tensor of shape (n_features, n_features).
    """
    v_t = torch.as_tensor(vectors, dtype=torch.float)
    # Your implementation here
    n_observations = v_t.shape[1]
    mean_v_t = v_t.mean(dim=1, keepdim = True)
    v_t_ = v_t - mean_v_t
    covar_matrix = v_t_ @ v_t_.T / (n_observations - 1)
    return covar_matrix    
