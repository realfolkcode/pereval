"""Some code is borrowed from: https://github.com/YoonjinXD/kadtk"""

import torch
from typing import List


def median_pairwise_distance(x, subsample=None):
    """
    Compute the median pairwise distance of an embedding set.
    
    Args:
    x: torch.Tensor of shape (n_samples, embedding_dim)
    subsample: int, number of random pairs to consider (optional)
    
    Returns:
    The median pairwise distance between points in x.
    """
    x = torch.tensor(x, dtype=torch.float32)
    n_samples = x.shape[0]
    
    if subsample is not None and subsample < n_samples * (n_samples - 1) / 2:
        # Randomly select pairs of indices
        idx1 = torch.randint(0, n_samples, (subsample,))
        idx2 = torch.randint(0, n_samples, (subsample,))
        
        # Ensure idx1 != idx2
        mask = idx1 == idx2
        idx2[mask] = (idx2[mask] + 1) % n_samples
        
        # Compute distances for selected pairs
        distances = torch.sqrt(torch.sum((x[idx1] - x[idx2])**2, dim=1))
    else:
        # Compute all pairwise distances
        distances = torch.pdist(x)
        
    return torch.median(distances).item()


class KernelMusicDistance:
    def __init__(self,
                 scale_factor: float = 100):
        """Initializes an instance of KernelMusicDistance.
        
        Args:
            scale_factor: Scale factor of KMD.
        """
        self.scale_factor = scale_factor
    
    def compute_kmd(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        bandwidth=None,
        kernel='gaussian',
        precision=torch.float32,
        eps=1e-8,
        return_components=False,
    ) -> torch.Tensor:
        """
        Compute the Kernel Music Distance (KAD) between two samples using PyTorch.

        Args:
            x: The set of model embeddings of shape (m, embedding_dim).
            y: The set of reference embeddings of shape (n, embedding_dim).
            bandwidth: The bandwidth value for the Gaussian RBF kernel.
            kernel: Kernel function to use ('gaussian', 'iq', 'imq').
            precision: Type setting for matrix calculation precision.
            eps: Small value to prevent division by zero.
            return_components: If True, returns mean components.

        Returns:
            The KAD between x and y embedding sets.
        """
        # Ensure x and y are of the correct precision
        x = x.to(dtype=precision)
        y = y.to(dtype=precision)

        # Use median distance heuristic if bandwidth not provided
        if bandwidth is None:
            bandwidth = median_pairwise_distance(y)

        m, n = x.shape[0], y.shape[0]
        
        # Define kernel functions
        gamma = 1 / (2 * bandwidth**2 + eps)
        if kernel == 'gaussian':    # Gaussian Kernel
            kernel = lambda a: torch.exp(-gamma * a)
        elif kernel == 'iq':        # Inverse Quadratic Kernel
            kernel = lambda a: 1 / (1 + gamma * a)
        elif kernel == 'imq':       # Inverse Multiquadric Kernel
            kernel = lambda a: 1 / torch.sqrt(1 + gamma * a)
        else:
            raise ValueError("Invalid kernel type. Valid kernels: 'gaussian', 'iq', 'imq'")
        
        xx = x @ x.T
        x_sqnorms = torch.diagonal(xx)
        d2_xx = x_sqnorms.unsqueeze(1) + x_sqnorms.unsqueeze(0) - 2 * xx # shape (m, m)
                
        k_xx = kernel(d2_xx)
        k_xx = k_xx - torch.diag(torch.diagonal(k_xx))
        k_xx_mean = k_xx.sum() / (m * (m - 1))

        yy = y @ y.T
        y_sqnorms = torch.diagonal(yy)
        d2_yy = y_sqnorms.unsqueeze(1) + y_sqnorms.unsqueeze(0) - 2 * yy # shape (n, n)

        k_yy = kernel(d2_yy)
        k_yy = k_yy - torch.diag(torch.diagonal(k_yy))
        k_yy_mean = k_yy.sum() / (n * (n - 1))
        
        # Compute kernel statistics for xy
        xy = x @ y.T
        d2_xy = x_sqnorms.unsqueeze(1) + y_sqnorms.unsqueeze(0) - 2 * xy # shape (m, n)
        k_xy = kernel(d2_xy)
        k_xy_mean = k_xy.mean()

        k_xx_mean *= self.scale_factor
        k_yy_mean *= self.scale_factor
        k_xy_mean *= self.scale_factor
        
        # Compute MMD
        if return_components:
            return k_xx_mean, k_yy_mean, -2 * k_xy_mean
        else:
            result = k_xx_mean + k_yy_mean - 2 * k_xy_mean
            return result


class KernelPerformanceDistance:
    def __init__(self,
                 scale_factor: float = 100):
        """Initializes an instance of KernelPerformanceDistance.
        
        Args:
            scale_factor: Scale factor of KPD.
        """
        self.scale_factor = scale_factor
    
    def compute_kpd(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        label_lst: List,
        bandwidth=None,
        kernel='gaussian',
        precision=torch.float32,
        eps=1e-8
    ) -> torch.Tensor:
        """
        Compute the Kernel Performance Distance (KPD) between two samples using PyTorch.

        Args:
            x: The first set of embeddings of shape (m, embedding_dim).
            y: The second set of embeddings of shape (n, embedding_dim).
            label_lst: The list of labels.
            bandwidth: The bandwidth value for the Gaussian RBF kernel.
            kernel: Kernel function to use ('gaussian', 'iq', 'imq').
            precision: Type setting for matrix calculation precision.
            eps: Small value to prevent division by zero.

        Returns:
            The KPD between x and y embedding sets.
        """
        # Use median distance heuristic if bandwidth not provided
        if bandwidth is None:
            bandwidth = median_pairwise_distance(y)
        
        # Define kernel functions
        gamma = 1 / (2 * bandwidth**2 + eps)
        if kernel == 'gaussian':    # Gaussian Kernel
            kernel = lambda a: torch.exp(-gamma * a)
        elif kernel == 'iq':        # Inverse Quadratic Kernel
            kernel = lambda a: 1 / (1 + gamma * a)
        elif kernel == 'imq':       # Inverse Multiquadric Kernel
            kernel = lambda a: 1 / torch.sqrt(1 + gamma * a)
        else:
            raise ValueError("Invalid kernel type. Valid kernels: 'gaussian', 'iq', 'imq'")
        
        n = len(x)

        xx = x @ x.T
        x_sqnorms = torch.diagonal(xx)
        d2_xx = x_sqnorms.unsqueeze(1) + x_sqnorms.unsqueeze(0) - 2 * xx # shape (n, n)
                
        k_xx = kernel(d2_xx)
        k_xx = k_xx - torch.diag(torch.diagonal(k_xx))

        yy = y @ y.T
        y_sqnorms = torch.diagonal(yy)
        d2_yy = y_sqnorms.unsqueeze(1) + y_sqnorms.unsqueeze(0) - 2 * yy # shape (n, n)

        k_yy = kernel(d2_yy)
        k_yy = k_yy - torch.diag(torch.diagonal(k_yy))
        
        # Compute kernel statistics for xy
        xy = x @ y.T
        d2_xy = x_sqnorms.unsqueeze(1) + y_sqnorms.unsqueeze(0) - 2 * xy # shape (n, n)
        k_xy = kernel(d2_xy)
        k_xy = k_xy - torch.diag(torch.diagonal(k_xy))

        h = k_xx + k_yy - 2 * k_xy

        k_label = torch.zeros_like(h)
        for i in range(n):
            for j in range(n):
                k_label[i, j] = int(label_lst[i] == label_lst[j])

        h = h * k_label
        h_mean = h.sum() / (n * (n - 1))

        return h_mean
    
    def _compute_h(self,
                   x1,
                   x2,
                   y1,
                   y2,
                   kernel):
        return kernel(((x1 - x2)**2).sum()) + kernel(((y1 - y2)**2).sum()) \
             - kernel(((x1 - y2)**2).sum()) - kernel(((x2 - y1)**2).sum())
