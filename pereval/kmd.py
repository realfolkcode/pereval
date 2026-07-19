"""
Code is adapted from: https://github.com/YoonjinXD/kadtk
"""

import torch
import numpy as np
from typing import List, Dict


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
    
    def calculate_bandwidth(self,
                            ref_emb: torch.Tensor,
                            precision=torch.float32) -> None:
        """Calculates the kernel bandwidth.

        Args:
            ref_emb: The set of reference embeddings.
            precision: Type setting for matrix calculation precision.
        """
        ref_emb = ref_emb.to(dtype=precision)
        self.bandwidth = median_pairwise_distance(ref_emb)
    
    def compute_kmd(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        bandwidth=None,
        precision=torch.float32,
        eps=1e-8,
        return_components=False,
    ) -> torch.Tensor:
        """
        Compute the Kernel Music Distance (KMD) between two samples using PyTorch.

        Args:
            x: The set of model embeddings of shape (m, embedding_dim).
            y: The set of reference embeddings of shape (n, embedding_dim).
            bandwidth: The bandwidth value for the Gaussian RBF kernel.
            precision: Type setting for matrix calculation precision.
            eps: Small value to prevent division by zero.
            return_components: If True, returns mean components.

        Returns:
            The KMD between x and y embedding sets.
        """
        # Ensure x and y are of the correct precision
        x = x.to(dtype=precision)
        y = y.to(dtype=precision)

        if bandwidth is None:
            bandwidth = self.bandwidth

        m, n = x.shape[0], y.shape[0]
        
        gamma = 1 / (2 * bandwidth**2 + eps)
        kernel = lambda a: torch.exp(-gamma * a)
        
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
            scale_factor: Scale factor of KMD.
        """
        self.scale_factor = scale_factor
    
    def calculate_bandwidth(self,
                            ref_emb: torch.Tensor,
                            label_lst: List[str],
                            precision=torch.float32) -> None:
        """Calculates the kernel bandwidth.

        Args:
            ref_emb: The set of reference embeddings.
            label_lst: The list of labels.
            precision: Type setting for matrix calculation precision.
        """
        ref_emb = ref_emb.to(dtype=precision)

        label_lst = np.array(label_lst)
        n = len(label_lst)

        dist_lst = []
        for i in range(n):
            for j in range(i+1, n):
                if label_lst[i] == label_lst[j]:
                    dist = torch.linalg.vector_norm(ref_emb[i] - ref_emb[j])
                    dist_lst.append(dist.item())
        dist_lst = torch.tensor(dist_lst)
        self.bandwidth = torch.median(dist_lst).item()
    
    def compute_kmd(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        kernel = None,
        eps=1e-8
    ):
        if kernel is None:
            gamma = 1 / (2 * self.bandwidth**2 + eps)
            kernel = lambda a: torch.exp(-gamma * a)
        
        m, n = x.shape[0], y.shape[0]
        
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

        return k_xx_mean, k_yy_mean, -2 * k_xy_mean
    
    def compute_kpd(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        x_label_lst: List[str],
        y_label_lst: List[str],
        bandwidth=None,
        precision=torch.float32,
        eps=1e-8,
        return_components=False):
        """
        Compute the Kernel Performance Distance (KPD) between two samples using PyTorch.

        Args:
            x: The set of model embeddings of shape (m, embedding_dim).
            y: The set of reference embeddings of shape (n, embedding_dim).
            x_label_lst: The list of labels for x.
            y_label_lst: The list of labels for y. 
            bandwidth: The bandwidth value for the Gaussian RBF kernel.
            precision: Type setting for matrix calculation precision.
            eps: Small value to prevent division by zero.
            return_components: If True, returns mean components.

        Returns:
            The KPD between x and y embedding sets.
        """
        x = x.to(dtype=precision)
        y = y.to(dtype=precision)

        if bandwidth is None:
            bandwidth = self.bandwidth
        
        gamma = 1 / (2 * bandwidth**2 + eps)
        kernel = lambda a: torch.exp(-gamma * a)

        unique_labels = set(x_label_lst)
        x_label_lst = np.array(x_label_lst)
        y_label_lst = np.array(y_label_lst)

        k_xx_lst = []
        k_yy_lst = []
        k_xy_lst = []

        for label in unique_labels:
            if (x_label_lst == label).sum() < 2:
                continue
            k_xx, k_yy, k_xy = self.compute_kmd(x[x_label_lst == label],
                                                y[y_label_lst == label], 
                                                kernel)
            k_xx_lst.append(k_xx.item())
            k_yy_lst.append(k_yy.item())
            k_xy_lst.append(k_xy.item())

        k_xx = torch.tensor(k_xx_lst).mean().item()
        k_yy = torch.tensor(k_yy_lst).mean().item()
        k_xy = torch.tensor(k_xy_lst).mean().item()

        if return_components:
            return k_xx, k_yy, k_xy

        return k_xx + k_yy + k_xy
