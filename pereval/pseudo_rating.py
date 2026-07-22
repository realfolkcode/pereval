import numpy as np
from numpy.typing import NDArray
from typing import List
from sklearn.covariance import LedoitWolf
from tqdm import tqdm


def estimate_score_precision(cov_emb: NDArray,
                             score_lst: List[str]) -> NDArray:
    """Estimates the score-conditional precision matrix (the inverse
    of the covariance).
    
    Args:
        cov_emb: The embeddings of shape (N, D).
        score_lst: The names of the pieces for each embedding.
    
    Returns:
        Score-conditional precision matrix.
    """
    D = cov_emb.shape[-1]
    cov = np.zeros((D, D))

    n = 0
    for score_name in tqdm(set(score_lst)):
        human_emb = cov_emb[np.array(score_lst) == score_name]
        if len(human_emb) < 2:
            continue

        class_cov = LedoitWolf().fit(human_emb)

        cov += class_cov.covariance_ * len(human_emb)
        n += len(human_emb)

    cov /= n
    inv_cov = np.linalg.inv(cov)
    return inv_cov


class MahalanobisRating:
    """Pseudo-rating based on Mahalanobis distance."""

    def fit_precision(self,
                      cov_emb: NDArray,
                      score_lst: List[str]):
        """Fits score-conditional precision matrix.

        Args:
            cov_emb: The embeddings of shape (N, D).
            score_lst: The names of the pieces for each embedding.
        """
        self.inv_cov = estimate_score_precision(cov_emb=cov_emb,
                                                score_lst=score_lst)

    def predict(self,
                emb: NDArray,
                ref_emb: NDArray) -> NDArray:
        """Predicts a pseudo-rating.
        
        Args:
            emb: The embeddings of shape (N, D) for which the ratings should be predicted.
            ref_emb: The reference embeddings.
        
        Returns:
            An array of ratings of shape (N,).
        """
        diff = emb - ref_emb.mean(0)
        if len(diff.shape) < 2:
            diff = diff.unsqueeze(0)
        
        rating = diff @ self.inv_cov @ diff.T
        rating = np.diagonal(rating)

        return rating


class MarginalMahalanobisRating:
    """Pseudo-rating based on Marginal Mahalanobis distance."""

    def fit_precision(self,
                      cov_emb: NDArray):
        """Fits marginal precision matrix.

        Args:
            cov_emb: The embeddings of shape (N, D).
        """
        self.marginal_cov = LedoitWolf().fit(cov_emb)

    def predict(self,
                emb: NDArray) -> NDArray:
        """Predicts a pseudo-rating.
        
        Args:
            emb: The embeddings of shape (N, D) for which the ratings should be predicted.
        
        Returns:
            An array of ratings of shape (N,).
        """
        rating = self.marginal_cov.mahalanobis(emb)
        return rating
