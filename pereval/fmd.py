"""Some code is borrowed from: https://github.com/microsoft/fadtk"""

import numpy as np
import scipy
from numpy.typing import NDArray


class FrechetMusicDistance:
    def compute_fmd(self,
                    ref_emb: NDArray,
                    test_emb: NDArray) -> float:
        """Computes Frechet Music Distance.

        Args:
            ref_emb: Reference embeddings of shape (N, D).
            test_emb: Test embeddings of shape (M, D).
        
        Returns:
            Frechet Music Distance.
        """
        mu_ref = np.mean(ref_emb, axis=0)
        sigma_ref = np.cov(ref_emb, rowvar=False)

        mu_test = np.mean(test_emb, axis=0)
        sigma_test = np.cov(test_emb, rowvar=False)

        diff = mu_test - mu_ref

        eps = 1e-6

        # Product might be almost singular
        covmean, _ = scipy.linalg.sqrtm(sigma_test.dot(sigma_ref), disp=False)
        if not np.isfinite(covmean).all():
            msg = f"FMD calculation produces singular product; adding {eps} to diagonal of cov estimates"
            print(msg)
            offset = np.eye(sigma_test.shape[0]) * eps
            covmean = scipy.linalg.sqrtm((sigma_test + offset).dot(sigma_ref + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                msg = f"Imaginary component {m}"
                raise ValueError(msg)
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        fmd = (diff.dot(diff) + np.trace(sigma_test) + np.trace(sigma_ref) - 2 * tr_covmean).item()
        return fmd
