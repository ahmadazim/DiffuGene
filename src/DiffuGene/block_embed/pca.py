import numpy as np
import torch
from sklearn.decomposition import TruncatedSVD

class PCA_Block:
    """
    Learn a k-dimensional SNP-space PCA for one LD block.
    Handles cases where n_snps < k by using actual_k = min(k, n_snps).
    Methods:
      - fit(X): center & fit PCA on X → stores means, components_
      - encode(X): center X, project to PC scores
      - decode(scores): reconstruct X from scores + add means
    """
    def __init__(self, k: int):
        self.k = k                  # Target number of components
        self.actual_k = None        # Actual number of components fitted (≤ k)
        self.means = None           # shape (n_snps,)
        self.components_ = None     # shape (actual_k, n_snps) (TruncatedSVD API)
        self.svd = None             # Will be initialized in fit() with actual_k

    def fit(self, X: np.ndarray):
        """
        Fit PCA on genotype matrix X (n_samples × n_snps).
        Centers X internally.
        Handles cases where n_snps < k by using actual_k = min(k, n_snps).
        """
        n_samples, n_snps = X.shape
        
        # Determine actual number of components to use
        self.actual_k = min(self.k, n_snps)
        
        # Initialize SVD with actual number of components
        self.svd = TruncatedSVD(n_components=self.actual_k, algorithm="randomized")
        
        # compute & store means
        self.means = X.mean(axis=0)
        Xc = X - self.means
        
        # fit SVD
        scores = self.svd.fit_transform(Xc)   # (n_samples, actual_k)
        self.components_ = self.svd.components_  # (actual_k, n_snps)
        
        # If actual_k < k, pad the scores with zeros for consistency
        if self.actual_k < self.k:
            padded_scores = np.zeros((n_samples, self.k))
            padded_scores[:, :self.actual_k] = scores
            scores = padded_scores
        
        return scores

    def encode(self, X: np.ndarray) -> np.ndarray:
        """
        Project X into the learned k-dimensional PC space.
        Returns scores: shape (n_samples, k) - padded with zeros if actual_k < k
        """
        if self.means is None or self.components_ is None or self.actual_k is None:
            raise RuntimeError("You must call fit() before encode().")
        
        Xc = X - self.means
        scores = Xc @ self.components_.T  # (n_samples, actual_k)
        
        # Pad with zeros if actual_k < k
        if self.actual_k < self.k:
            padded_scores = np.zeros((scores.shape[0], self.k))
            padded_scores[:, :self.actual_k] = scores
            scores = padded_scores
            
        return scores

    def decode(self, scores) -> torch.Tensor:
        """
        Reconstruct genotypes from PC scores.
        Only uses the first actual_k dimensions of scores.
        Returns X_recon (n_samples, n_snps)
        Handles both numpy arrays and PyTorch tensors.
        """
        if self.means is None or self.components_ is None or self.actual_k is None:
            raise RuntimeError("You must call fit() before decode().")
        
        # Only use the first actual_k dimensions of scores
        if isinstance(scores, torch.Tensor):
            effective_scores = scores[:, :self.actual_k]
        else:
            effective_scores = scores[:, :self.actual_k]
        
        # Handle PyTorch tensors
        if isinstance(scores, torch.Tensor):
            device = scores.device
            if isinstance(self.components_, torch.Tensor):
                components = self.components_
            else:
                components = torch.from_numpy(self.components_).to(device)
            if isinstance(self.means, torch.Tensor):
                means = self.means
            else:
                means = torch.from_numpy(self.means).to(device)
                
            Xc_recon = effective_scores @ components
            return Xc_recon + means
        else:
            # Handle numpy arrays (original behavior)
            Xc_recon = effective_scores @ self.components_
            return Xc_recon + self.means
