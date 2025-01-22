import numpy as np
import torch
import scipy
from tqdm import tqdm
from .utils import (
    _process_perm_idxs, 
    column_corr_torch, 
    deltas_to_alphas, 
)

class PermuteWeights():
    def __init__(self, X, Y, alphas, device='cpu', dtype=torch.float):
        self.device = device
        self.dtype = dtype
        self.X = X
        self.Y = torch.tensor(Y, device=self.device, dtype=self.dtype)
        self.alphas = alphas
        self.weights = torch.zeros((self.X.shape[1], self.Y.shape[1]), device=self.device, dtype=self.dtype)

    def prepare(self, verbose=True):
        unique_alphas, alphas_idxs = np.unique(self.alphas, axis=0, return_inverse=True)
        self._alpha_masks = [alphas_idxs==i for i in range(len(unique_alphas))]
        self._all_VDUt = []
        U, s, Vt = scipy.linalg.svd(self.X, full_matrices=False)
        if verbose:
            unique_alphas = tqdm(unique_alphas, desc='Computing Initial SVDs')
        for alpha in unique_alphas:
            d = s/(s**2 + alpha)
            VDUt = (Vt.T * d) @ U.T
            self._all_VDUt.append(torch.tensor(VDUt, device=self.device, dtype=self.dtype))
        
    def fit_true_weights(self):
        return self.fit_permutation(permutation=False)
    
    def fit_permutation(self, permutation=True, block_len=1):
        perm_idxs = _process_perm_idxs(permutation, len(self.Y), block_len)
        for alpha_mask, VDUt in zip(self._alpha_masks, self._all_VDUt):
            self.weights[:,alpha_mask] = VDUt @ self.Y[perm_idxs][:,alpha_mask]
        return self.weights
    
    def score(self, Xtest, Ytest, permutation=False, block_len=1, correlation=False):
        Xtest = torch.tensor(Xtest, device=self.device, dtype=self.dtype)
        Ytest = torch.tensor(Ytest, device=self.device, dtype=self.dtype)
        perm_idxs = _process_perm_idxs(permutation, Ytest.shape[-2], block_len)
        Ypred = Xtest@self.weights
        score = column_corr_torch(Ypred, Ytest[...,perm_idxs,:])
        if correlation:
            return score
        else:
            return torch.sign(score)*torch.square(torch.abs(score))


class PermuteWeightsGrouped():
    def __init__(self, X, Y, deltas, device='cpu', dtype=torch.float):
        self.device = device
        self.dtype = dtype
        self.feature_counts = [fs.shape[1] for fs in X]
        self.X = np.concatenate(X, axis=1)
        print(f"Concatenated X shape: {self.X.shape}")
        self.Y = torch.tensor(Y, device=self.device, dtype=self.dtype)
        print(f"Y shape: {self.Y.shape}")
        if deltas is not None:
            self.alphas = deltas_to_alphas(deltas)
        print(f"Alphas shape: {self.alphas.shape}")
        self.weights = torch.zeros((self.X.shape[1], self.Y.shape[1]), device=self.device, dtype=self.dtype)
        print(f"Weights shape: {self.weights.shape}")

    def prepare(self, verbose=True):
        unique_alphas, alphas_idxs = np.unique(self.alphas, axis=0, return_inverse=True)
        print(f"Number of unique alphas: {len(unique_alphas)}")
        # Create masks for each unique alpha combination
        self._alpha_masks = [torch.arange(self.Y.shape[1], device=self.device) == i for i in range(len(unique_alphas))]
        self._all_VDUt = []
        if verbose:
            unique_alphas = tqdm(unique_alphas, desc='Computing Initial SVDs')
        for alpha in unique_alphas:
            # Create scaling vector for each feature group
            scaling = np.concatenate([np.full(fs_size, a**.5) for fs_size, a in zip(self.feature_counts, alpha)])
            print(f"Scaling shape: {scaling.shape}")
            
            # Scale the features
            X_scaled = self.X / scaling[None, :]  # Scale columns
            print(f"X_scaled shape: {X_scaled.shape}")
            
            # Compute SVD
            U, s, Vt = scipy.linalg.svd(X_scaled, full_matrices=False)
            print(f"U shape: {U.shape}, s shape: {s.shape}, Vt shape: {Vt.shape}")
            
            # Compute d
            d = s/(s**2 + 1)
            print(f"d shape: {d.shape}")
            
            # Scale V by the inverse of the scaling factors
            V = Vt.T
            print(f"V shape: {V.shape}")
            
            # Compute final matrix in steps
            V_scaled = V / scaling[None, :]  # Scale columns
            print(f"V_scaled shape: {V_scaled.shape}")
            VD = V_scaled * d[:, None]  # Scale rows
            print(f"VD shape: {VD.shape}")
            VDUt = VD @ U.T
            print(f"VDUt shape: {VDUt.shape}")
            
            self._all_VDUt.append(torch.tensor(VDUt, device=self.device, dtype=self.dtype))
    
    def fit_true_weights(self):
        return self.fit_permutation(permutation=False)
    
    def fit_permutation(self, permutation=True, block_len=1):
        perm_idxs = _process_perm_idxs(permutation, len(self.Y), block_len)
        # Use all columns for each VDUt since we're using grouped alphas
        for VDUt in self._all_VDUt:
            self.weights = VDUt @ self.Y[perm_idxs]
        return self.weights
    
    def score(self, Xtest, Ytest, permutation=False, block_len=1, correlation=False):
        Xtest = torch.tensor(Xtest, device=self.device, dtype=self.dtype)
        Ytest = torch.tensor(Ytest, device=self.device, dtype=self.dtype)
        perm_idxs = _process_perm_idxs(permutation, Ytest.shape[-2], block_len)
        Ypred = Xtest@self.weights
        score = column_corr_torch(Ypred, Ytest[...,perm_idxs,:])
        if correlation:
            return score
        else:
            return torch.sign(score)*torch.square(torch.abs(score))