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
        self.Y = torch.tensor(Y, device=self.device, dtype=self.dtype)
        if deltas is not None:
            self.alphas = deltas_to_alphas(deltas)
        self.weights = torch.zeros((self.X.shape[1], self.Y.shape[1]), device=self.device, dtype=self.dtype)

    def prepare(self, verbose=True):
        unique_alphas, alphas_idxs = np.unique(self.alphas, axis=0, return_inverse=True)
        self._alpha_masks = [alphas_idxs==i for i in range(len(unique_alphas))]
        self._all_VDUt = []
        if verbose:
            unique_alphas = tqdm(unique_alphas, desc='Computing Initial SVDs')
        for alpha in unique_alphas:
            scaling = np.hstack([np.full(fs_size, a**.5) for fs_size, a in zip(self.feature_counts, alpha)])
            U, s, Vt = scipy.linalg.svd(self.X/scaling, full_matrices=True)
            d = s/(s**2 + 1)
            VDUt = ((1/scaling * Vt).T[:,:len(d)] * d) @ U.T
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