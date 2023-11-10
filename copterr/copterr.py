import numpy as np
import torch
import scipy
from tqdm import tqdm


def create_permutation_idxs(n_tpts, n_permutations, block_len):
    """Very similar to utils.get_permutation_index, but varies where it begins slicing,
    such that the values within each block will vary, not just the block order.
    Should probably be merged in some way into a single function in the future.

    Parameters
    ----------
    n_tpts : int
        Length of the dimension to be permuted.
    n_permutations : int
        Number of different rows of permuted indices to return.
        Returned array will be n_permutations X n_tpts
    block_len : int
        Length of 'blocks' to divide list of indices into prior to
        permuting these blocks.  Helps maintain temporal correlations that
        exist regardless of whether meaningful signal exists--correlations that
        would be destroyed with block-less permutation.

    Returns
    -------
    np.ndarray
        n_permutations X n_tpts array of block-permuted indices.  To loop through permutations,
        loop through rows.
    """
    permutation_idxs = np.empty((n_permutations, n_tpts), dtype=int)
    n_blocks = np.ceil(n_tpts / block_len)
    mod = n_tpts % block_len

    for permutation in range(n_permutations):
        idxs = []
        mod_block = np.random.randint(0, n_blocks)
        i = 0
        block_idxs = []
        current_mod = 0
        while True:
            if i == n_tpts:
                idxs.append(block_idxs)
                break
            if len(idxs) + 1 == mod_block:
                current_mod = block_len - mod
            if i > 0 and (i + current_mod) % block_len == 0:
                idxs.append(block_idxs)
                block_idxs = []
            block_idxs.append(i)
            i += 1
        np.random.shuffle(idxs)
        permutation_idxs[permutation] = np.hstack(idxs)
    return permutation_idxs


def column_corr_torch(A, B, unbiased=False):
    """Similar to Stats.utils.column_corr, except works with tensors.  Should be tested more
    thoroughly for possible differences.

    Parameters
    ----------
    A : np.ndarray
        First 2d array to measure correlation with other array.
    B : np.ndarray
        Other 2d array to measure correlation with other array (order does not matter).
    unbiased : bool
        Whether to compute std as a biased estimator.  False computes equivalently to 
        dof=0 in column_corr.

    Returns
    -------
    np.ndarray
        1d array, with each value representing the correlation between the corresponding
        columns of arrays A and B.
    """
    A, B = torch.tensor(A), torch.tensor(B)
    def zs(x): 
        return (x-torch.mean(x, dim=0)) / torch.std(x, dim=0, unbiased=unbiased)
    rTmp = torch.sum(zs(A)*zs(B), dim=0)
    n = A.shape[0]
    # make sure not to count nans
    nNaN = torch.sum(torch.isnan(zs(A)) | torch.isnan(zs(B)), dim=0)
    n = n - nNaN
    r = rTmp/n
    return r


class PermuteWeights():
    def __init__(self, X, Y, alphas, device='cpu', dtype=torch.float):
        self.device = device
        self.dtype = dtype
        self.X = X
        self.Y = torch.tensor(Y, device=self.device, dtype=self.dtype)
        self.alphas = alphas
        self.weights = torch.zeros((self.X.shape[1], self.Y.shape[1]), device=self.device, dtype=self.dtype)

    def prepare(self):
        unique_alphas, alphas_idxs = np.unique(self.alphas, axis=0, return_inverse=True)
        self._alpha_masks = [alphas_idxs==i for i in range(len(unique_alphas))]
        self._all_VDUt = []
        U, s, Vt = scipy.linalg.svd(self.X, full_matrices=False)
        for alpha in unique_alphas:
            d = s/(s**2 + alpha)
            VDUt = (Vt.T * d) @ U.T
            self._all_VDUt.append(torch.tensor(VDUt, device=self.device, dtype=self.dtype))
        
    def permute_weights(self, permutation):
        for alpha_mask, VDUt in zip(self._alpha_masks, self._all_VDUt):
            self.weights[:,alpha_mask] = VDUt @ self.Y[permutation][:,alpha_mask]
        return self.weights

    def compute_true_weights(self):
        return self.permute_weights(np.arange(len(self.Y)))


class PermuteWeightsGrouped():
    def __init__(self, X, Y, deltas, device='cpu', dtype=torch.float):
        self.device = device
        self.dtype = dtype
        self.feature_counts = [fs.shape[1] for fs in X]
        self.X = np.concatenate(X, axis=1)
        self.Y = torch.tensor(Y, device=self.device, dtype=self.dtype)
        self.alphas = np.exp(deltas.T/2)
        self.weights = torch.zeros((self.X.shape[1], self.Y.shape[1]), device=self.device, dtype=self.dtype)

    def prepare(self):
        unique_alphas, alphas_idxs = np.unique(self.alphas, axis=0, return_inverse=True)
        self._alpha_masks = [alphas_idxs==i for i in range(len(unique_alphas))]
        self._all_VDUt = []
        for alpha in unique_alphas:
            scaling = np.hstack([np.ones(fs_size)*a for fs_size, a in zip(self.feature_counts, alpha)])
            U, s, Vt = scipy.linalg.svd(self.X*scaling, full_matrices=True)
            d = s/(s**2 + 1)
            VDUt = ((scaling * Vt).T[:,:len(d)] * d) @ U.T
            self._all_VDUt.append(torch.tensor(VDUt, device=self.device, dtype=self.dtype))
    
    def permute_weights(self, permutation):
        for alpha_mask, VDUt in zip(self._alpha_masks, self._all_VDUt):
            self.weights[:,alpha_mask] = VDUt @ self.Y[permutation][:,alpha_mask]
        return self.weights

    def compute_true_weights(self):
        return self.permute_weights(np.arange(len(self.Y)))