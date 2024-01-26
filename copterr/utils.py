import numpy as np
import torch

def create_permutation_idxs(n_timepoints, n_permutations, block_len=1):
    permutation_idxs = []
    for _ in range(n_permutations):
        idxs = np.arange(n_timepoints)
        start_idx = np.random.randint(block_len)+1
        start_block, remaining_idxs = np.array_split(idxs, [start_idx])
        blocks = [start_block] + np.array_split(remaining_idxs, np.arange(block_len,len(remaining_idxs),block_len))
        np.random.shuffle(blocks)
        permutation_idxs.append(np.concatenate(blocks))
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
    def zs(x): 
        return (x-torch.mean(x, dim=-2, keepdims=True)) / torch.std(x, dim=-2, keepdims=True, unbiased=unbiased)
    rTmp = torch.sum(zs(A)*zs(B), dim=-2, keepdims=True)
    n = A.shape[0]
    # make sure not to count nans
    nNaN = torch.sum(torch.isnan(zs(A)) | torch.isnan(zs(B)), dim=-2, keepdims=True)
    n = n - nNaN
    r = rTmp/n
    return r.squeeze()

def r2_score(y_true, y_pred):
    error = ((y_true - y_pred) ** 2).sum(0)
    var = ((y_true - y_true.mean(0)) ** 2).sum(0)
    r2 = 1. - error / var
    r2[var == 0] = 0
    return r2

def quantize_alphas(alphas, base=10):
    alphas_quantized = base**np.round(np.emath.logn(base, alphas))
    return alphas_quantized


def alphas_to_deltas(alphas):
    deltas = 2*np.log(alphas.T)
    return deltas


def deltas_to_alphas(deltas):
    alphas = np.exp(deltas.T/2)
    return alphas


def compute_p_values(true_values, permutation_values, two_tailed=False):
    if two_tailed:
        true_values = abs(true_values)
        permutation_values = abs(permutation_values)
    p_values = (permutation_values >= true_values).mean(0)
    return p_values


def _process_perm_idxs(permutation, Y_len, block_len):
    if isinstance(permutation, np.ndarray):
        pass
    elif isinstance(permutation, bool):
        if permutation is True:
            permutation = create_permutation_idxs(Y_len, 1, block_len)[0]
        elif permutation is False:
            permutation = np.arange(Y_len)    
    elif isinstance(permutation, (list, tuple)):
        permutation = np.array(permutation)
    else:
        raise ValueError("Input 'permutation' should be either False (no permutation), True (to auto-generate), or either a list, tuple, or numpy array of indices.")
    return permutation