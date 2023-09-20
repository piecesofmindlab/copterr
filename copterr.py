import numpy as np
import torch
import scipy
from tqdm import tqdm


def get_permutation_idxs(n_tpts, n_permutations, block_len):
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


def gen_permutation_weights(X, Y, alphas, permutation_idxs=None, dtype=torch.float, device='cpu', verbose=True):
    """Generator object for recomputing regression weights given a single set of features, 
    a set of brain responses (which are permuted each iteration), and precomputed 
    voxel-wise alphas.  For multiple sets of features, use gen_permutation_weights_grouped.

    Parameters
    ----------
    X : np.ndarray
        Design matrix for a single feature set
    Y : np.ndarray
        Matrix of brain responses.
    alphas : np.ndarray
        1D array of alphas, one for each voxel/column of Y.
    permutation_idxs : None, optional
        Array of permuted indices, as returned from get_permutation_idxs.  If None, will compute on
        the unshuffled Y values a single time (useful for verifying parity with the original fitting).
    n_permutations : int, optional
        Number of permutations to yield before stopping.  If array is passed for permutation_idxs,
        this kwarg will be ignored.
    block_len : int, optional
        Length of 'blocks' of indices which are permuted.  If array is passed for permutation_idxs,
        this kwarg will be ignored.
    dtype : TYPE
        torch dtype to use in final computation and yielded weights.  Should be some variation of
        float, with float16 yielding a small decrease in memory usage.
    device : str, optional
        Device for pytorch to use.  If 'cuda', will use gpu (requires CUDA and a CUDA-compatible GPU).
        If 'cpu', will use CPU and RAM instead.

    Yields
    ------
    torch.tensor
        Tensor containing recomputed weights.  To convert to a numpy array, add .numpy() (if 
        device=='cpu') or .cpu().numpy() (if device=='cuda').
    """
    if verbose: print("Performing initial setup...")
    if isinstance(permutation_idxs, type(None)):
        permutation_idxs = np.arange(len(Y))
    X, Y, alphas = [np.nan_to_num(var) for var in [X, Y, alphas]]
    unique_alphas = np.unique(alphas)
    alpha_idxs = [np.argwhere(unique_alphas == alpha).flatten()[0] for alpha in alphas]
    U, s, Vt = scipy.linalg.svd(X, full_matrices=True)
    VDUt = np.empty((len(unique_alphas), *X.T.shape))
    D = np.zeros(X.T.shape)
    for i, alpha in enumerate(unique_alphas):
        np.fill_diagonal(D, s/(s**2 + alpha**2))
        VDUt[i] = Vt.T @ D @ U.T
    VDUt, Y = [torch.tensor(var, dtype=dtype, device=device)
               for var in (VDUt, Y)]
    if verbose: print("Initial setup done.")
    for permutation in permutation_idxs:
        full_weights = torch.zeros((X.shape[1], Y.shape[1]), dtype=dtype, device=device)
        for i, alpha in enumerate(unique_alphas):
            alphas_mask = np.array(alpha_idxs)==i
            full_weights[:,alphas_mask] = VDUt[i] @ Y[permutation][:,alphas_mask]
        yield full_weights
    del VDUt, Y
    torch.cuda.empty_cache()


def gen_permutation_weights_grouped(X, Y, alphas, permutation_idxs=None, dtype=torch.float, device='cpu', verbose=True):
    """Generator object for recomputing regression weights given a list of feature sets, 
    a set of brain responses (which are permuted each iteration), and precomputed 
    voxel-wise alphas.  For a single set of features, use gen_permutation_weights.

    Parameters
    ----------
    X : list
        List of np.ndarrays, with each array being a design matrix for a single feature set
    Y : np.ndarray
        Matrix of brain responses.
    alphas : np.ndarray
        2D array of alphas, with each row representing the alphas for a single voxel/column of Y.
    permutation_idxs : None, optional
        Array of permuted indices, as returned from get_permutation_idxs.  If None, will compute on
        the unshuffled Y values a single time (useful for verifying parity with the original fitting).
    n_permutations : int, optional
        Number of permutations to yield before stopping.  If array is passed for permutation_idxs,
        this kwarg will be ignored.
    block_len : int, optional
        Length of 'blocks' of indices which are permuted.  If array is passed for permutation_idxs,
        this kwarg will be ignored.
    dtype : TYPE
        torch dtype to use in final computation and yielded weights.  Should be some variation of
        float, with float16 yielding a small decrease in memory usage.
    device : str, optional
        Device for pytorch to use.  If 'cuda', will use gpu (requires CUDA and a CUDA-compatible GPU).
        If 'cpu', will use CPU and RAM instead.

    Yields
    ------
    torch.tensor
        Tensor containing recomputed weights.  To convert to a numpy array, add .numpy() (if 
        device=='cpu') or .cpu().numpy() (if device=='cuda').
    """
    if verbose: print("Performing initial computations...")
    if isinstance(permutation_idxs, type(None)):
        permutation_idxs = [np.arange(len(Y))]
    X = [np.nan_to_num(x) for x in X]
    Y, alphas = [np.nan_to_num(var) for var in [Y, alphas]]
    fs_sizes = [fs.shape[1] for fs in X]
    X = np.concatenate(X, axis=1)
    unique_alphas = np.unique(alphas, axis=0)
    alpha_idxs = [np.where((unique_alphas == alpha).all(axis=1))[
        0][0] for alpha in alphas]
    D = np.zeros(X.T.shape)
    scaled_VDUt = np.empty((len(unique_alphas), *X.T.shape))
    if verbose:
        unique_alphas = tqdm(unique_alphas)
        unique_alphas.set_description("Performing SVDs...")
    for i, alpha in enumerate(unique_alphas):
        C = np.hstack([np.full((fs_size), a) for fs_size, a in zip(fs_sizes, alpha)])
        U, s, Vt = scipy.linalg.svd(X/C, full_matrices=True)
        np.fill_diagonal(D, (s/(s**2 + 1)))
        scaled_VDUt[i] = np.diag(1/C) @ Vt.T @ D @ U.T
    scaled_VDUt, Y = [torch.tensor(var, dtype=dtype, device=device)
                      for var in (scaled_VDUt, Y)]
    if verbose: print("Initial computation done.")
    for permutation in permutation_idxs:
        full_weights = torch.zeros((X.shape[1], Y.shape[1]), dtype=dtype, device=device)
        for i, alpha in enumerate(unique_alphas):
            alphas_mask = np.array(alpha_idxs)==i
            full_weights[:,alphas_mask] = scaled_VDUt[i] @ Y[permutation][:,alphas_mask]
        yield full_weights
    del scaled_VDUt, Y
    torch.cuda.empty_cache()