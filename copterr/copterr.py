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
    """
    A class to perform permutation-based weight fitting using Singular Value Decomposition (SVD).

    Attributes:
        X (array-like): The input feature matrix. shape: n_samples x n_features
        Y (array-like): The target matrix. shape: n_samples x n_voxels
        alphas (array-like): Regularization parameters for each feature. shape: 1 x n_voxels
        device (str): The device to perform computations on ('cpu' or 'cuda').
        dtype (torch.dtype): The data type for torch tensors.
        full_matrices (bool): Whether to compute the full SVD or reduced SVD.
    """

    def __init__(self, X, Y, alphas, device='cpu', dtype=torch.float, full_matrices=False):
        """
        Args:
            X (array-like): The input feature matrix.
            Y (array-like): The target matrix.
            alphas (array-like): Regularization parameters for each feature.
            device (str): The device to perform computations on ('cpu' or 'cuda').
            dtype (torch.dtype): The data type for torch tensors.
            full_matrices (bool): Whether to compute the full SVD or reduced SVD.
        """
        self.device = device
        self.dtype = dtype
        self.X = X
        self.Y = torch.tensor(Y, device=self.device, dtype=self.dtype)
        self.alphas = alphas
        self.weights = torch.zeros((self.X.shape[1], self.Y.shape[1]), device=self.device, dtype=self.dtype)
        self.full_matrices = full_matrices

    def prepare(self, verbose=True):
        """
        Prepares the SVD components for each unique alpha value.

        Args:
            verbose (bool): If True, displays a progress bar during computation.
        """
        unique_alphas, alphas_idxs = np.unique(self.alphas, axis=0, return_inverse=True)
        self._alpha_masks = [alphas_idxs==i for i in range(len(unique_alphas))]
        self._all_VDUt = []
        U, s, Vt = scipy.linalg.svd(self.X, full_matrices=self.full_matrices)
        if verbose:
            unique_alphas = tqdm(unique_alphas, desc='Computing Initial SVDs')
        for alpha in unique_alphas:
            d = s/(s**2 + alpha)
            VDUt = (Vt.T * d) @ U.T
            self._all_VDUt.append(torch.tensor(VDUt, device=self.device, dtype=self.dtype))
        
    def fit_true_weights(self):
        """
        Fits the true weights without permutation.

        Returns:
            torch.Tensor: The fitted weights.
        """
        return self.fit_permutation(permutation=False)
    
    def fit_permutation(self, permutation=True, block_len=1):
        """
        Fits the weights using permutation testing.

        Args:
            permutation (bool): If True, permutes the target data.
            block_len (int): The block length for permutation.

        Returns:
            torch.Tensor: The fitted weights.
        """
        perm_idxs = _process_perm_idxs(permutation, len(self.Y), block_len)
        for alpha_mask, VDUt in zip(self._alpha_masks, self._all_VDUt):
            self.weights[:,alpha_mask] = VDUt @ self.Y[perm_idxs][:,alpha_mask]
        return self.weights
    
    def score(self, Xtest, Ytest, permutation=False, block_len=1, correlation=False):
        """
        Scores the model using test data.

        Args:
            Xtest (array-like): The test feature matrix.
            Ytest (array-like): The test target matrix.
            permutation (bool): If True, permutes the test target data.
            block_len (int): The block length for permutation.
            correlation (bool): If True, returns correlation scores.

        Returns:
            torch.Tensor: The score of the model.
        """
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
    """
    A class to perform permutation-based weight fitting for grouped feature spaces using SVD.

    Attributes:
        X (array-like): The concatenated n features of input feature matrix from multiple groups. shape: its a list with each element being n_samples x n_features
        Y (array-like): The target matrix. shape: n_samples x n_voxels
        alphas (array-like): Regularization parameters for each feature group. shape: n_features x n_voxels
        deltas (numpy.ndarray, optional): Regularization parameters in delta form (deprecated).
                If provided and 'alphas' is None, they will be automatically converted to alphas.
        device (str): The device to perform computations on ('cpu' or 'cuda').
        dtype (torch.dtype): The data type for torch tensors.
        full_matrices (bool): Whether to compute the full SVD or reduced SVD.
    """

    def __init__(self, X, Y, alphas=None, deltas=None, device='cpu', dtype=torch.float, full_matrices=True):
        """
        Args:
            X (array-like): The concatenated input feature matrix from multiple groups.
            Y (array-like): The target matrix.
            alphas (array-like): Regularization parameters for each feature group.
            deltas (numpy.ndarray, optional): Regularization parameters in delta form (deprecated).
                If provided and 'alphas' is None, they will be automatically converted to alphas.
            device (str): The device to perform computations on ('cpu' or 'cuda').
            dtype (torch.dtype): The data type for torch tensors.
            full_matrices (bool): Whether to compute the full SVD or reduced SVD.
        Raises:
            ValueError: If neither 'alphas' nor 'deltas' is provided.
        """
        self.device = device
        self.dtype = dtype
        self.feature_counts = [fs.shape[1] for fs in X]
        self.X = np.concatenate(X, axis=1)
        self.Y = torch.tensor(Y, device=self.device, dtype=self.dtype)
        # Backward compatibility: if alphas is not provided but deltas is provided,
        # compute alphas from deltas. Moving forward, new code should provide alphas.
        if alphas is None:
            if deltas is not None:
                self.alphas = deltas_to_alphas(deltas)
            else:
                raise ValueError("One of 'alphas' or 'deltas' must be provided.")
        else:
            self.alphas = alphas
        self.weights = torch.zeros((self.X.shape[1], self.Y.shape[1]), device=self.device, dtype=self.dtype)
        self.full_matrices = full_matrices

    def prepare(self, verbose=True):
        """
        Prepares the SVD components for each unique alpha value, considering grouped feature spaces.

        Args:
            verbose (bool): If True, displays a progress bar during computation.
        """
        # Convert alphas to a hashable type for unique operation
        alphas_tuple = tuple(map(tuple, self.alphas.T))
        unique_alphas_tuple = list(set(alphas_tuple))
        alphas_idxs = np.array([unique_alphas_tuple.index(a) for a in alphas_tuple])
        unique_alphas = np.array(unique_alphas_tuple)
        
        self._alpha_masks = [alphas_idxs==i for i in range(len(unique_alphas))]
        self._all_VDUt = []
        
        if verbose:
            unique_alphas = tqdm(unique_alphas, desc='Computing Initial SVDs')
            
        for alpha in unique_alphas:
            scaling = np.hstack([np.full(fs_size, a**.5) for fs_size, a in zip(self.feature_counts, alpha)])
            U, s, Vt = scipy.linalg.svd(self.X/scaling, full_matrices=self.full_matrices)
            d = s/(s**2 + 1)
            if self.full_matrices: # this should be back compatible with previous version, but having full_matrices=True may lead to extra computation in SVD with part of the matrix unused because of the indexing any way.
                if self.X.shape[0] < self.X.shape[1]:  # wide matrix
                    VDUt = ((1/scaling * Vt).T[:,:len(d)] * d) @ U.T
                else:  # tall matrix
                    VDUt = ((1/scaling * Vt).T * d) @ U.T[:len(d),:]
            else:
                VDUt = ((1/scaling * Vt).T * d) @ U.T

            self._all_VDUt.append(torch.tensor(VDUt, device=self.device, dtype=self.dtype))
    
    def fit_true_weights(self):
        """
        Fits the true weights without permutation.

        Returns:
            torch.Tensor: The fitted weights.
        """
        return self.fit_permutation(permutation=False)
    
    def fit_permutation(self, permutation=True, block_len=1):
        """
        Fits the weights using permutation testing for grouped feature spaces.

        Args:
            permutation (bool): If True, permutes the target data.
            block_len (int): The block length for permutation.

        Returns:
            torch.Tensor: The fitted weights.
        """
        perm_idxs = _process_perm_idxs(permutation, len(self.Y), block_len)
        for alpha_mask, VDUt in zip(self._alpha_masks, self._all_VDUt):
            self.weights[:,alpha_mask] = VDUt @ self.Y[perm_idxs][:,alpha_mask]
        return self.weights
    
    def score(self, Xtest, Ytest, permutation=False, block_len=1, correlation=False):
        """
        Scores the model using test data for grouped feature spaces.

        Args:
            Xtest (array-like): The test feature matrix.
            Ytest (array-like): The test target matrix.
            permutation (bool): If True, permutes the test target data.
            block_len (int): The block length for permutation.
            correlation (bool): If True, returns correlation scores.

        Returns:
            torch.Tensor: The score of the model.
        """
        Xtest = torch.tensor(Xtest, device=self.device, dtype=self.dtype)
        Ytest = torch.tensor(Ytest, device=self.device, dtype=self.dtype)
        perm_idxs = _process_perm_idxs(permutation, Ytest.shape[-2], block_len)
        Ypred = Xtest@self.weights
        score = column_corr_torch(Ypred, Ytest[...,perm_idxs,:])
        if correlation:
            return score
        else:
            return torch.sign(score)*torch.square(torch.abs(score))