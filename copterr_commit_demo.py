'''
Summary
Fix SVD dimension mismatch and optimize matrix computations. Added detailed documentation and backward compatibility for redundant delta to alpha conversion.

Main Changes:
1. Added dimension-aware matrix handling for SVD operations:
   - For wide matrices (samples < features): slice Vt.T
   - For tall matrices (samples > features): slice U.T
2. Maintained backward compatibility with full_matrices=True
3. Added optimized path using full_matrices=False that avoids computing unused matrix components
4. Added detailed documentation for copterr.py file.
5. Ensured backward compatibility for the `deltas` parameter.
6. Added explicit imports in the `__init__.py` file.

Technical Details:
- Previous code assumed wide matrices (common in fMRI).
- New code handles both wide and tall matrices correctly.
- Setting full_matrices=False is computationally optimal but optional.

Demonstration:
See `copterr_commit_demo.py` for a demonstration of the behavior with different matrix shapes.
'''


# %%
from copterr import PermuteWeights, PermuteWeightsGrouped, alphas_to_deltas, quantize_alphas
import numpy as np
import torch
import os
from vm_tools.utils import avg_wts_torch
from tqdm import tqdm
import vm_tools as vmt
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from tqdm import tqdm
import re

# %%
# tall fake data with features: X1 be 8000x1000 (sample x feature1), X2 be 8000x500 (sample x feature2) Y be 8000x2000 (sample x voxel), alpha be 2x2000 (n_features x voxel)
# currently the large dimension is only to demonstrate the speed difference between full_matrices=True and full_matrices=False, for initial testing, we should be fine with much smaller dimensions
n_samples = 8000
feature_numbers = 2
n_features1 = 1000
n_features2 = 500
n_voxels = 2000
trn_fs1 = torch.randn(n_samples, n_features1)
trn_fs2 = torch.randn(n_samples, n_features2)
trn_fs = [trn_fs1, trn_fs2]
Y = torch.randn(n_samples, n_voxels)
# this is consistent with the alpha
alpha = torch.from_numpy(np.random.choice([1.e+02, 1.e+04, 1.e+06, 1.e+08, 1.e+10, 1.e+12, 1.e+14, 1.e+16, 1.e+18, 1.e+20, 1.e+22, 1.e+24, np.inf], size=(feature_numbers, n_voxels))).float()

DEVICE = 'cuda:0'
DTYPE = torch.float32

# %%
# took 32 seconds with full_matrices=True
permuter = PermuteWeightsGrouped(trn_fs, Y, deltas=alphas_to_deltas(alpha), device=DEVICE, dtype=DTYPE)
permuter.prepare()

# %%
# previously on main branch would yield error:
# ---------------------------------------------------------------------------
# ValueError                                Traceback (most recent call last)
# ~/remote_mounts/pomcloud0/projects_shared/VMBA/notebooks/NSD_joseph_ipynbs/3_variance_partition/3.5_copterr_comit_demo.py in <module>
#       1 # %%
#       2 permuter = PermuteWeightsGrouped(trn_fs, Y, deltas=alphas_to_deltas(alpha), device=DEVICE, dtype=DTYPE, full_matrices=True)
# ----> 3 permuter.prepare()

# ~/anaconda3/envs/pomlab/lib/python3.7/site-packages/copterr/copterr.py in prepare(self, verbose)
#      73             U, s, Vt = scipy.linalg.svd(self.X/scaling, full_matrices=self.full_matrices)
#      74             d = s/(s**2 + 1)
# ---> 75             VDUt = ((1/scaling * Vt).T * d) @ U.T
#      76             # VDUt = ((1/scaling * Vt).T[:,:len(d)] * d) @ U.T
#      77             # if full_matrices is True, then we should use this line

# ValueError: matmul: Input operand 1 has a mismatch in its core dimension 0, with gufunc signature (n?,k),(k,m?)->(n?,m?) (size 80 is different from 15)
# %%
# took 9 seconds with full_matrices=False
permuter = PermuteWeightsGrouped(trn_fs, Y, alphas=alpha, device=DEVICE, dtype=DTYPE, full_matrices=False)
permuter.prepare()
