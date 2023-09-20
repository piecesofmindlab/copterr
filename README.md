# COPTeRR
<pre>  
<b>C</b>ompute-<b>O</b>ptimized <b>P</b>ermutation <b>Te</b>sting for <b>R</b>idge <b>R</b>egression
                                                                       ______.........--=T=--.........______
                                                                          .             |:|
                                                                     :-. //           /""""""-.
                                                                     ': '-._____..--""(""""""()`---.__
                                                                      /:   _..__   ''  ":""""'[] |""`\\
                                                                      ': :'     `-.     _:._     '"""" :
                                                                       ::          '--=:____:.___....-"
                                                                                         O"       O"</pre>

COPTeRR is a Python Package for efficiently computing a permutation tests on regression model weights, performance and other metrics without re-fitting the model for each permutation.  This enables rapid estimation of null distributions (and thus, p-values) for essentially any model characteristic.

## Getting started
To run, you'll need to import either `gen_permutation_weights` or `gen_permutation_weights_grouped` (for multi-feature-set fits) and provide the function with three inputs: X (input features), Y (output targets) and alphas (previously-selected regularization parameters).  For this example, we'll generate fake data.

``` python
import numpy as np
import torch
from tqdm import tqdm
from scipy.stats import zscore

n_timepoints_train, n_timepoints_test = 100, 50
n_features = 25
n_targets = 3
noise_amount = 2

X_train = np.random.rand(n_timepoints_train, n_features)
X_test = np.random.rand(n_timepoints_test, n_features)

true_weights = np.random.randn(n_features, n_targets)
Y_train = X_train @ true_weights + noise_amount * np.random.randn(n_timepoints_train, n_targets)
Y_test = X_test @ true_weights + noise_amount * np.random.randn(n_timepoints_test, n_targets)
Y_train = zscore(Y_train, axis=0)
Y_test = zscore(Y_test, axis=0)
```

You'll need to generate the regression model you intend to test.  In this example, we'll use the package `himalaya` for cross-validated ridge regression.

``` python
from himalaya.ridge import RidgeCV

model_cv = RidgeCV(alphas=np.logspace(-2, 5, 8))
model_cv.fit(X_train, Y_train)
model_performance = model_cv.score(X_test, Y_test)
print("R2 Score Per Target: ", model_performance)
```
This performance seems pretty good, but how can we be sure that it's significant?  We can determine this by generating a null distribution of chance values using COPTeRR!

``` python
from copterr import gen_permutation_weights, get_permutation_idxs, column_corr_torch

n_permutations=10000
perm_indices = get_permutation_idxs(n_timepoints_train, n_permutations=n_permutations, block_len=5)

model_alphas = model_cv.best_alphas_
# Creates a generator object that permutes the targets and re-estimates weights
generator = gen_permutation_weights(X_train, Y_train, model_alphas, perm_indices, verbose=False)

perm_performance = []
for perm_weights in tqdm(generator, total=n_permutations):
    # Estimate model performance using permutation weights
    perm_predictions = torch.tensor(X_test, dtype=torch.float)@perm_weights
    perm_correlations = column_corr_torch(perm_predictions, Y_test)
    perm_r2 = perm_correlations**2
    perm_performance.append(perm_r2.cpu().numpy())

p_values = np.mean(np.array(perm_performance) >= model_performance, axis=0)
print("P-Values:", p_values)
```
Feel free to vary the noise_amount variable and check out the effects--lower amounts of noise should yield lower p-values for our model's performance. Also note that model r<sup>2</sup> is just a single example--anything you compute from your model's weights can be tested in this manner!

## Setup
COPTeRR requires only `numpy`, `pytorch`, `scipy` and `tqdm` and can be used with a variety of Python 3 versions. `cudatoolkit` is required for CUDA-based GPU acceleration.

``` bash
# Optional clean environment
conda create -n copterr python=3.10
conda activate copterr

# General dependencies
conda install numpy torch scipy tqdm -y

# Optional, for CUDA
conda install -c nvidia cudatoolkit -y

# Optional, for Jupyter notebook usage
conda install ipython ipywidgets ipykernel -y

# Optional, for model fitting and performance comparison
pip install himalaya
```
