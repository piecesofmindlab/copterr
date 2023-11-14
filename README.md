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

COPTeRR is a Python package for efficiently computing permutation tests on ridge regression model weights, performance, and other metrics, without re-fitting the model for each permutation.  This enables rapid estimation of null distributions (and thus, p-values) for any weights-derived model characteristic.

## Simple Example
First, let's make some basic demo data.

``` python
import numpy as np
import torch
from tqdm import tqdm

n_timepoints_train, n_timepoints_test = 100, 50
n_features, n_targets = 25, 5
noise_amount = 1.5

X_train = np.random.rand(n_timepoints_train, n_features)
X_test = np.random.rand(n_timepoints_test, n_features)
true_weights = np.random.randn(n_features, n_targets)
Y_train = X_train @ true_weights + noise_amount * np.random.randn(n_timepoints_train, n_targets)
Y_test = X_test @ true_weights + noise_amount * np.random.randn(n_timepoints_test, n_targets)
```

COPTeRR computes permutation tests on already-fit ridge regression models. You can perform this initial fit using a variety of packages, but we recommend [himalaya](https://github.com/gallantlab/himalaya).

``` python
from himalaya.ridge import RidgeCV

model = RidgeCV(alphas=np.logspace(-2, 5, 8))
model.fit(X_train, Y_train)
model_performance = model.score(X_test, Y_test)
print("R2 Score Per Target: ", model_performance)
```

This performance seems pretty good, but how can we be sure that it's significant? We can determine this by generating a null distribution of chance values using COPTeRR!

``` python
from copterr import PermuteWeights, create_permutation_idxs, column_corr_torch

permuter = PermuteWeights(X_train, Y_train, model.best_alphas_)
permuter.prepare()
```

First, let's check that COPTeRR is estimating model weights the same way that himalaya is.

``` python
himalaya_weights = model.coef_
copterr_weights = permuter.compute_true_weights()
print(np.allclose(himalaya_weights, copterr_weights))
```
Looks good! Now, let's perform a bunch of permutations, and compute p-values for the model's performance.
``` python
perm_indices = create_permutation_idxs(n_timepoints_train, n_permutations=10000, block_len=5)
perm_performance = []
for indices in tqdm(perm_indices):
    perm_weights = permuter.permute_weights(indices)
    perm_predictions = torch.tensor(X_test, dtype=torch.float)@perm_weights
    perm_correlations = column_corr_torch(perm_predictions, Y_test)
    perm_r2 = perm_correlations**2
    perm_performance.append(perm_r2.cpu().numpy())

p_values = np.mean(np.array(perm_performance) >= model_performance, axis=0)
print("P-Values:", p_values)
```

Feel free to vary the noise_amount variable and check out the effects--lower amounts of noise should yield lower p-values for our model's performance. Also note that model r2 is just a single example--anything you compute from your model's weights can be tested in this manner!