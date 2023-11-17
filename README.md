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
from tqdm import tqdm

n_timepoints_train, n_timepoints_test = 500, 100
n_features, n_targets = 10, 3

# Create features
X_train = np.random.rand(n_timepoints_train, n_features)
X_test = np.random.rand(n_timepoints_test, n_features)
true_weights = np.random.randn(n_features, n_targets)

# Create targets
Y_train = X_train @ true_weights
Y_test = X_test @ true_weights

# Add different amounts of noise
Y_train[:,0] += 1  * np.random.randn(n_timepoints_train)
Y_train[:,1] += 3  * np.random.randn(n_timepoints_train)
Y_train[:,2] += 15 * np.random.randn(n_timepoints_train)

Y_test[:,0] += 1  * np.random.randn(n_timepoints_test)
Y_test[:,1] += 3  * np.random.randn(n_timepoints_test)
Y_test[:,2] += 15 * np.random.randn(n_timepoints_test)
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
from copterr import PermuteWeights

permuter = PermuteWeights(X_train, Y_train, model.best_alphas_)
permuter.prepare()

himalaya_weights = model.coef_
copterr_weights = permuter.fit_true_weights()
print('Weights Equivalent:', np.allclose(himalaya_weights, copterr_weights, atol=1e-5))
```

Now, let's perform a bunch of permutations, and compute p-values for the model's performance.
``` python
from copterr.utils import compute_p_values

perm_performance = []
for permutation in tqdm(range(10000)):
    perm_weights = permuter.fit_permutation(permutation=True)
    perm_r2 = permuter.score(X_test, Y_test, permutation=True)
    perm_performance.append(perm_r2.numpy())

p_values = compute_p_values(model_performance, perm_performance)
print("P-Values:", p_values)
```

Feel free to vary the noise_amount variable and check out the effects--lower amounts of noise should yield smaller p-values for our model's performance. Also note that model r2 is just a single example--anything you compute from your model's weights can be tested in this manner!