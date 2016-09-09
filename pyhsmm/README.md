# Sampling Inference for Bayesian HSMMs and HMMs #
This is a Python library for approximate unsupervised sampling inference in
Bayesian Hidden Markov Models (HMMs) and explicit-duration Hidden semi-Markov
Models (HSMMs), focusing on the Bayesian Nonparametric extensions, the HDP-HMM
and HDP-HSMM, via the weak-limit approximation.

There are also some plugins to extend the functionality:

* [factorial models](https://github.com/mattjj/pyhsmm-factorial)
* [autoregressive models](https://github.com/mattjj/pyhsmm-autoregressive)
* [collapsed HDP sampling inference](https://github.com/mattjj/pyhsmm-collapsedinfinite).

The inference can be run in parallel over multiple cores and/or multiple
machines (even on EC2!) using [ipython](https://github.com/ipython/ipython)'s
excellent `ipython.parallel` module. Someday I might even document how to do
it!

## Installing ##
You can clone this library and its dependencies into your current working directory with

```bash
git clone --recursive git://github.com/mattjj/pyhsmm.git
```

For the Python interpreter to be able to import pyhsmm, you'll need it on your
Python path. Since the current working directory is usually included in the
Python path, you can probably run the examples from the same directory in which
you run the git clone with commands like `python pyhsmm/examples/hsmm.py`. You
might also want to add pyhsmm to your global Python path (e.g. by copying it to
your site-packages directory).

To pull updates, you can do this:

```bash
cd /path/to/pyhsmm
git pull
git submodule update --init --recursive
```

The library depends on `numpy`, `scipy`, and, for visualization, `matplotlib`.

Disabling assertions may speed things up; to disable assertions in CPython,
call it with the `-O` flag.

## A Simple Demonstration ##
Here's how to draw from the HDP-HSMM posterior over HSMMs given a sequence of
observations. (The same example, along with the code to generate the synthetic
data loaded in this example, can be found in `examples/basic.py`.)

Let's say we have some 2D data in a data.txt file:

```bash
$ head -5 data.txt
-3.711962552600095444e-02 1.456401745267922598e-01
7.553818775915704942e-02 2.457422192223903679e-01
-2.465977987699214502e+00 5.537627981813508793e-01
-7.031638516485749779e-01 1.536468304146855757e-01
-9.224669847039665971e-01 3.680035337673161489e-01
```

In Python, we can plot the data in a 2D plot, collapsing out the time dimension:

```python
import numpy as np
from matplotlib import pyplot as plt

data = np.loadtxt('data.txt')
plt.plot(data[:,0],data[:,1],'kx')
```

![2D data](http://www.mit.edu/~mattjj/github/pyhsmm/data.png)

We can also make a plot of time versus the first principal component:

```python
from pyhsmm.util.plot import pca_project_data
plt.plot(pca_project_data(data,1))
```

![Data first principal component vs time](http://www.mit.edu/~mattjj/github/pyhsmm/data_vs_time.png)

To learn an HSMM, we'll use `pyhsmm` to create an `HSMM` instance using some
reasonable hyperparameters. We'll ask this model to infer the number of states
as well (since an HDP-HSMM is instantiated by default), so we'll give it an
`Nmax` parameter:

```python
import pyhsmm
import pyhsmm.basic.distributions as distributions

obs_dim = 2
Nmax = 25

obs_hypparams = {'mu_0':np.zeros(obs_dim),
                'sigma_0':np.eye(obs_dim),
                'kappa_0':0.3,
                'nu_0':obs_dim+5}
dur_hypparams = {'alpha_0':2*30,
                 'beta_0':2}

obs_distns = [distributions.Gaussian(**obs_hypparams) for state in range(Nmax)]
dur_distns = [distributions.PoissonDuration(**dur_hypparams) for state in range(Nmax)]

posteriormodel = pyhsmm.models.HSMM(
        alpha=6.,gamma=6., # better to sample over these; see concentration-resampling.py
        init_state_concentration=6., # pretty inconsequential
        obs_distns=obs_distns,
        dur_distns=dur_distns)
```

(The first two arguments set the "new-table" proportionality constant for the
meta-Chinese Restaurant Process and the other CRPs, respectively, in the HDP
prior on transition matrices. For this example, they really don't matter at
all, but on real data it's much better to infer these parameters, as in
`examples/concentration_resampling.py`.)

Then, we add the data we want to condition on:

```python
posteriormodel.add_data(data,trunc=60)
```

The `trunc` parameter is an optional argument that can speed up inference: it
sets a truncation limit on the maximum duration for any state. If you don't
pass in the `trunc` argument, no truncation is used and all possible state
duration lengths are considered. (pyhsmm has fancier ways to speed up message
passing over durations, but they aren't documented.)

If we had multiple observation sequences to learn from, we could add them to the
model just by calling `add_data()` for each observation sequence.

Now we run a resampling loop. For each iteration of the loop, all the latent
variables of the model will be resampled by Gibbs sampling steps, including the
transition matrix, the observation means and covariances, the duration
parameters, and the hidden state sequence. We'll also copy some samples so that
we can plot them.

```python
models = []
for idx in progprint_xrange(150):
    posteriormodel.resample_model()
    if (idx+1) % 10 == 0:
        models.append(copy.deepcopy(posteriormodel))
```

Now we can plot our saved samples:

```python
fig = plt.figure()
for idx, model in enumerate(models):
    plt.clf()
    model.plot()
    plt.gcf().suptitle('HDP-HSMM sampled after %d iterations' % (10*(idx+1)))
    plt.savefig('iter_%.3d.png' % (10*(idx+1)))
```

![Sampled models](http://www.mit.edu/~mattjj/github/pyhsmm/posterior_animation.gif)

I generated these data from an HSMM that looked like this:

![Randomly-generated model and data](http://www.mit.edu/~mattjj/github/pyhsmm/truth.png)

So the posterior samples look pretty good!

A convenient shortcut to build a list of sampled models is to write

```python
model_samples = [model.resample_and_copy() for itr in progprint_xrange(150)]
```

That will build a list of model objects (each of which can be inspected,
plotted, pickled, etc, independently) in a way that won't duplicate data that
isn't changed (like the observations or hyperparameter arrays) so that memory
usage is minimized. It also minimizes file size if you save samples like

```python
import cPickle
with open('sampled_models.pickle','w') as outfile:
    cPickle.dump(model_samples,outfile,protocol=-1)
```

## Speed ##

HSMMs constitute a much more powerful model class than plain-old HMMs, and that
enhanced power comes with a computational price: each sampling iteration for an
HSMM is much slower than that of an HMM. But that price is often worthwhile if
you want to place priors on state durations or have the model learn duration
structure present in the data. (In the example, strong duration structure is
what made the inference algorithm latch onto the correct explanation so
easily.) In addition, the increased cost of each iteration often pays for
itself, since HSMM samplers empirically seem to take fewer iterations to
converge than comparable HMM samplers.

Using my nothing-special i7-920 desktop machine and a NumPy/SciPy built against
Intel's MKL along with the Eigen-backed classes, here's how long the demo
iterations took:

```
$ python examples/hsmm.py
.........................  [  25/100,    0.05sec avg,    3.95sec ETA ]
.........................  [  50/100,    0.05sec avg,    2.64sec ETA ]
.........................  [  75/100,    0.05sec avg,    1.34sec ETA ]
.........................  [ 100/100,    0.05sec avg,    0.05sec ETA ]
   0.05sec avg,    5.21sec total
```

## Extending the Code ##
To add your own observation or duration distributions, implement the interfaces
defined in `basic/abstractions.py`. Also see the plugins. To get a flavor of
the style, see [pybasicbayes](https://github.com/mattjj/pybasicbayes).

## References ##
* Matthew J. Johnson and Alan S. Willsky. [Bayesian Nonparametric Hidden
  Semi-Markov Models](http://www.jmlr.org/papers/volume14/johnson13a/johnson13a.pdf).
  Journal of Machine Learning Research (JMLR), 14:673–701, February 2013.

* Matthew J. Johnson and Alan S. Willsky, [The Hierarchical Dirichlet Process
  Hidden Semi-Markov Model](http://www.mit.edu/~mattjj/papers/uai2010.pdf). 26th
  Conference on Uncertainty in Artificial Intelligence (UAI 2010), Avalon,
  California, July 2010.

```bibtex
@article{johnson2013hdphsmm,
    title={Bayesian Nonparametric Hidden Semi-Markov Models},
    author={Johnson, Matthew J. and Willsky, Alan S.},
    journal={Journal of Machine Learning Research},
    pages={673--701},
    volume={14},
    month={February},
    year={2013},
}
```

