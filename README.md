# Nonparametric Bayesian Double Articulation Analyzer

This is a Python implementation for Nonparametric Bayesian Double Articulation Analyzer (NPB-DAA). The NPB-DAA can directly acquire language and acoustic models from observed continuous speech signals.

This generative model is called hiererichel Dirichlet process hidden language model (HDP-HLM), which is obtained by extending the hierarchical Dirichlet process hidden semi-Markov model (HDP-HSMM) proposed by Johnson et al. An inference procedure for the HDP-HLM is derived using the blocked Gibbs sampler originally proposed for the HDP-HSMM.

# Description
・NPB_DAA/README - There is a NPB-DAA tutorial in PDF.(In Japanese. English version is coming soon.)

・NPB_DAA/HDP-HSMM - Python Library for HDP-HSMM. You can get it at [ https://github.com/mattjj/pyhsmm ]. (Please check this VERSION at README)

・NPB_DAA/HDP-HLM - Python code for NPB-DAA

# Requirement

・Ubuntu 12.04.5 LTS

`sudo apt-get install`

・python 2.7.3

・numpy 1.6.1

・matplotlib 1.1.1rc

・scipy 0.9.0

・scikit-learn 0.10

`sudo pip install`

・Paver 1.2.4

・pyzmq==14.4.0/14.5.0/14.6.0 (14.6.0)

・ipython　3.2.1

# Usage
前のバージョンのものですが，勉強会の資料があります．
ディレクトリの構成が変更されていますので，よろしく，おねがいします．


# Troubleshooting
If you are in trouble, please look at this document. You can get information about environment operability confirmed and actions on error occuring often.
  [Troubleshooting document](https://docs.google.com/document/d/1fwcnwNEZNvc1vVZvyRJsMtPdC_FNAtY9S3dyS5CVxZQ/edit?usp=sharing)

# References
・Taniguchi, Tadahiro, Shogo Nagasaka, and Ryo Nakashima. [Nonparametric Bayesian double articulation analyzer for direct language acquisition from continuous speech signals](http://ieeexplore.ieee.org/document/7456220/?arnumber=7456220), 2015.

・Matthew J. Johnson and Alan S. Willsky. [Bayesian Nonparametric Hidden Semi-Markov Models](http://www.jmlr.org/papers/volume14/johnson13a/johnson13a.pdf). Journal of Machine Learning Research (JMLR), 14:673–701, 2013.

# Authors
Tadahiro Taniguch, Ryo Nakashima, Nagasaka Shogo, Tada Yuki, Kaede Hayashi.

## License
* MIT
    * see LICENSE
