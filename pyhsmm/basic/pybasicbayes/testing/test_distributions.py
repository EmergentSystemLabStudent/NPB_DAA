from __future__ import division
import numpy as np

from nose.plugins.attrib import attr

from .. import distributions as distributions
from mixins import BigDataGibbsTester, GewekeGibbsTester

@attr('geometric')
class TestGeometric(BigDataGibbsTester,GewekeGibbsTester):
    @property
    def distribution_class(self):
        return distributions.Geometric

    @property
    def hyperparameter_settings(self):
        return (dict(alpha_0=2,beta_0=20),dict(alpha_0=5,beta_0=5))

    def params_close(self,d1,d2):
        return np.allclose(d1.p,d2.p,rtol=0.05)

    def geweke_statistics(self,d,data):
        return d.p

    @property
    def geweke_pval(self):
        return 0.5

@attr('poisson')
class TestPoisson(BigDataGibbsTester,GewekeGibbsTester):
    @property
    def distribution_class(self):
        return distributions.Poisson

    @property
    def hyperparameter_settings(self):
        return (dict(alpha_0=30,beta_0=3),)

    def params_close(self,d1,d2):
        return np.allclose(d1.lmbda,d2.lmbda,rtol=0.05)

    def geweke_statistics(self,d,data):
        return d.lmbda

@attr('negbinfixedr')
class TestNegativeBinomialFixedR(BigDataGibbsTester,GewekeGibbsTester):
    @property
    def distribution_class(self):
        return distributions.NegativeBinomialFixedR

    @property
    def hyperparameter_settings(self):
        return (dict(r=5,alpha_0=1,beta_0=9),)

    def params_close(self,d1,d2):
        return np.allclose(d1.p,d2.p,rtol=0.1)

    def geweke_statistics(self,d,data):
        return d.p

@attr('negbinintr')
class TestNegativeBinomialIntegerR(BigDataGibbsTester,GewekeGibbsTester):
    @property
    def distribution_class(self):
        return distributions.NegativeBinomialIntegerR

    @property
    def hyperparameter_settings(self):
        return (dict(r_discrete_distn=np.r_[0.,0,0,1,1,1],alpha_0=5,beta_0=5),)

    def params_close(self,d1,d2):
        # since it's easy to be off by 1 in r and still look like the same
        # distribution, best just to check moment parameters
        def mean(d):
            return d.r*d.p/(1.-d.p)
        def var(d):
            return mean(d)/(1.-d.p)
        return np.allclose(mean(d1),mean(d2),rtol=0.1) and np.allclose(var(d1),var(d2),rtol=0.1)

    def geweke_statistics(self,d,data):
        return d.p

    @property
    def geweke_pval(self):
        return 0.005 # since the statistic is on (0,1), it's really sensitive, or something

@attr('negbinintrvariant')
class TestNegativeBinomialIntegerRVariant(TestNegativeBinomialIntegerR):
    @property
    def distribution_class(self):
        return distributions.NegativeBinomialIntegerRVariant

@attr('categorical')
class TestCategorical(BigDataGibbsTester,GewekeGibbsTester):
    @property
    def distribution_class(self):
        return distributions.Categorical

    @property
    def hyperparameter_settings(self):
        return (dict(alpha_0=5.,K=5),)

    @property
    def big_data_size(self):
        return 20000

    def params_close(self,d1,d2):
        return np.allclose(d1.weights,d2.weights,atol=0.05)

    def geweke_statistics(self,d,data):
        return d.weights

    @property
    def geweke_pval(self):
        return 0.05

@attr('gaussian')
class TestGaussian(BigDataGibbsTester,GewekeGibbsTester):
    @property
    def distribution_class(self):
        return distributions.Gaussian

    @property
    def hyperparameter_settings(self):
        return (dict(mu_0=np.zeros(2),sigma_0=np.eye(2),kappa_0=1.,nu_0=4.),)

    def params_close(self,d1,d2):
        return np.linalg.norm(d1.mu-d2.mu) < 0.1 and np.linalg.norm(d1.sigma-d2.sigma) < 0.1

    def geweke_statistics(self,d,data):
        return np.concatenate((d.mu,np.diag(d.sigma)))

    @property
    def geweke_nsamples(self):
        return 30000

    @property
    def geweke_data_size(self):
        return 1

    @property
    def geweke_pval(self):
        return 0.05

    def geweke_numerical_slice(self,setting_idx):
        return slice(0,2)

@attr('diagonalgaussian')
class TestDiagonalGaussian(BigDataGibbsTester,GewekeGibbsTester):
    @property
    def distribution_class(self):
        return distributions.DiagonalGaussian

    @property
    def hyperparameter_settings(self):
        return (dict(mu_0=np.zeros(2),nus_0=7,alphas_0=np.r_[5.,10.],betas_0=np.r_[1.,4.]),)

    def params_close(self,d1,d2):
        return np.linalg.norm(d1.mu-d2.mu) < 0.1 and np.linalg.norm(d1.sigmas-d2.sigmas) < 0.25

    def geweke_statistics(self,d,data):
        return np.concatenate((d.mu,d.sigmas))

    @property
    def geweke_nsamples(self):
        return 50000

    @property
    def geweke_data_size(self):
        return 2

    @property
    def geweke_pval(self):
        return 0.05

    def geweke_numerical_slice(self,setting_idx):
        return slice(0,2)

@attr('CRP')
class TestCRP(BigDataGibbsTester):
    @property
    def distribution_class(self):
        return distributions.CRP

    @property
    def hyperparameter_settings(self):
        return (dict(a_0=1.,b_0=1./10),)

    @property
    def big_data_size(self):
        return [50]*200

    def params_close(self,d1,d2):
        return np.abs(d1.concentration - d2.concentration) < 1.0

@attr('GammaCompoundDirichlet')
class TestDirichletCompoundGamma(object):
    def test_weaklimit(self):
        a = distributions.CRP(10,1)
        b = distributions.GammaCompoundDirichlet(1000,10,1)

        a.concentration = b.concentration = 10.

        from matplotlib import pyplot as plt

        plt.figure()
        crp_counts = np.zeros(10)
        gcd_counts = np.zeros(10)
        for itr in range(500):
            crp_rvs = np.sort(a.rvs(25))[::-1][:10]
            crp_counts[:len(crp_rvs)] += crp_rvs
            gcd_counts += np.sort(b.rvs(25))[::-1][:10]

        plt.plot(crp_counts/200,gcd_counts/200,'bx-')
        plt.xlim(0,10)
        plt.ylim(0,10)

        import os
        figpath = os.path.join(os.path.dirname(__file__),'figures',
                self.__class__.__name__,'weaklimittest.pdf')
        if not os.path.exists(os.path.dirname(figpath)):
            os.mkdir(os.path.dirname(figpath))
        plt.savefig(figpath)

