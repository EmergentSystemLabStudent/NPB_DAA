from __future__ import division
import numpy as np
np.seterr(divide='ignore')
from numpy import newaxis as na
from numpy.core.umath_tests import inner1d
import scipy.stats as stats
import scipy.special as special
import scipy.linalg
import matplotlib.pyplot as plt
import copy
from abstractions import GibbsSampling, MeanField, Collapsed, MaxLikelihood, MAP, DurationDistribution
from util import sample_niw, invwishart_entropy, invwishart_log_partitionfunction, getdatasize, flattendata_distribution, getdatadimension, combinedata_distribution, multivariate_t_loglik


##############################################
#  Mixins for making duratino distributions  #
##############################################
class _StartAtOneMixin(object):
    def log_likelihood(self,x,*args,**kwargs):
        return super(_StartAtOneMixin,self).log_likelihood(x-1,*args,**kwargs)
    def log_sf(self,x,*args,**kwargs):
        return super(_StartAtOneMixin,self).log_sf(x-1,*args,**kwargs)
    def rvs(self,size=None):
        return super(_StartAtOneMixin,self).rvs(size)+1
    def rvs_given_greater_than(self,x):
        return super(_StartAtOneMixin,self).rvs_given_greater_than(x-1)+1
    def resample(self,data=[],*args,**kwargs):
        if isinstance(data,np.ndarray):
            return super(_StartAtOneMixin,self).resample(data-1,*args,**kwargs)
        else:
            return super(_StartAtOneMixin,self).resample([d-1 for d in data],*args,**kwargs)
    def max_likelihood(self,data,weights=None,*args,**kwargs):
        if weights is not None:
            raise NotImplementedError
        else:
            if isinstance(data,np.ndarray):
                return super(_StartAtOneMixin,self).max_likelihood(data-1,weights=None,*args,**kwargs)
            else:
                return super(_StartAtOneMixin,self).max_likelihood([d-1 for d in data],weights=None,*args,**kwargs)


##########################################################
#  Multivariate Gaussian distribution classes  #
##########################################################
class _GaussianBase(object):
    @property
    def params(self):
        return dict(mu=self.mu,sigma=self.sigma)
    ### internals
    def getsigma(self):
        return self._sigma
    def setsigma(self,sigma):
        self._sigma = sigma
        self._sigma_chol = None
    sigma = property(getsigma,setsigma)
    @property
    def sigma_chol(self):
        if self._sigma_chol is None:
            self._sigma_chol = np.linalg.cholesky(self._sigma)
        return self._sigma_chol
    ### distribution stuff
    def rvs(self,size=None):
        size = 1 if size is None else size
        size = size + (self.mu.shape[0],) if isinstance(size,tuple) else (size,self.mu.shape[0])
        return self.mu + np.random.normal(size=size).dot(self.sigma_chol.T)
    def log_likelihood(self,x):
        mu, sigma, D = self.mu, self.sigma, self.mu.shape[0]
        sigma_chol = self.sigma_chol
        bads = np.isnan(np.atleast_2d(x)).any(axis=1)
        x = np.nan_to_num(x).reshape((-1,D)) - mu
        xs = scipy.linalg.solve_triangular(sigma_chol,x.T,lower=True)
        out = -1./2. * inner1d(xs.T,xs.T) - D/2*np.log(2*np.pi) \
                - np.log(sigma_chol.diagonal()).sum()
        out[bads] = 0
        return out
    ### plotting
    def plot(self,data=None,indices=None,color='b',plot_params=True,label=''):
        from util import project_data, plot_gaussian_projection, plot_gaussian_2D
        if data is not None:
            data = flattendata_distribution(data)
        D = self.mu.shape[0]
        if D > 2 and ((not hasattr(self,'plotting_subspace_basis'))
                or (self.plotting_subspace_basis.shape[1] != D)):
            # TODO improve this bookkeeping. need a notion of collection. it's
            # totally potentially broken and confusing to set class members like
            # this!
            subspace = np.random.randn(D,2)
            self.__class__.plotting_subspace_basis = np.linalg.qr(subspace)[0].T.copy()
        if data is not None:
            if D > 2:
                data = project_data(data,self.plotting_subspace_basis)
            plt.plot(data[:,0],data[:,1],marker='.',linestyle=' ',color=color)
        if plot_params:
            if D > 2:
                plot_gaussian_projection(self.mu,self.sigma,self.plotting_subspace_basis,
                        color=color,label=label)
            else:
                plot_gaussian_2D(self.mu,self.sigma,color=color,label=label)
    def to_json_dict(self):
        D = self.mu.shape[0]
        assert D == 2
        U,s,_ = np.linalg.svd(self.sigma)
        U /= np.linalg.det(U)
        theta = np.arctan2(U[0,0],U[0,1])*180/np.pi
        return {'x':self.mu[0],'y':self.mu[1],'rx':np.sqrt(s[0]),'ry':np.sqrt(s[1]),
                'theta':theta}

class Gaussian(_GaussianBase, GibbsSampling, MeanField, Collapsed, MAP, MaxLikelihood):
    '''
    Multivariate Gaussian distribution class.
    NOTE: Only works for 2 or more dimensions. For a scalar Gaussian, use one of
    the scalar classes.  Uses a conjugate Normal/Inverse-Wishart prior.
    Hyperparameters mostly follow Gelman et al.'s notation in Bayesian Data
    Analysis, except sigma_0 is proportional to expected covariance matrix:
        nu_0, sigma_0
        mu_0, kappa_0
    Parameters are mean and covariance matrix:
        mu, sigma
    '''
    def __init__(self,mu=None,sigma=None,
            mu_0=None,sigma_0=None,kappa_0=None,nu_0=None,
            kappa_mf=None,nu_mf=None):
        self.mu    = mu
        self.sigma = sigma
        self.mu_0    = mu_0
        self.sigma_0 = sigma_0
        self.kappa_0 = kappa_0
        self.nu_0    = nu_0
        self.kappa_mf = kappa_mf if kappa_mf is not None else kappa_0
        self.nu_mf    = nu_mf if nu_mf is not None else nu_0
        self.mu_mf    = mu
        self.sigma_mf = sigma
        if (mu,sigma) == (None,None) and None not in (mu_0,sigma_0,kappa_0,nu_0):
            self.resample() # initialize from prior
    @property
    def hypparams(self):
        return dict(mu_0=self.mu_0,sigma_0=self.sigma_0,kappa_0=self.kappa_0,nu_0=self.nu_0)
    @property
    def num_parameters(self):
        D = len(self.mu)
        return D*(D+1)/2
    @staticmethod
    def _get_statistics(data,D=None):
        n = getdatasize(data)
        if n > 0:
            D = getdatadimension(data) if D is None else D
            if isinstance(data,np.ndarray):
                xbar = np.reshape(data,(-1,D)).mean(0)
                centered = data - xbar
                sumsq = np.dot(centered.T,centered)
            else:
                xbar = sum(np.reshape(d,(-1,D)).sum(0) for d in data) / n
                sumsq = sum(np.dot((np.reshape(d,(-1,D))-xbar).T,(np.reshape(d,(-1,D))-xbar))
                        for d in data)
        else:
            xbar, sumsq = None, None
        return n, xbar, sumsq
    @staticmethod
    def _get_weighted_statistics(data,weights,D=None):
        # NOTE: _get_statistics is special case with all weights being 1
        # this is kept as a separate method for speed and modularity
        if isinstance(data,np.ndarray):
            neff = weights.sum()
            if neff > 0:
                D = getdatadimension(data) if D is None else D
                xbar = np.dot(weights,np.reshape(data,(-1,D))) / neff
                centered = np.reshape(data,(-1,D)) - xbar
                sumsq = np.dot(centered.T,(weights[:,na] * centered))
            else:
                xbar, sumsq = None, None
        else:
            neff = sum(w.sum() for w in weights)
            if neff > 0:
                D = getdatadimension(data) if D is None else D
                xbar = sum(np.dot(w,np.reshape(d,(-1,D))) for w,d in zip(weights,data)) / neff
                sumsq = sum(np.dot((np.reshape(d,(-1,D))-xbar).T,w[:,na]*(np.reshape(d,(-1,D))-xbar))
                        for w,d in zip(weights,data))
            else:
                xbar, sumsq = None, None
        return neff, xbar, sumsq
    def _posterior_hypparams(self,n,xbar,sumsq):
        mu_0, sigma_0, kappa_0, nu_0 = self.mu_0, self.sigma_0, self.kappa_0, self.nu_0
        if n > 0:
            mu_n = self.kappa_0 / (self.kappa_0 + n) * self.mu_0 + n / (self.kappa_0 + n) * xbar
            kappa_n = self.kappa_0 + n
            nu_n = self.nu_0 + n
            sigma_n = self.sigma_0 + sumsq + \
                    self.kappa_0*n/(self.kappa_0+n) * np.outer(xbar-self.mu_0,xbar-self.mu_0)
            return mu_n, sigma_n, kappa_n, nu_n
        else:
            return mu_0, sigma_0, kappa_0, nu_0
    def empirical_bayes(self,data):
        D = getdatadimension(data)
        self.kappa_0 = 0
        self.nu_0 = 0
        self.mu_0 = np.zeros(D)
        self.sigma_0 = np.zeros((D,D))
        self.mu_0, self.sigma_0, self.kappa_0, self.nu_0 = \
                self._posterior_hypparams(*self._get_statistics(data))
        if (self.mu,self.sigma) == (None,None):
            self.resample() # intialize from prior
        return self
    ### Gibbs sampling
    def resample(self,data=[]):
        D = len(self.mu_0)
        self.mu_mf, self.sigma_mf = self.mu, self.sigma = \
                sample_niw(*self._posterior_hypparams(*self._get_statistics(data,D)))
        return self
    def copy_sample(self):
        new = copy.copy(self)
        new.mu = self.mu.copy()
        new.sigma = self.sigma.copy()
        return new
    ### Mean Field
    # NOTE my sumsq is Bishop's Nk*Sk
    def _get_sigma_mf(self):
        return self._sigma_mf
    def _set_sigma_mf(self,val):
        self._sigma_mf = val
        self._sigma_mf_chol = None
    sigma_mf = property(_get_sigma_mf,_set_sigma_mf)
    @property
    def sigma_mf_chol(self):
        if self._sigma_mf_chol is None:
            self._sigma_mf_chol = np.linalg.cholesky(self.sigma_mf)
        return self._sigma_mf_chol
    def meanfieldupdate(self,data,weights):
        # update
        D = len(self.mu_0)
        self.mu_mf, self.sigma_mf, self.kappa_mf, self.nu_mf = \
                self._posterior_hypparams(*self._get_weighted_statistics(data,weights,D))
        self.mu, self.sigma = self.mu_mf, self.sigma_mf/(self.nu_mf - D - 1) # for plotting
    def get_vlb(self):
        # return avg energy plus entropy, our contribution to the mean field
        # variational lower bound
        D = len(self.mu_0)
        loglmbdatilde = self._loglmbdatilde()
        # see Eq. 10.77 in Bishop
        q_entropy = -0.5 * (loglmbdatilde + D * (np.log(self.kappa_mf/(2*np.pi))-1)) \
                + invwishart_entropy(self.sigma_mf,self.nu_mf)
        # see Eq. 10.74 in Bishop, we aren't summing over K
        p_avgengy = 0.5 * (D * np.log(self.kappa_0/(2*np.pi)) + loglmbdatilde \
                - D*self.kappa_0/self.kappa_mf - self.kappa_0*self.nu_mf*\
                np.dot(self.mu_mf -
                    self.mu_0,np.linalg.solve(self.sigma_mf,self.mu_mf - self.mu_0))) \
                + invwishart_log_partitionfunction(self.sigma_0,self.nu_0) \
                + (self.nu_0 - D - 1)/2*loglmbdatilde - 1/2*self.nu_mf*\
                np.linalg.solve(self.sigma_mf,self.sigma_0).trace()
        return p_avgengy + q_entropy
    def expected_log_likelihood(self,x):
        mu_n, sigma_n, kappa_n, nu_n = self.mu_mf, self.sigma_mf, self.kappa_mf, self.nu_mf
        D = len(mu_n)
        x = np.reshape(x,(-1,D)) - mu_n # x is now centered
        xs = np.linalg.solve(self.sigma_mf_chol,x.T)
        # see Eqs. 10.64, 10.67, and 10.71 in Bishop
        return self._loglmbdatilde()/2 - D/(2*kappa_n) - nu_n/2 * \
                inner1d(xs.T,xs.T) - D/2*np.log(2*np.pi)
    def _loglmbdatilde(self):
        # see Eq. 10.65 in Bishop
        D = len(self.mu_0)
        chol = self.sigma_mf_chol
        return special.digamma((self.nu_mf-np.arange(D))/2).sum() \
                + D*np.log(2) - 2*np.log(chol.diagonal()).sum()
    ### Collapsed
    def log_marginal_likelihood(self,data):
        n, D = getdatasize(data), len(self.mu_0)
        return self._log_partition_function(*self._posterior_hypparams(*self._get_statistics(data))) \
                - self._log_partition_function(self.mu_0,self.sigma_0,self.kappa_0,self.nu_0) \
                - n*D/2 * np.log(2*np.pi)
    def _log_partition_function(self,mu,sigma,kappa,nu):
        D = len(mu)
        chol = np.linalg.cholesky(sigma)
        return nu*D/2*np.log(2) + special.multigammaln(nu/2,D) + D/2*np.log(2*np.pi/kappa) \
                - nu*np.log(chol.diagonal()).sum()
    def log_predictive_studentt_datapoints(self,datapoints,olddata):
        D = len(self.mu_0)
        mu_n, sigma_n, kappa_n, nu_n = self._posterior_hypparams(*self._get_statistics(olddata,D))
        return multivariate_t_loglik(datapoints,nu_n-D+1,mu_n,(kappa_n+1)/(kappa_n*(nu_n-D+1))*sigma_n)
    def log_predictive_studentt(self,newdata,olddata):
        # an alternative computation to the generic log_predictive, which is implemented
        # in terms of log_marginal_likelihood. mostly for testing, I think
        newdata = np.atleast_2d(newdata)
        return sum(self.log_predictive_studentt_datapoints(d,combinedata_distribution((olddata,newdata[:i])))[0]
                        for i,d in enumerate(newdata))
    ### Max likelihood
    # NOTE: could also use sumsq/(n-1) as the covariance estimate, which would
    # be unbiased but not max likelihood, but if we're in the regime where that
    # matters we've got bigger problems!
    def max_likelihood(self,data,weights=None):
        D = getdatadimension(data)
        if weights is None:
            n, muhat, sumsq = self._get_statistics(data)
        else:
            n, muhat, sumsq = self._get_weighted_statistics(data,weights)
        # this SVD is necessary to check if the max likelihood solution is
        # degenerate, which can happen in the EM algorithm
        if n < D or (np.linalg.svd(sumsq,compute_uv=False) > 1e-6).sum() < D:
            # broken!
            self.mu = 99999999*np.ones(D)
            self.sigma = np.eye(D)
            self.broken = True
        else:
            self.mu = muhat
            self.sigma = sumsq/n
        return self
    def MAP(self,data,weights=None):
        # max likelihood with prior pseudocounts included in data
        if weights is None:
            n, muhat, sumsq = self._get_statistics(data)
        else:
            n, muhat, sumsq = self._get_weighted_statistics(data,weights)

        self.mu, self.sigma, _, _ = self._posterior_hypparams(n,muhat,sumsq)
        return self


##########################################################
#  Scalar Gaussian distribution classes  #
##########################################################
class _ScalarGaussianBase(object):
    @property
    def params(self):
        return dict(mu=self.mu,sigmasq=self.sigmasq)
    def rvs(self,size=None):
        return np.sqrt(self.sigmasq)*np.random.normal(size=size)+self.mu
    def log_likelihood(self,x):
        x = np.reshape(x,(-1,1))
        return (-0.5*(x-self.mu)**2/self.sigmasq - np.log(np.sqrt(2*np.pi*self.sigmasq))).ravel()
    def __repr__(self):
        return self.__class__.__name__ + '(mu=%f,sigmasq=%f)' % (self.mu,self.sigmasq)
    def plot(self,data=None,indices=None,color='b',plot_params=True,label=None):
        data = np.concatenate(data) if data is not None else None
        indices = np.concatenate(indices) if indices is not None else None
        if data is not None:
            assert indices is not None
            plt.plot(indices,data,color=color,marker='x',linestyle='')
        if plot_params:
            assert indices is not None
            if len(indices) > 1:
                from util import rle
                vals, lens = rle(np.diff(indices))
                starts = np.concatenate(((0,),lens.cumsum()[:-1]))
                for start, blocklen in zip(starts[vals == 1], lens[vals == 1]):
                    plt.plot(indices[start:start+blocklen],
                            np.repeat(self.mu,blocklen),color=color,linestyle='--')
            else:
                plt.plot(indices,[self.mu],color=color,marker='+')

# TODO meanfield, max_likelihood
class ScalarGaussianNIX(_ScalarGaussianBase, GibbsSampling, Collapsed):
    '''
    Conjugate Normal-(Scaled-)Inverse-ChiSquared prior. (Another parameterization is the
    Normal-Inverse-Gamma.)
    '''
    def __init__(self,mu=None,sigmasq=None,mu_0=None,kappa_0=None,sigmasq_0=None,nu_0=None):
        self.mu = mu
        self.sigmasq = sigmasq
        self.mu_0 = mu_0
        self.kappa_0 = kappa_0
        self.sigmasq_0 = sigmasq_0
        self.nu_0 = nu_0
        if (mu,sigmasq) == (None,None) and None not in (mu_0,kappa_0,sigmasq_0,nu_0):
            self.resample() # intialize from prior
    @property
    def hypparams(self):
        return dict(mu_0=self.mu_0,kappa_0=self.kappa_0,
                sigmasq_0=self.sigmasq_0,nu_0=self.nu_0)
    def _posterior_hypparams(self,n,ybar,sumsqc):
        mu_0, kappa_0, sigmasq_0, nu_0 = self.mu_0, self.kappa_0, self.sigmasq_0, self.nu_0
        if n > 0:
            kappa_n = kappa_0 + n
            mu_n = (kappa_0 * mu_0 + n * ybar) / kappa_n
            nu_n = nu_0 + n
            sigmasq_n = 1/nu_n * (nu_0 * sigmasq_0 + sumsqc + kappa_0 * n / (kappa_0 + n) * (ybar - mu_0)**2)
            return mu_n, kappa_n, sigmasq_n, nu_n
        else:
            return mu_0, kappa_0, sigmasq_0, nu_0
    ### Gibbs sampling
    def resample(self,data=[]):
        mu_n, kappa_n, sigmasq_n, nu_n = self._posterior_hypparams(*self._get_statistics(data))
        self.sigmasq = nu_n * sigmasq_n / np.random.chisquare(nu_n)
        self.mu = np.sqrt(self.sigmasq / kappa_n) * np.random.randn() + mu_n
        return self
    def _get_statistics(self,data):
        assert isinstance(data,np.ndarray) or \
                (isinstance(data,list) and all((isinstance(d,np.ndarray))
                    for d in data)) or \
                (isinstance(data,int) or isinstance(data,float))
        n = getdatasize(data)
        if n > 0:
            if isinstance(data,np.ndarray):
                ybar = data.mean()
                sumsqc = ((data-ybar)**2).sum()
            elif isinstance(data,list):
                ybar = sum(d.sum() for d in data)/n
                sumsqc = sum(np.sum((d-ybar)**2) for d in data)
            else:
                ybar = data
                sumsqc = 0
        else:
            ybar = None
            sumsqc = None
        return n, ybar, sumsqc
    ### Collapsed
    def log_marginal_likelihood(self,data):
        n = getdatasize(data)
        mu_0, kappa_0, sigmasq_0, nu_0 = self.mu_0, self.kappa_0, self.sigmasq_0, self.nu_0
        mu_n, kappa_n, sigmasq_n, nu_n = self._posterior_hypparams(*self._get_statistics(data))
        return special.gammaln(nu_n/2) - special.gammaln(nu_0/2) \
                + 0.5*(np.log(kappa_0) - np.log(kappa_n) \
                       + nu_0 * (np.log(nu_0) + np.log(sigmasq_0)) \
                         - nu_n * (np.log(nu_n) + np.log(sigmasq_n)) \
                       - n*np.log(np.pi))
    def log_predictive_single(self,y,olddata):
        # mostly for testing or speed
        mu_n, kappa_n, sigmasq_n, nu_n = self._posterior_hypparams(*self._get_statistics(olddata))
        return stats.t.logpdf(y,nu_n,loc=mu_n,scale=np.sqrt((1+kappa_n)*sigmasq_n/kappa_n))


##########################################################
#  Poisson distribution classes  #
##########################################################
class Poisson(GibbsSampling, Collapsed):
    '''
    Poisson distribution with a conjugate Gamma prior.
    NOTE: the support is {0,1,2,...}
    Hyperparameters (following Wikipedia's notation):
        alpha_0, beta_0
    Parameter is the mean/variance parameter:
        lmbda
    '''
    def __init__(self,lmbda=None,alpha_0=None,beta_0=None):
        self.lmbda = lmbda
        self.alpha_0 = alpha_0
        self.beta_0 = beta_0
        if lmbda is None and None not in (alpha_0,beta_0):
            self.resample() # intialize from prior
    @property
    def params(self):
        return dict(lmbda=self.lmbda)
    @property
    def hypparams(self):
        return dict(alpha_0=self.alpha_0,beta_0=self.beta_0)
    def log_sf(self,x):
        return stats.poisson.logsf(x,self.lmbda)
    def _posterior_hypparams(self,n,tot):
        return self.alpha_0 + tot, self.beta_0 + n
    def rvs(self,size=None):
        return np.random.poisson(self.lmbda,size=size)
    def log_likelihood(self,x):
        lmbda = self.lmbda
        x = np.array(x,ndmin=1)
        raw = np.empty(x.shape)
        raw[x>=0] = -lmbda + x[x>=0]*np.log(lmbda) - special.gammaln(x[x>=0]+1)
        raw[x<0] = -np.inf
        return raw if isinstance(x,np.ndarray) else raw[0]
    ### Gibbs Sampling
    def resample(self,data=[]):
        alpha_n, beta_n = self._posterior_hypparams(*self._get_statistics(data))
        self.lmbda = np.random.gamma(alpha_n,1/beta_n)
        return self
    def _get_statistics(self,data):
        if isinstance(data,np.ndarray):
            n = data.shape[0]
            tot = data.sum()
        elif isinstance(data,list):
            n = sum(d.shape[0] for d in data)
            tot = sum(d.sum() for d in data)
        else:
            assert isinstance(data,int)
            n = 1
            tot = data
        return n, tot
    def _get_weighted_statistics(self,data,weights):
        pass # TODO
    ### Collapsed
    def log_marginal_likelihood(self,data):
        return self._log_partition_function(*self._posterior_hypparams(*self._get_statistics(data))) \
                - self._log_partition_function(self.alpha_0,self.beta_0) \
                - self._get_sum_of_gammas(data)
    def _log_partition_function(self,alpha,beta):
        return special.gammaln(alpha) - alpha * np.log(beta)
    def _get_sum_of_gammas(self,data):
        if isinstance(data,np.ndarray):
            return special.gammaln(data+1).sum()
        elif isinstance(data,list):
            return sum(special.gammaln(d+1).sum() for d in data)
        else:
            assert isinstance(data,int)
            return special.gammaln(data+1)
    ### Max likelihood
    def max_likelihood(self,data,weights=None):
        if weights is None:
            n, tot = self._get_statistics(data)
        else:
            n, tot = self._get_weighted_statistics(data,weights)
        self.lmbda = tot/n

class PoissonDuration(_StartAtOneMixin,Poisson,DurationDistribution):
    pass
