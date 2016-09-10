from __future__ import division
import numpy as np
from matplotlib import pyplot as plt
import abc
from util import flattendata, sample_discrete_from_log, combinedata, combinedata_distribution


# PLAN: have a resample_with_truncations method default implementation which
# just samples out the other business using rvs_greater_than
########################################################
#  Distribution base class  #
########################################################
class Distribution(object):
    __metaclass__ = abc.ABCMeta
    @abc.abstractproperty
    def params(self):
        'distribution parameters'
        pass
    @abc.abstractmethod
    def rvs(self,size=[]):
        'random variates (samples)'
        pass
    @abc.abstractmethod
    def log_likelihood(self,x):
        '''
        log likelihood (either log probability mass function or log probability
        density function) of x, which has the same type as the output of rvs()
        '''
        pass
    def __repr__(self):
        return '%s(params={%s})' % (self.__class__.__name__,self._formatparams(self.params))
    @staticmethod
    def _formatparams(dct):
        return ','.join(('{}:{:3.3G}' if isinstance(val,(int,long,float,complex))
                                        else '{}:{}').format(name,val)
                    for name,val in dct.iteritems()).replace('\n','').replace(',',', ')

class DurationDistribution(Distribution):
    __metaclass__ = abc.ABCMeta
    # in addition to the methods required by Distribution, we also require a
    # log_sf implementation
    @abc.abstractmethod
    def log_sf(self,x):
        '''
        log survival function, defined by log_sf(x) = log(P[X \gt x]) =
        log(1-cdf(x)) where cdf(x) = P[X \leq x]
        '''
        pass
    def log_pmf(self,x):
        return self.log_likelihood(x)
    # default implementations below
    def pmf(self,x):
        return np.exp(self.log_pmf(x))
    def rvs_given_greater_than(self,x):
        tail = self.log_sf(x)
        trunc = 500
        while self.log_sf(x+trunc) - tail > -20:
            trunc *= 1.1
        logprobs = self.log_pmf(np.arange(x+1,x+trunc+1)) - tail
        return sample_discrete_from_log(logprobs)+x+1
    def resample_with_truncations(self,data=[],truncated_data=[]):
        '''
        truncated_data is full of observations that were truncated, so this
        method samples them out to be at least that large
        '''
        if not isinstance(truncated_data,list):
            filled_in = np.asarray([self.rvs_given_greater_than(x-1) for x in truncated_data])
        else:
            filled_in = np.asarray([self.rvs_given_greater_than(x-1)
                for xx in truncated_data for x in xx])
        self.resample(data=combinedata((data,filled_in)))
    @property
    def mean(self):
        trunc = 500
        while self.log_sf(trunc) > -20:
            trunc *= 1.5
        return np.arange(1,trunc+1).dot(self.pmf(np.arange(1,trunc+1)))
    def plot(self,data=None,color='b'):
        data = flattendata(data) if data is not None else None
        try:
            tmax = np.where(np.exp(self.log_sf(np.arange(1,1000))) < 1e-3)[0][0]
        except IndexError:
            tmax = 2*self.rvs(1000).mean()
        tmax = max(tmax,data.max()) if data is not None else tmax
        t = np.arange(1,tmax+1)
        plt.plot(t,self.pmf(t),color=color)
        if data is not None:
            if len(data) > 1:
                plt.hist(data,bins=t-0.5,color=color,normed=len(set(data)) > 1)
            else:
                plt.hist(data,bins=t-0.5,color=color)

class BayesianDistribution(Distribution):
    __metaclass__ = abc.ABCMeta
    @abc.abstractproperty
    def hypparams(self):
        'hyperparameters define a prior distribution over parameters'
        pass
    def empirical_bayes(self,data):
        '''
        (optional) set hyperparameters via empirical bayes
        e.g. treat argument as a pseudo-dataset for exponential family
        '''
        raise NotImplementedError
    def __repr__(self):
        if not all(v is None for v in self.hypparams.itervalues()):
            return '%s(\nparams={%s},\nhypparams={%s})' % (self.__class__.__name__,
                    self._formatparams(self.params),self._formatparams(self.hypparams))
        else:
            return super(BayesianDistribution,self).__repr__()


#########################################################
#  Algorithm interfaces for inference in distributions  #
#########################################################
class GibbsSampling(BayesianDistribution):
    __metaclass__ = abc.ABCMeta
    @abc.abstractmethod
    def resample(self,data=[]):
        pass
    def copy_sample(self):
        '''
        return an object copy suitable for making lists of posterior samples
        (override this method to prevent copying shared structures into each sample)
        '''
        return copy.deepcopy(self)
    def resample_and_copy(self):
        self.resample()
        return self.copy_sample()

class MeanField(BayesianDistribution):
    __metaclass__ = abc.ABCMeta
    @abc.abstractmethod
    def expected_log_likelihood(self,x):
        pass
    @abc.abstractmethod
    def meanfieldupdate(self,data,weights):
        pass
    def get_vlb(self):
        raise NotImplementedError

class Collapsed(BayesianDistribution):
    __metaclass__ = abc.ABCMeta
    @abc.abstractmethod
    def log_marginal_likelihood(self,data):
        pass
    def log_predictive(self,newdata,olddata):
        return self.log_marginal_likelihood(combinedata_distribution((newdata,olddata))) \
                    - self.log_marginal_likelihood(olddata)
    def predictive(self,*args,**kwargs):
        return np.exp(self.log_predictive(*args,**kwargs))

class MaxLikelihood(Distribution):
    __metaclass__ = abc.ABCMeta
    @abc.abstractmethod
    def max_likelihood(self,data,weights=None):
        '''
        sets the parameters set to their maximum likelihood values given the
        (weighted) data
        '''
        pass
    @property
    def num_parameters(self):
        raise NotImplementedError

class MAP(BayesianDistribution):
    __metaclass__ = abc.ABCMeta
    @abc.abstractmethod
    def MAP(self,data,weights=None):
        '''
        sets the parameters to their MAP values given the (weighted) data
        analogous to max_likelihood but includes hyperparameters
        '''
        pass


####################################################
#  Models  #
####################################################
# a "model" is differentiated from a "distribution" in this code by latent state
# over data: a model attaches a latent variable (like a label or state sequence)
# to data, and so it 'holds onto' data. Hence the add_data method.
class Model(object):
    __metaclass__ = abc.ABCMeta
    @abc.abstractmethod
    def add_data(self,data):
        pass
    @abc.abstractmethod
    def generate(self,keep=True,**kwargs):
        '''
        Like a distribution's rvs, but this also fills in latent state over
        data and keeps references to the data.
        '''
        pass
    def rvs(self,*args,**kwargs):
        return self.generate(*args,keep=False,**kwargs)[0] # 0th component is data, not latent stuff


##################################################
#  Algorithm interfaces for inference in models  #
##################################################
class ModelGibbsSampling(Model):
    __metaclass__ = abc.ABCMeta
    @abc.abstractmethod
    def resample_model(self): # TODO niter?
        pass
    def copy_sample(self):
        '''
        return an object copy suitable for making lists of posterior samples
        (override this method to prevent copying shared structures into each sample)
        '''
        return copy.deepcopy(self)
    def resample_and_copy(self):
        self.resample_model()
        return self.copy_sample()

class _EMBase(Model):
    __metaclass__ = abc.ABCMeta
    @abc.abstractmethod
    def log_likelihood(self):
        # returns a log likelihood number on attached data
        pass
    def _EM_fit(self,method,tol=1e-1,maxiter=100):
        # NOTE: doesn't re-initialize!
        likes = []
        for itr in xrange(maxiter):
            method()
            likes.append(self.log_likelihood())
            if len(likes) > 1:
                if likes[-1]-likes[-2] < tol:
                    return likes
                elif likes[-1] < likes[-2]:
                    # probably oscillation, do one more
                    method()
                    likes.append(self.log_likelihood())
                    return likes
        print 'WARNING: EM_fit reached maxiter of %d' % maxiter
        return likes

class ModelEM(_EMBase):
    __metaclass__ = abc.ABCMeta
    def EM_fit(self,tol=1e-1,maxiter=100):
        return self._EM_fit(self.EM_step,tol=tol,maxiter=maxiter)
    @abc.abstractmethod
    def EM_step(self):
        pass

class ModelMAPEM(_EMBase):
    __metaclass__ = abc.ABCMeta
    def MAP_EM_fit(self,tol=1e-1,maxiter=100):
        return self._EM_fit(self.MAP_EM_step,tol=tol,maxiter=maxiter)
    @abc.abstractmethod
    def MAP_EM_step(self):
        pass
