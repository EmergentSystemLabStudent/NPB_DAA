from __future__ import division
import numpy as np
np.seterr(divide='ignore')
import scipy.stats as stats
import scipy.special as special
import scipy.weave
from abstractions import GibbsSampling, MeanField, MaxLikelihood, MAP
from util import sample_discrete_distribution


######################################################
#  Discrete distribution classes (Clustering)  #
######################################################
class Categorical(GibbsSampling, MeanField, MaxLikelihood, MAP):
    '''
    This class represents a categorical distribution over labels, where the
    parameter is weights and the prior is a Dirichlet distribution.
    For example, if K == 3, then five samples may look like
        [0,1,0,2,1]
    Each entry is the label of a sample, like the outcome of die rolls. In other
    words, generated data or data passed to log_likelihood are indices, not
    indicator variables!  (But when 'weighted data' is passed, like in mean
    field or weighted max likelihood, the weights are over indicator
    variables...)
    This class can be used as a weak limit approximation for a DP, particularly by
    calling __init__ with alpha_0 and K arguments, in which case the prior will be
    a symmetric Dirichlet with K components and parameter alpha_0/K; K is then the
    weak limit approximation parameter.
    Hyperparaemters:
        alphav_0 (vector) OR alpha_0 (scalar) and K
    Parameters:
        weights, a vector encoding a finite pmf
    '''
    def __init__(self,weights=None,alpha_0=None,K=None,alphav_0=None,alpha_mf=None):
        self.K = K
        self.alpha_0 = alpha_0
        self.alphav_0 = alphav_0
        self.weights = weights
        if weights is None and self.alphav_0 is not None:
            self.resample() # intialize from prior
    def _get_alpha_0(self):
        return self._alpha_0
    def _set_alpha_0(self,alpha_0):
        self._alpha_0 = alpha_0
        if None not in (self.K, self._alpha_0):
            self.alphav_0 = np.repeat(self._alpha_0/self.K,self.K)
    alpha_0 = property(_get_alpha_0,_set_alpha_0)
    def _get_alphav_0(self):
        return self._alphav_0 if hasattr(self,'_alphav_0') else None
    def _set_alphav_0(self,alphav_0):
        if alphav_0 is not None:
            self._alphav_0 = alphav_0
            self.K = len(alphav_0)
    alphav_0 = property(_get_alphav_0,_set_alphav_0)
    @property
    def params(self):
        return dict(weights=self.weights)
    @property
    def hypparams(self):
        return dict(alphav_0=self.alphav_0)
    @property
    def num_parameters(self):
        return len(self.weights)
    def rvs(self,size=None):
        return sample_discrete_distribution(self.weights,size)
    def log_likelihood(self,x):
        return np.log(self.weights)[x]
    def _posterior_hypparams(self,counts):
        return self.alphav_0 + counts
    ### Gibbs sampling
    def resample(self,data=[]):
        'data is an array of indices (i.e. labels) or a list of such arrays'
        hypparams = self._posterior_hypparams(*self._get_statistics(data,len(self.alphav_0)))
        self.weights = np.random.dirichlet(np.where(hypparams>1e-2,hypparams,1e-2))
        self._alpha_mf = self.weights * self.alphav_0.sum()
        return self
    @staticmethod
    def _get_statistics(data,K):
        if isinstance(data,np.ndarray):
            counts = np.bincount(data,minlength=K)
        else:
            counts = sum(np.bincount(d,minlength=K) for d in data)
        return counts,
    ### Mean Field
    def meanfieldupdate(self,data,weights):
        # update
        self._alpha_mf = self._posterior_hypparams(*self._get_weighted_statistics(data,weights))
        self.weights = self._alpha_mf / self._alpha_mf.sum() # for plotting
        return self
    def get_vlb(self):
        # return avg energy plus entropy, our contribution to the vlb
        # see Eq. 10.66 in Bishop
        logpitilde = self.expected_log_likelihood() # default is on np.arange(self.K)
        q_entropy = -1* ((logpitilde*(self._alpha_mf-1)).sum() \
                + special.gammaln(self._alpha_mf.sum()) - special.gammaln(self._alpha_mf).sum())
        p_avgengy = special.gammaln(self.alphav_0.sum()) - special.gammaln(self.alphav_0).sum() \
                + ((self.alphav_0-1)*logpitilde).sum()
        return p_avgengy + q_entropy
    def expected_log_likelihood(self,x=None):
        # usually called when np.all(x == np.arange(self.K))
        x = x if x is not None else slice(None)
        return special.digamma(self._alpha_mf[x]) - special.digamma(self._alpha_mf.sum())
    @staticmethod
    def _get_weighted_statistics(data,weights):
        # data is just a placeholder; technically it should always be
        # np.arange(K)[na,:].repeat(N,axis=0), but this code ignores it
        if isinstance(weights,np.ndarray):
            counts = np.atleast_2d(weights).sum(0)
        else:
            counts = sum(np.atleast_2d(w).sum(0) for w in weights)
        return counts,
    ### Max likelihood
    def max_likelihood(self,data,weights=None):
        K = self.K
        if weights is None:
            counts, = self._get_statistics(data,K)
        else:
            counts, = self._get_weighted_statistics(data,weights)
        self.weights = counts/counts.sum()
        return self
    def MAP(self,data,weights=None):
        K = self.K
        if weights is None:
            counts, = self._get_statistics(data,K)
        else:
            counts, = self._get_weighted_statistics(data,weights)
        self.weights = counts/counts.sum()
        return self

class CRP(GibbsSampling):
    '''
    concentration ~ Gamma(a_0,b_0) [b_0 is inverse scale, inverse of numpy scale arg]
    rvs ~ CRP(concentration)
    This class models CRPs. The parameter is the concentration parameter (proportional
    to probability of starting a new table given some number of customers in the
    restaurant), which has a Gamma prior.
    '''
    def __init__(self,a_0,b_0,concentration=None):
        self.a_0 = a_0
        self.b_0 = b_0
        if concentration is not None:
            self.concentration = concentration
        else:
            self.resample(niter=1)
    @property
    def params(self):
        return dict(concentration=self.concentration)
    @property
    def hypparams(self):
        return dict(a_0=self.a_0,b_0=self.b_0)
    def rvs(self,customer_counts):
        # could replace this with one of the faster C versions I have lying
        # around, but at least the Python version is clearer
        assert isinstance(customer_counts,list) or isinstance(customer_counts,int)
        if isinstance(customer_counts,int):
            customer_counts = [customer_counts]
        restaurants = []
        for num in customer_counts:
            # a CRP with num customers
            tables = []
            for c in range(num):
                newidx = sample_discrete_distribution(np.array(tables + [self.concentration]))
                if newidx == len(tables):
                    tables += [1]
                else:
                    tables[newidx] += 1
            restaurants.append(tables)
        return restaurants if len(restaurants) > 1 else restaurants[0]
    def log_likelihood(self,restaurants):
        assert isinstance(restaurants,list) and len(restaurants) > 0
        if not isinstance(restaurants[0],list): restaurants=[restaurants]
        likes = []
        for counts in restaurants:
            counts = np.array([c for c in counts if c > 0])    # remove zero counts b/c of gammaln
            K = len(counts) # number of tables
            N = sum(counts) # number of customers
            likes.append(K*np.log(self.concentration) + np.sum(special.gammaln(counts)) +
                            special.gammaln(self.concentration) -
                            special.gammaln(N+self.concentration))
        return np.asarray(likes) if len(likes) > 1 else likes[0]
    def resample(self,data=[],niter=25):
        for itr in range(niter):
            a_n, b_n = self._posterior_hypparams(*self._get_statistics(data))
            self.concentration = np.random.gamma(a_n,scale=1./b_n)
    def _posterior_hypparams(self,sample_numbers,total_num_distinct):
        # NOTE: this is a stochastic function: it samples auxiliary variables
        if total_num_distinct > 0:
            sample_numbers = np.array(sample_numbers)
            sample_numbers = sample_numbers[sample_numbers > 0]
            wvec = np.random.beta(self.concentration+1,sample_numbers)
            svec = np.array(stats.bernoulli.rvs(sample_numbers/(sample_numbers+self.concentration)))
            return self.a_0 + total_num_distinct-svec.sum(), (self.b_0 - np.log(wvec).sum())
        else:
            return self.a_0, self.b_0
        return self
    def _get_statistics(self,data):
        assert isinstance(data,list)
        if len(data) == 0:
            sample_numbers = 0
            total_num_distinct = 0
        else:
            if isinstance(data[0],list):
                sample_numbers = np.array(map(sum,data))
                total_num_distinct = sum(map(len,data))
            else:
                sample_numbers = np.array(sum(data))
                total_num_distinct = len(data)
        return sample_numbers, total_num_distinct

class GammaCompoundDirichlet(CRP):
    # TODO this class is a bit ugly
    '''
    Implements a Gamma(a_0,b_0) prior over finite dirichlet concentration
    parameter. The concentration is scaled according to the weak-limit sequence.
    For each set of counts i, the model is
        concentration ~ Gamma(a_0,b_0)
        pi_i ~ Dir(concentration/K)
        data_i ~ Multinomial(pi_i)
    K is a free parameter in that with big enough K (relative to the size of the
    sampled data) everything starts to act like a DP; K is just the size of the
    size of the mesh projection.
    '''
    def __init__(self,K,a_0,b_0,concentration=None):
        self.K = K
        super(GammaCompoundDirichlet,self).__init__(a_0=a_0,b_0=b_0,
                concentration=concentration)
    @property
    def params(self):
        return dict(concentration=self.concentration)
    @property
    def hypparams(self):
        return dict(a_0=self.a_0,b_0=self.b_0,K=self.K)
    def rvs(self,sample_counts):
        if isinstance(sample_counts,int):
            sample_counts = [sample_counts]
        out = np.empty((len(sample_counts),self.K),dtype=int)
        for idx,c in enumerate(sample_counts):
            out[idx] = np.random.multinomial(c,
                np.random.dirichlet(np.repeat(self.concentration/self.K,self.K)))
        return out if out.shape[0] > 1 else out[0]
    def resample(self,data=[],niter=25,weighted_cols=None):
        if weighted_cols is not None:
            self.weighted_cols = weighted_cols
        else:
            self.weighted_cols = np.ones(self.K)
        # all this is to check if data is empty
        if isinstance(data,np.ndarray):
            size = data.sum()
        elif isinstance(data,list):
            size = sum(d.sum() for d in data)
        else:
            assert data == 0
            size = 0
        if size > 0:
            return super(GammaCompoundDirichlet,self).resample(data,niter=niter)
        else:
            return super(GammaCompoundDirichlet,self).resample(data,niter=1)
    def _get_statistics(self,data):
        # NOTE: this is a stochastic function: it samples auxiliary variables
        counts = np.array(data,ndmin=2)
        # sample m's, which sample an inverse of the weak limit projection
        if counts.sum() == 0:
            return 0, 0
        else:
            msum = np.array(0.)
            weighted_cols = self.weighted_cols
            concentration = self.concentration
            N,K = counts.shape
            scipy.weave.inline(
                    '''
                    int tot = 0;
                    for (int i=0; i < N; i++) {
                        for (int j=0; j < K; j++) {
                            for (int c=0; c < counts[i*K + j]; c++) {
                                tot += ((float) rand()) / RAND_MAX <
                                    ((float) concentration/K*weighted_cols[j]) /
                                            (c + concentration/K*weighted_cols[j]);
                            }
                        }
                    }
                    *msum = tot;
                    ''',
                    ['weighted_cols','concentration','N','K','msum','counts'],
                    extra_compile_args=['-O3'])
            return counts.sum(1), int(msum)
    def _get_statistics_python(self,data):
        counts = np.array(data,ndmin=2)
        # sample m's
        if counts.sum() == 0:
            return 0, 0
        else:
            m = 0
            for (i,j), n in np.ndenumerate(counts):
                m += (np.random.rand(n) < self.concentration*self.K*self.weighted_cols[j] \
                        / (np.arange(n)+self.concentration*self.K*self.weighted_cols[j])).sum()
            return counts.sum(1), m
