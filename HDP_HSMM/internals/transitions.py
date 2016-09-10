from __future__ import division
import numpy as np
from numpy import newaxis as na
np.seterr(invalid='raise')
import scipy.stats as stats
import scipy.weave
import operator, copy
from ..basic.clustering import GammaCompoundDirichlet
from ..basic.util import rle


##################################################
#  Misc  #
##################################################
# TODO scaling by self.state_dim in concresampling is the confusing result of
# having a DirGamma object and not a WLDPGamma object! make one
# TODO reuse Multinomial/Categorical code
# TODO change concentrationresampling from mixin to metaprogramming?
# TODO add model ref, change trans_counts to cached property
class ConcentrationResampling(object):
    def __init__(self,state_dim,alpha_a_0,alpha_b_0,gamma_a_0,gamma_b_0):
        self.gamma_obj = GammaCompoundDirichlet(state_dim,gamma_a_0,gamma_b_0)
        self.alpha_obj = GammaCompoundDirichlet(state_dim,alpha_a_0,alpha_b_0)
    def resample(self):
        # multiply by state_dim because the trans objects divide by it (since
        # their parameters correspond to the DP parameters, and so they convert
        # into weak limit scaling)
        self.alpha_obj.resample(self.trans_counts,weighted_cols=self.beta)
        self.alpha = self.alpha_obj.concentration
        self.gamma_obj.resample(self.m)
        self.gamma = self.gamma_obj.concentration


##############################################################
#  HDP-HMM classes  #
##############################################################
class HDPHMMTransitions(object):
    def __init__(self,state_dim,alpha,gamma,beta=None,A=None):
        self.state_dim = state_dim
        self.alpha = alpha
        self.gamma = gamma
        if A is None or beta is None:
            self.resample()
        else:
            self.A = A
            self.beta = beta
    ### Gibbs sampling
    def resample(self,states_list=[]):
        trans_counts = self._count_transitions(states_list)
        m = self._get_m(trans_counts)
        self._resample_beta(m)
        self._resample_A(trans_counts)
    def copy_sample(self):
        new = copy.deepcopy(self)
        if hasattr(new,'trans_counts'):
            del new.trans_counts
        if hasattr(new,'m'):
            del new.m
        return new
    def _resample_beta(self,m):
        self.beta = np.random.dirichlet(self.gamma / self.state_dim + m.sum(0) + 1e-2)
    def _resample_A(self,trans_counts):
        self.A = stats.gamma.rvs(self.alpha * self.beta + trans_counts + 1e-2)
        self.A /= self.A.sum(1)[:,na]
    def _count_transitions(self,states_list):
        trans_counts = np.zeros((self.state_dim,self.state_dim),dtype=np.int32)
        for states in states_list:
            if len(states) >= 2:
                for idx in xrange(len(states)-1):
                    trans_counts[states[idx],states[idx+1]] += 1
        self.trans_counts = trans_counts
        return trans_counts
    def _get_m_slow(self,trans_counts):
        m = np.zeros((self.state_dim,self.state_dim),dtype=np.int32)
        if not (0 == trans_counts).all():
            for (rowidx, colidx), val in np.ndenumerate(trans_counts):
                if val > 0:
                    m[rowidx,colidx] = (np.random.rand(val) < self.alpha * self.beta[colidx] \
                            /(np.arange(val) + self.alpha*self.beta[colidx])).sum()
        self.m = m
        return m
    def _get_m(self,trans_counts):
        N = trans_counts.shape[0]
        m = np.zeros((N,N),dtype=np.int32)
        if not (0 == trans_counts).all():
            alpha, beta = self.alpha, self.beta
            scipy.weave.inline(
                    '''
                    for (int i=0; i<N; i++) {
                        for (int j=0; j<N; j++) {
                            int tot = 0;
                            for (int k=0; k<trans_counts[N*i+j]; k++) {
                                tot += ((double)rand())/RAND_MAX < (alpha * beta[j])/(k+alpha*beta[j]);
                            }
                            m[N*i+j] = tot;
                        }
                    }
                    ''',
                    ['trans_counts','N','m','alpha','beta'],
                    extra_compile_args=['-O3'])
        self.m = m
        return m
    ### max likelihood
    # TODO these methods shouldn't really be in this class... maybe put them in
    # a base class
    def max_likelihood(self,stateseqs,expectations_list=None):
        if expectations_list is not None:
            trans_counts = self._count_weighted_transitions(expectations_list,self.A)
        else:
            trans_counts = self._count_transitions(stateseqs)
        errs = np.seterr(invalid='ignore',divide='ignore')
        self.A = trans_counts / trans_counts.sum(1)[:,na]
        np.seterr(**errs)
        self.A[np.isnan(self.A)] = 0. # 1./self.state_dim # NOTE: just a reasonable hack, o/w undefined
    # NOTE: only needs aBl because the message computation saves betal and not
    # betastarl TODO compute betastarl like a civilized gentleman
    @staticmethod
    def _count_weighted_transitions(expectations_list,A):
        trans_softcounts = np.zeros_like(A)
        Al = np.log(A)
        for alphal, betal, aBl in expectations_list:
            log_joints = alphal[:-1,:,na] + (betal[1:,na,:] + aBl[1:,na,:]) + Al[na,...]
            log_joints -= np.logaddexp.reduce(alphal[0] + betal[0]) # p(y)
            joints = np.exp(log_joints,out=log_joints)
            trans_softcounts += joints.sum(0)
        return trans_softcounts

class HDPHMMTransitionsConcResampling(HDPHMMTransitions,ConcentrationResampling):
    def __init__(self,state_dim,
            alpha_a_0,alpha_b_0,gamma_a_0,gamma_b_0,
            **kwargs):
        ConcentrationResampling.__init__(self,state_dim,alpha_a_0,alpha_b_0,gamma_a_0,gamma_b_0)
        super(HDPHMMTransitionsConcResampling,self).__init__(state_dim,
                alpha=self.alpha_obj.concentration*state_dim,
                gamma=self.gamma_obj.concentration*state_dim,
                **kwargs)
    def resample(self,*args,**kwargs):
        super(HDPHMMTransitionsConcResampling,self).resample(*args,**kwargs)
        ConcentrationResampling.resample(self)


##############################################################
#  HDP-HSMM classes  #
##############################################################
class HDPHSMMTransitions(HDPHMMTransitions):
    '''
    HDPHSMM transition distribution class.
    Uses a weak-limit HDP prior. Zeroed diagonal to forbid self-transitions.
    Hyperparameters follow the notation in Fox et al.
        gamma: concentration paramter for beta
        alpha: total mass concentration parameter for each row of trans matrix
    Parameters are the shared transition vector beta, the full transition matrix,
    and the matrix with the diagonal zeroed.
        beta, A, fullA
    '''
    def __init__(self,state_dim,alpha,gamma,beta=None,A=None,fullA=None):
        self.alpha = alpha
        self.gamma = gamma
        self.state_dim = state_dim
        if A is None or fullA is None or beta is None:
            self.resample()
        else:
            self.A = A
            self.beta = beta
            self.fullA = fullA
    def resample(self,stateseqs=[]):
        states_noreps = map(operator.itemgetter(0),map(rle, stateseqs))
        augmented_data = self._augment_data(self._count_transitions(states_noreps))
        m = self._get_m(augmented_data)
        self._resample_beta(m)
        self._resample_A(augmented_data)
    def _augment_data(self,trans_counts):
        trans_counts = trans_counts.copy()
        if trans_counts.sum() > 0:
            froms = trans_counts.sum(1)
            self_transitions = [np.random.geometric(1-pi_ii,size=n).sum() if n > 0 else 0
                    for pi_ii,n in zip(self.fullA.diagonal(),froms)]
            trans_counts += np.diag(self_transitions)
        self.trans_counts = trans_counts
        return trans_counts
    def _resample_A(self,augmented_data):
        super(HDPHSMMTransitions,self)._resample_A(augmented_data)
        self.fullA = self.A.copy()
        self.A.flat[::self.A.shape[0]+1] = 0
        self.A /= self.A.sum(1)[:,na]

class HDPHSMMTransitionsConcResampling(HDPHSMMTransitions,ConcentrationResampling):
    def __init__(self,state_dim,
            alpha_a_0,alpha_b_0,gamma_a_0,gamma_b_0,
            **kwargs):
        ConcentrationResampling.__init__(self,state_dim,alpha_a_0,alpha_b_0,gamma_a_0,gamma_b_0)
        super(HDPHSMMTransitionsConcResampling,self).__init__(state_dim,
                alpha=self.alpha_obj.concentration*state_dim,
                gamma=self.gamma_obj.concentration*state_dim,
                **kwargs)
    def resample(self,*args,**kwargs):
        super(HDPHSMMTransitionsConcResampling,self).resample(*args,**kwargs)
        ConcentrationResampling.resample(self)
