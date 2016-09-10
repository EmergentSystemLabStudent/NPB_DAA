import numpy as np
from numpy import newaxis as na
from numpy.random import random
np.seterr(invalid='raise')
import scipy.weave
import copy, os
from ..basic.util import sample_discrete, top_eigenvector, rle
from ..basic.clustering import Categorical


#########################################################
#  Eigen stuff  #
#########################################################
# TODO move away from weave, which is not maintained. numba? ctypes? cffi?
# cython? probably cython or ctypes.
eigen_path = os.path.join(os.path.dirname(__file__),'eigen3/')
eigen_code_dir = os.path.join(os.path.dirname(__file__),'cpp_eigen_code/')
codestrs = {}
def _get_codestr(name):
    if name not in codestrs:
        with open(os.path.join(eigen_code_dir,name+'.cpp')) as infile:
            codestrs[name] = infile.read()
    return codestrs[name]


##################################################
#  Initial  #
##################################################
class InitialState(Categorical):
    def __init__(self,state_dim,rho,pi_0=None):
        super(InitialState,self).__init__(alpha_0=rho,K=state_dim,weights=pi_0)
    @property
    def pi_0(self):
        return self.weights

class SteadyState(object):
    def __init__(self,model):
        self.model = model
        self.clear_caches()
    def clear_caches(self):
        self._pi = None
    @property
    def pi_0(self):
        if self._pi is None:
            self._pi = top_eigenvector(self.model.trans_distn.A)
        return self._pi
    def resample(self,*args,**kwargs):
        pass

class HSMMSteadyState(SteadyState):
    @property
    def pi_0(self):
        if self._pi is None:
            markov_part = super(HSMMSteadyState,self).pi_0
            duration_expectations = np.array([d.mean for d in self.model.dur_distns])
            self._pi = markov_part * duration_expectations
            self._pi /= self._pi.sum()
        return self._pi


##################################################
#  HMM state  #
##################################################
# TODO using log(A) in message passing can hurt stability a bit, -1000 turns
# into -inf
# TODO abstract this cache handling... metaclass and a cached decorator?
class HMMStatesPython(object):
    def __init__(self,model,T=None,data=None,stateseq=None,
            initialize_from_prior=True,**kwargs):
        self.model = model
        assert (data is None) ^ (T is None)
        self.T = data.shape[0] if data is not None else T
        self.data = data
        self.clear_caches()
        if stateseq is not None:
            self.stateseq = np.array(stateseq,dtype=np.int32)
        else:
            if data is not None and not initialize_from_prior:
                self.resample(**kwargs)
            else:
                self.generate_states()
    @property
    def trans_matrix(self):
        return self.model.trans_distn.A
    @property
    def pi_0(self):
        return self.model.init_state_distn.pi_0
    @property
    def obs_distns(self):
        return self.model.obs_distns
    @property
    def state_dim(self):
        return self.model.state_dim
    ### generation
    def generate(self):
        self.generate_states()
        return self.generate_obs()
    def generate_states(self):
        T = self.T
        nextstate_distn = self.pi_0
        A = self.trans_matrix
        stateseq = np.zeros(T,dtype=np.int32)
        for idx in xrange(T):
            stateseq[idx] = sample_discrete(nextstate_distn)
            nextstate_distn = A[stateseq[idx]]
        self.stateseq = stateseq
        return stateseq
    def generate_obs(self):
        obs = []
        for state,dur in zip(*rle(self.stateseq)):
            obs.append(self.obs_distns[state].rvs(int(dur)))
        return np.concatenate(obs)
    ### caching common computation needed for several methods
    # this stuff depends on model parameters, so it must be cleared when the
    # model changes
    def clear_caches(self):
        self._aBl = None
        self._betal = None
    @property
    def aBl(self):
        if self._aBl is None:
            data = self.data
            aBl = self._aBl = np.empty((data.shape[0],self.state_dim))
            for idx, obs_distn in enumerate(self.obs_distns):
                aBl[:,idx] = np.nan_to_num(obs_distn.log_likelihood(data))
        return self._aBl
    ### message passing
    @staticmethod
    def _messages_backwards(trans_matrix,log_likelihoods):
        Al = np.log(trans_matrix)
        aBl = log_likelihoods
        betal = np.zeros_like(aBl)
        for t in xrange(betal.shape[0]-2,-1,-1):
            np.logaddexp.reduce(Al + betal[t+1] + aBl[t+1],axis=1,out=betal[t])
        return betal
    def messages_backwards(self):
        if self._betal is not None:
            return self._betal
        aBl = self.aBl/self.temp if hasattr(self,'temp') and self.temp is not None else self.aBl
        self._betal = self._messages_backwards(self.trans_matrix,aBl)
        return self._betal
    @staticmethod
    def _messages_forwards(trans_matrix,init_state_distn,log_likelihoods):
        Al = np.log(trans_matrix)
        aBl = log_likelihoods
        alphal = np.zeros_like(aBl)
        alphal[0] = np.log(init_state_distn) + aBl[0]
        for t in xrange(alphal.shape[0]-1):
            alphal[t+1] = np.logaddexp.reduce(alphal[t] + Al.T,axis=1) + aBl[t+1]
        return alphal
    def messages_forwards(self):
        return self._messages_forwards(self.trans_matrix,self.pi_0,self.aBl)
    ### Gibbs sampling
    def resample(self,temp=None):
        self.temp = temp
        betal = self.messages_backwards()
        self.sample_forwards(betal)
    def copy_sample(self,newmodel):
        new = copy.copy(self)
        new.clear_caches() # saves space, though may recompute later for likelihoods
        new.model = newmodel
        new.stateseq = self.stateseq.copy()
        return new
    @staticmethod
    def _sample_forwards(betal,trans_matrix,init_state_distn,log_likelihoods):
        A = trans_matrix
        aBl = log_likelihoods
        T = aBl.shape[0]
        stateseq = np.empty(T,dtype=np.int32)
        nextstate_unsmoothed = init_state_distn
        for idx in xrange(T):
            logdomain = betal[idx] + aBl[idx]
            logdomain[nextstate_unsmoothed == 0] = -np.inf # to enforce constraints in the trans matrix
            stateseq[idx] = sample_discrete(nextstate_unsmoothed * np.exp(logdomain - np.amax(logdomain)))
            nextstate_unsmoothed = A[stateseq[idx]]
        return stateseq
    def sample_forwards(self,betal):
        aBl = self.aBl/self.temp if self.temp is not None else self.aBl
        self.stateseq = self._sample_forwards(betal,self.trans_matrix,self.pi_0,aBl)
    ### EM
    def E_step(self):
        alphal = self.alphal = self.messages_forwards()
        betal = self.betal = self.messages_backwards()
        expectations = self.expectations = alphal + betal
        expectations -= expectations.max(1)[:,na]
        np.exp(expectations,out=expectations)
        expectations /= expectations.sum(1)[:,na]
        self.stateseq = expectations.argmax(1)
    ### Viterbi
    def Viterbi(self):
        scores, args = self.maxsum_messages_backwards()
        self.maximize_forwards(scores,args)
    @staticmethod
    def _maxsum_messages_backwards(trans_matrix, log_likelihoods):
        Al = np.log(trans_matrix)
        aBl = log_likelihoods
        scores = np.zeros_like(aBl)
        args = np.zeros(aBl.shape,dtype=np.int32)
        for t in xrange(scores.shape[0]-2,-1,-1):
            vals = Al + scores[t+1] + aBl[t+1]
            vals.argmax(axis=1,out=args[t+1])
            vals.max(axis=1,out=scores[t])
        return scores, args
    def maxsum_messages_backwards(self):
        return self._maxsum_messages_backwards(self.trans_matrix,self.aBl)
    @staticmethod
    def _maximize_forwards(scores,args,init_state_distn,log_likelihoods):
        aBl = log_likelihoods
        T = aBl.shape[0]
        stateseq = np.empty(T,dtype=np.int32)
        stateseq[0] = (scores[0] + np.log(init_state_distn) + aBl[0]).argmax()
        for idx in xrange(1,T):
            stateseq[idx] = args[idx,stateseq[idx-1]]
        return stateseq
    def maximize_forwards(self,scores,args):
        self.stateseq = self._maximize_forwards(scores,args,self.pi_0,self.aBl)
    ### plotting
    def plot(self,colors_dict=None,vertical_extent=(0,1),**kwargs):
        from matplotlib import pyplot as plt
        states,durations = rle(self.stateseq)
        X,Y = np.meshgrid(np.hstack((0,durations.cumsum())),vertical_extent)
        if colors_dict is not None:
            C = np.array([[colors_dict[state] for state in states]])
        else:
            C = states[na,:]
        plt.pcolor(X,Y,C,vmin=0,vmax=1,**kwargs)
        plt.ylim(vertical_extent)
        plt.xlim((0,durations.sum()))
        plt.yticks([])

class HMMStatesEigen(HMMStatesPython):
    ### common messages (Gibbs, EM, likelihood calculation)
    @staticmethod
    def _messages_backwards(trans_matrix,log_likelihoods):
        global eigen_path
        hmm_messages_backwards_codestr = _get_codestr('hmm_messages_backwards')
        T,M = log_likelihoods.shape
        AT = trans_matrix.T.copy() # because Eigen is fortran/col-major, numpy default C/row-major
        aBl = log_likelihoods
        betal = np.zeros((T,M))
        scipy.weave.inline(hmm_messages_backwards_codestr,['AT','betal','aBl','T','M'],
                headers=['<Eigen/Core>'],include_dirs=[eigen_path],
                extra_compile_args=['-O3','-DNDEBUG'])
        return betal
    @staticmethod
    def _messages_forwards(trans_matrix,init_state_distn,log_likelihoods):
        global eigen_path
        hmm_messages_forwards_codestr = _get_codestr('hmm_messages_forwards')
        T,M = log_likelihoods.shape
        A = trans_matrix
        aBl = log_likelihoods
        alphal = np.empty((T,M))
        alphal[0] = np.log(init_state_distn) + aBl[0]
        scipy.weave.inline(hmm_messages_forwards_codestr,['A','alphal','aBl','T','M'],
                headers=['<Eigen/Core>'],include_dirs=[eigen_path],
                extra_compile_args=['-O3','-DNDEBUG'])
        return alphal
    ### sampling
    @staticmethod
    def _sample_forwards(betal,trans_matrix,init_state_distn,log_likelihoods):
        global eigen_path
        hmm_sample_forwards_codestr = _get_codestr('hmm_sample_forwards')
        T,M = betal.shape
        A = trans_matrix
        pi0 = init_state_distn
        aBl = log_likelihoods
        stateseq = np.zeros(T,dtype=np.int32)
        scipy.weave.inline(hmm_sample_forwards_codestr,['A','T','pi0','stateseq','aBl','betal','M'],
                headers=['<Eigen/Core>','<limits>'],include_dirs=[eigen_path],
                extra_compile_args=['-O3','-DNDEBUG'])
        return stateseq
    ### Vitberbi
    @staticmethod
    def _maxsum_messages_backwards(trans_matrix,log_likelihoods):
        global eigen_path
        hmm_maxsum_messages_backwards_codestr = _get_codestr('hmm_maxsum_messages_backwards')
        Al = np.log(trans_matrix)
        aBl = log_likelihoods
        T,M = log_likelihoods.shape
        scores = np.zeros_like(aBl)
        args = np.zeros(aBl.shape,dtype=np.int32)
        scipy.weave.inline(hmm_maxsum_messages_backwards_codestr,['Al','aBl','T','M','scores','args'],
                headers=['<Eigen/Core>','<limits>'],include_dirs=[eigen_path],
                extra_compile_args=['-O3','-DNDEBUG'])
        return scores, args
    @staticmethod
    def _maximize_forwards(scores,args,init_state_distn,log_likelihoods):
        global eigen_path
        hmm_maximize_forwards_codestr = _get_codestr('hmm_maximize_forwards')
        T,M = log_likelihoods.shape
        stateseq = np.empty(T,dtype=np.int32)
        stateseq[0] = (scores[0] + np.log(init_state_distn) + log_likelihoods[0]).argmax()
        scipy.weave.inline(hmm_maximize_forwards_codestr,['stateseq','args','scores','T','M'],
                headers=['<Eigen/Core>','<limits>'],include_dirs=[eigen_path],
                extra_compile_args=['-O3','-DNDEBUG'])
        return stateseq


##################################################
#  HSMM state  #
##################################################
class HSMMStatesPython(HMMStatesPython):
    def __init__(self,model,right_censoring=True,left_censoring=False,trunc=None,
            stateseq=None,**kwargs):
        self.right_censoring = right_censoring
        self.left_censoring = left_censoring
        self.trunc = trunc
        super(HSMMStatesPython,self).__init__(model,stateseq=stateseq,**kwargs)
    def _get_stateseq(self):
        return self._stateseq
    def _set_stateseq(self,stateseq):
        self._stateseq = stateseq
        self._stateseq_norep = None
        self._durations_censored = None
    stateseq = property(_get_stateseq,_set_stateseq)
    @property
    def stateseq_norep(self):
        if self._stateseq_norep is None:
            self._stateseq_norep, self._durations_censored = rle(self.stateseq)
        return self._stateseq_norep
    @property
    def durations_censored(self):
        if self._durations_censored is None:
            self._stateseq_norep, self._durations_censored = rle(self.stateseq)
        return self._durations_censored
    @property
    def durations(self):
        durs = self.durations_censored.copy()
        if self.left_censoring:
            durs[0] = self.dur_distns[self.stateseq_norep[0]].rvs_given_greater_than(durs[0]-1)
        if self.right_censoring:
            durs[-1] = self.dur_distns[self.stateseq_norep[-1]].rvs_given_greater_than(durs[-1]-1)
        return durs
    @property
    def untrunc_slice(self):
        return slice(1 if self.left_censoring else 0, -1 if self.right_censoring else None)
    @property
    def trunc_slice(self):
        # not really a slice, but this can be passed in to an ndarray
        trunced = []
        if self.left_censoring:
            trunced.append(0)
        if self.right_censoring:
            trunced.append(-1)
        return trunced
    @property
    def pi_0(self):
        if not self.left_censoring:
            return self.model.init_state_distn.pi_0
        else:
            return self.model.left_censoring_init_state_distn.pi_0
    @property
    def dur_distns(self):
        return self.model.dur_distns
    ### generation
    def generate_states(self):
        if self.left_censoring:
            raise NotImplementedError # TODO
        idx = 0
        nextstate_distr = self.pi_0
        A = self.trans_matrix
        stateseq = np.empty(self.T,dtype=np.int32)
        # durations = []
        while idx < self.T:
            # sample a state
            state = sample_discrete(nextstate_distr)
            # sample a duration for that state
            duration = self.dur_distns[state].rvs()
            # save everything
            # durations.append(duration)
            stateseq[idx:idx+duration] = state # this can run off the end, that's okay
            # set up next state distribution
            nextstate_distr = A[state,]
            # update index
            idx += duration
        self.stateseq = stateseq
    ### caching
    def clear_caches(self):
        self.temp = 1
        self._aDl = None
        self._aDsl = None
        self._betal, self._betastarl = None, None
        super(HSMMStatesPython,self).clear_caches()
    @property
    def aDl(self):
        if self._aDl is None:
            self._aDl = aDl = np.empty((self.T,self.state_dim))
            possible_durations = np.arange(1,self.T + 1,dtype=np.float64)
            for idx, dur_distn in enumerate(self.dur_distns):
                aDl[:,idx] = dur_distn.log_pmf(possible_durations)
        return self._aDl
    @property
    def aD(self):
        return np.exp(self.aDl)
    @property
    def aDsl(self):
        if self._aDsl is None:
            self._aDsl = aDsl = np.empty((self.T,self.state_dim))
            possible_durations = np.arange(1,self.T + 1,dtype=np.float64)
            for idx, dur_distn in enumerate(self.dur_distns):
                aDsl[:,idx] = dur_distn.log_sf(possible_durations)
        return self._aDsl
    ### message passing
    def messages_backwards(self):
        if self._betal is not None and self._betastarl is not None:
            return self._betal, self._betastarl
        aDl, aDsl, Al = self.aDl, self.aDsl, np.log(self.trans_matrix)
        T,state_dim = aDl.shape
        trunc = self.trunc if self.trunc is not None else T
        betal = np.zeros((T,state_dim),dtype=np.float64)
        betastarl = np.zeros((T,state_dim),dtype=np.float64)
        for t in xrange(T-1,-1,-1):
            np.logaddexp.reduce(betal[t:t+trunc] + self.cumulative_likelihoods(t,t+trunc) + aDl[:min(trunc,T-t)],axis=0, out=betastarl[t])
            if T-t-1 < trunc and self.right_censoring:
                np.logaddexp(betastarl[t], self.likelihood_block(t,None) + aDsl[T-t -1], betastarl[t])
            np.logaddexp.reduce(betastarl[t] + Al,axis=1,out=betal[t-1])
        betal[-1] = 0.
        self._betal, self._betastarl = betal, betastarl
        return betal, betastarl
    def cumulative_likelihoods(self,start,stop):
        out = np.cumsum(self.aBl[start:stop],axis=0)
        return out if self.temp is None else out/self.temp
    def cumulative_likelihood_state(self,start,stop,state):
        out = np.cumsum(self.aBl[start:stop,state])
        return out if self.temp is None else out/self.temp
    def likelihood_block(self,start,stop):
        out = np.sum(self.aBl[start:stop],axis=0)
        return out if self.temp is None else out/self.temp
    def likelihood_block_state(self,start,stop,state):
        out = np.sum(self.aBl[start:stop,state])
        return out if self.temp is None else out/self.temp
    ### Gibbs sampling
    def resample(self,temp=None):
        self.temp = temp
        betal, betastarl = self.messages_backwards()
        self.sample_forwards(betal,betastarl)
    def copy_sample(self,newmodel):
        new = super(HSMMStatesPython,self).copy_sample(newmodel)
        return new
    def sample_forwards(self,betal,betastarl):
        if self.left_censoring:
            raise NotImplementedError # TODO
        A = self.trans_matrix
        apmf = self.aD
        T, state_dim = betal.shape
        stateseq = self.stateseq = np.zeros(T,dtype=np.int32)
        idx = 0
        nextstate_unsmoothed = self.pi_0
        while idx < T:
            logdomain = betastarl[idx] - np.amax(betastarl[idx])
            nextstate_distr = np.exp(logdomain) * nextstate_unsmoothed
            if (nextstate_distr == 0.).all():
                # this is a numerical issue; no good answer, so we'll just follow the messages.
                nextstate_distr = np.exp(logdomain)
            state = sample_discrete(nextstate_distr)
            durprob = random()
            dur = 0 # always incremented at least once
            prob_so_far = 0.0
            while durprob > 0:
                # NOTE: funny indexing: dur variable is 1 less than actual dur
                # we're considering, i.e. if dur=5 at this point and we break
                # out of the loop in this iteration, that corresponds to
                # sampling a duration of 6
                p_d_prior = apmf[dur,state] if dur < T else 1.
                assert not np.isnan(p_d_prior)
                assert p_d_prior >= 0
                if p_d_prior == 0:
                    dur += 1
                    continue
                if idx+dur < T:
                    mess_term = np.exp(self.likelihood_block_state(idx,idx+dur+1,state) \
                            + betal[idx+dur,state] - betastarl[idx,state])
                    p_d = mess_term * p_d_prior
                    prob_so_far += p_d
                    assert not np.isnan(p_d)
                    durprob -= p_d
                    dur += 1
                else:
                    if self.right_censoring:
                        dur = self.dur_distns[state].rvs_given_greater_than(dur)
                    else:
                        dur += 1
                    break
            assert dur > 0
            stateseq[idx:idx+dur] = state
            # stateseq_norep.append(state)
            # assert len(stateseq_norep) < 2 or stateseq_norep[-1] != stateseq_norep[-2]
            # durations.append(dur)
            nextstate_unsmoothed = A[state,:]
            idx += dur
    ### plotting
    def plot(self,colors_dict=None,**kwargs):
        from matplotlib import pyplot as plt
        X,Y = np.meshgrid(np.hstack((0,self.durations_censored.cumsum())),(0,1))
        if colors_dict is not None:
            C = np.array([[colors_dict[state] for state in self.stateseq_norep]])
        else:
            C = self.stateseq_norep[na,:]
        plt.pcolor(X,Y,C,vmin=0,vmax=1,**kwargs)
        plt.ylim((0,1))
        plt.xlim((0,self.T))
        plt.yticks([])
        plt.title('State Sequence')

class HSMMStatesEigen(HSMMStatesPython):
    def sample_forwards(self,betal,betastarl):
        if self.left_censoring:
            raise NotImplementedError # TODO
        global eigen_path
        hsmm_sample_forwards_codestr = _get_codestr('hsmm_sample_forwards')
        A = self.trans_matrix
        apmf = self.aD
        T,M = betal.shape
        pi0 = self.pi_0
        aBl = self.aBl / self.temp if self.temp is not None else self.aBl
        stateseq = np.zeros(T,dtype=np.int32)
        scipy.weave.inline(hsmm_sample_forwards_codestr,
                ['betal','betastarl','aBl','stateseq','A','pi0','apmf','M','T'],
                headers=['<Eigen/Core>'],include_dirs=[eigen_path],
                extra_compile_args=['-O3','-DNDEBUG'])
        self.stateseq = stateseq # must have this line at end; it triggers stateseq_norep
