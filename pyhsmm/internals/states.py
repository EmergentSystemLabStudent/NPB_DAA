import numpy as np
from numpy import newaxis as na
from numpy.random import random
import scipy.weave
import abc, copy, warnings
import scipy.stats as stats

np.seterr(invalid='raise')

from ..util.stats import sample_discrete, sample_discrete_from_log
from ..util import general as util # perhaps a confusing name :P

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
        for state,dur in zip(*util.rle(self.stateseq)):
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
        states,durations = util.rle(self.stateseq)
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
            self._stateseq_norep, self._durations_censored = util.rle(self.stateseq)
        return self._stateseq_norep

    @property
    def durations_censored(self):
        if self._durations_censored is None:
            self._stateseq_norep, self._durations_censored = util.rle(self.stateseq)
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

class HSMMStatesGeoApproximation(HSMMStatesPython):
    def _get_hmm_transition_matrix(self):
        trunc = self.trunc if self.trunc is not None else self.T
        state_dim = self.state_dim
        hmm_A = self.trans_matrix.copy()
        hmm_A.flat[::state_dim+1] = 0
        thediag = np.array([np.exp(d.log_pmf(trunc+1)-d.log_pmf(trunc))[0] for d in self.dur_distns])
        assert (thediag < 1).all(), 'truncation is too small!'
        hmm_A *= ((1-thediag)/hmm_A.sum(1))[:,na]
        hmm_A.flat[::state_dim+1] = thediag
        return hmm_A

    def messages_backwards(self):
        'approximates duration tails at indices > trunc with geometric tails'
        aDl, aDsl, Al = self.aDl, self.aDsl, np.log(self.trans_matrix)
        trunc = self.trunc if self.trunc is not None else self.T
        T,state_dim = aDl.shape

        assert trunc > 1

        aBl = self.aBl/self.temp if self.temp is not None else self.aBl
        hmm_betal = HMMStatesEigen._messages_backwards(self._get_hmm_transition_matrix(),aBl)
        assert not np.isnan(hmm_betal).any()

        betal = np.zeros((T,state_dim),dtype=np.float64)
        betastarl = np.zeros_like(betal)

        for t in xrange(T-1,-1,-1):
            np.logaddexp.reduce(betal[t:t+trunc] + self.cumulative_likelihoods(t,t+trunc)
                    + aDl[:min(trunc,T-t)],axis=0, out=betastarl[t])
            if t+trunc < T:
                np.logaddexp(betastarl[t], self.likelihood_block(t,t+trunc+1) + aDsl[trunc -1]
                        + hmm_betal[t+trunc], out=betastarl[t])
            if T-t < trunc and self.right_censoring:
                np.logaddexp(betastarl[t], self.likelihood_block(t,None) + aDsl[T-t -1], betastarl[t])
            np.logaddexp.reduce(betastarl[t] + Al,axis=1,out=betal[t-1])
        betal[-1] = 0.

        return betal, betastarl

class HSMMStatesGeoDynamicApproximation(HSMMStatesGeoApproximation):
    def messages_backwards(self):
        raise NotImplementedError # figure out trunc based on where log_pmf becomes approximately flat TODO
        super(HSMMStatesGeoDynamicApproximation,self).messages_backwards()

class _HSMMStatesIntegerNegativeBinomialBase(HMMStatesEigen, HSMMStatesPython):
    # NOTE: I'm secretly an HMMStates first! I just shh

    __metaclass__ = abc.ABCMeta

    def __init__(self,*args,**kwargs):
        HSMMStatesPython.__init__(self,*args,**kwargs)

    def clear_caches(self):
        HSMMStatesPython.clear_caches(self)
        self._hmm_trans = None
        self._rs = None
        self._hmm_aBl = None
        # note: we never use aDl or aDsl in this class

    def copy_sample(self,*args,**kwargs):
        return HSMMStatesPython.copy_sample(self,*args,**kwargs)

    @property
    def dur_distns(self):
        return self.model.dur_distns

    @property
    def hsmm_aBl(self):
        return super(_HSMMStatesIntegerNegativeBinomialBase,self).aBl

    @property
    def hsmm_trans_matrix(self):
        return super(_HSMMStatesIntegerNegativeBinomialBase,self).trans_matrix

    @property
    def hsmm_pi_0(self):
        return super(_HSMMStatesIntegerNegativeBinomialBase,self).pi_0

    # the next methods are to override the calls that the parents' methods would
    # make so that the parent can view us as the effective HMM we are!

    @property
    def aBl(self):
        if self._hmm_aBl is None or True:
            self._hmm_aBl = self.hsmm_aBl.repeat(self.rs,axis=1)
        return self._hmm_aBl

    @property
    def rs(self):
        if True or self._rs is None or True: # TODO
            self._rs = np.array([d.r for d in self.dur_distns],dtype=np.int)
        return self._rs

    @abc.abstractproperty
    def pi_0(self):
        pass

    @abc.abstractproperty
    def trans_matrix(self):
        pass

    # generic implementation, these could be overridden for efficiency
    # they act like HMMs, and they're probably called from an HMMStates method

    def generate_states(self):
        ret = HMMStatesEigen.generate_states(self)
        self._map_states()
        return ret

    def messages_backwards(self):
        return self.messages_backwards_hmm(), None # 2nd is a dummy, see sample_forwards

    def messages_forwards(self):
        return HMMStatesEigen.messages_forwards(self)

    def sample_forwards(self,betal,dummy):
        return self.sample_forwards_hmm(betal)

    def maxsum_messages_backwards(self):
        return self.maxsum_messages_backwards_hmm()

    def maximize_forwards(self,scores,args):
        return self.maximize_forwards_hmm(scores,args)

    def _map_states(self):
        themap = np.arange(self.state_dim).repeat(self.rs)
        self.stateseq = themap[self.stateseq]

    ### for testing, ensures calling parent HMM methods

    def Viterbi_hmm(self):
        scores, args = self.maxsum_messages_backwards_hmm()
        return self.maximize_forwards_hmm(scores,args)

    def messages_backwards_hmm(self):
        return HMMStatesEigen.messages_backwards(self)

    def sample_forwards_hmm(self,betal):
        ret = HMMStatesEigen.sample_forwards(self,betal)
        self._map_states()
        return ret

    def maxsum_messages_backwards_hmm(self):
        return HMMStatesEigen.maxsum_messages_backwards(self)

    def maximize_forwards_hmm(self,scores,args):
        ret = HMMStatesEigen.maximize_forwards(self,scores,args)
        self._map_states()
        return ret

class HSMMStatesIntegerNegativeBinomialVariant(_HSMMStatesIntegerNegativeBinomialBase):
    def clear_caches(self):
        super(HSMMStatesIntegerNegativeBinomialVariant,self).clear_caches()
        self._hmm_trans = None
        self._betal, self._superbetal = None, None

    def generate_states(self):
        return super(HSMMStatesIntegerNegativeBinomialVariant,self).generate_states()

    @property
    def pi_0(self):
        if not self.left_censoring:
            rs = self.rs
            starts = np.concatenate(((0,),rs.cumsum()[:-1]))
            pi_0 = np.zeros(rs.sum())
            pi_0[starts] = self.hsmm_pi_0
            return pi_0
        else:
            return util.top_eigenvector(self.trans_matrix)

    @property
    def trans_matrix(self):
        if self._hmm_trans is None or True:
            rs = self.rs
            ps = np.array([d.p for d in self.dur_distns])

            trans_matrix = np.zeros((rs.sum(),rs.sum()))
            trans_matrix += np.diag(np.repeat(ps,rs))
            trans_matrix += np.diag(np.repeat(1.-ps,rs)[:-1],k=1)
            for z in rs[:-1].cumsum():
                trans_matrix[z-1,z] = 0

            ends = rs.cumsum()
            starts = np.concatenate(((0,),rs.cumsum()[:-1]))
            for (i,j), v in np.ndenumerate(self.hsmm_trans_matrix * (1-ps)[:,na]):
                if i != j:
                    trans_matrix[ends[i]-1,starts[j]] = v

            self._hmm_trans = trans_matrix

        return self._hmm_trans

    @property
    def _trans_matrix_terms(self):
        # returns diag part and off-diag part of (hmm) trans matrix, along with
        # lengths
        rs = self.rs
        ps = np.array([d.p for d in self.dur_distns])
        return np.repeat(ps,rs), (1-ps)[:,na] * self.hsmm_trans_matrix, rs

    def E_step(self):
        raise NotImplementedError

    ### structure-exploiting methods

    def messages_backwards(self):
        if self._betal is not None and self._superbetal is not None:
            return self._betal, self._superbetal

        global eigen_path
        hsmm_intnegbin_messages_backwards_codestr = _get_codestr('hsmm_intnegbinvariant_messages_backwards')

        aBl = self.hsmm_aBl / self.temp if self.temp is not None else self.hsmm_aBl
        T,M = aBl.shape

        rs = self.rs
        crs = rs.cumsum()
        start_indices = np.concatenate(((0,),crs[:-1]))
        end_indices = crs-1
        rtot = int(crs[-1])
        ps = np.array([d.p for d in self.dur_distns])
        AT = self.hsmm_trans_matrix.T.copy() * (1-ps)

        superbetal = np.zeros((T,M))
        betal = np.zeros((T,rtot))

        scipy.weave.inline(hsmm_intnegbin_messages_backwards_codestr,
                ['start_indices','end_indices','rtot','AT','ps','superbetal','betal','aBl','T','M'],
                headers=['<Eigen/Core>'],include_dirs=[eigen_path],
                extra_compile_args=['-O3','-DNDEBUG'])

        assert not np.isnan(betal).any() and not np.isnan(superbetal).any()

        self._betal, self._superbetal = betal, superbetal

        return betal, superbetal

    def sample_forwards(self,betal,superbetal):
        global eigen_path
        hsmm_intnegbin_sample_forwards_codestr = _get_codestr('hsmm_intnegbinvariant_sample_forwards')

        aBl = self.hsmm_aBl / self.temp if self.temp is not None else self.hsmm_aBl
        T,M = aBl.shape
        rs = self.rs
        crs = rs.cumsum()
        start_indices = np.concatenate(((0,),crs[:-1]))
        end_indices = crs-1
        rtot = int(crs[-1])
        ps = np.array([d.p for d in self.dur_distns])
        A = self.hsmm_trans_matrix * (1-ps[:,na])
        A.flat[::A.shape[0]+1] = ps

        if self.left_censoring:
            initial_substate = sample_discrete_from_log(np.log(self.pi_0) + betal[0] + aBl[0].repeat(rs))
            initial_superstate = np.arange(self.state_dim).repeat(self.rs)[initial_substate]
        else:
            initial_superstate = sample_discrete_from_log(np.log(self.hsmm_pi_0) + superbetal[0] + aBl[0])
            initial_substate = start_indices[initial_superstate]

        stateseq = np.zeros(T,dtype=np.int32)

        scipy.weave.inline(hsmm_intnegbin_sample_forwards_codestr,
                ['betal','superbetal','aBl','stateseq','A','initial_superstate','initial_substate','M','T','ps','rtot','start_indices','end_indices'],
                headers=['<Eigen/Core>'],include_dirs=[eigen_path],
                extra_compile_args=['-O3','-DNDEBUG'])

        self.stateseq = stateseq # must have this line at end; it triggers stateseq_norep

    def maxsum_messages_backwards(self):
        global eigen_path
        # these names are dumb
        hsmm_intnegbin_maxsum_messages_backwards_codestr = _get_codestr('hsmm_intnegbinvariant_maxsum_messages_backwards')

        aBl = self.hsmm_aBl / self.temp if self.temp is not None else self.hsmm_aBl
        T,M = aBl.shape

        rs = self.rs
        crs = rs.cumsum()
        start_indices = np.concatenate(((0,),crs[:-1]))
        end_indices = crs-1
        rtot = int(crs[-1])
        ps = np.array([d.p for d in self.dur_distns])
        logps = np.log(ps)
        log1mps = np.log1p(-ps)
        Al = np.log(self.hsmm_trans_matrix) + log1mps[:,na]

        scores = np.zeros((T,rtot))
        args = np.zeros((T,rtot),dtype=np.int32)

        scipy.weave.inline(hsmm_intnegbin_maxsum_messages_backwards_codestr,
                ['start_indices','end_indices','rtot','Al','aBl',
                    'logps','log1mps','scores','args','T','M'],
                headers=['<Eigen/Core>'],include_dirs=[eigen_path],
                extra_compile_args=['-O3','-DNDEBUG'])

        return scores, args

    def maximize_forwards(self,scores,args):
        global eigen_path
        hsmm_intnegbin_maximize_forwards_codestr = _get_codestr('hsmm_intnegbin_maximize_forwards') # same code as intnegbin

        T,rtot = scores.shape

        rs = self.rs
        crs = rs.cumsum()
        start_indices = np.concatenate(((0,),crs[:-1]))
        themap = np.arange(self.state_dim).repeat(rs)

        stateseq = np.empty(T,dtype=np.int32)
        stateseq[0] = (scores[0,start_indices] + np.log(self.hsmm_pi_0) + self.hsmm_aBl[0]).argmax()
        initial_hmm_state = start_indices[stateseq[0]]

        scipy.weave.inline(hsmm_intnegbin_maximize_forwards_codestr,
                ['T','rtot','themap','scores','args','stateseq','initial_hmm_state'],
                headers=['<Eigen/Core>'],include_dirs=[eigen_path],
                extra_compile_args=['-O3','-DNDEBUG'])

        self.stateseq = stateseq # must have this line at end; it triggers stateseq_norep

class HSMMStatesIntegerNegativeBinomial(_HSMMStatesIntegerNegativeBinomialBase):
    def clear_caches(self):
        super(HSMMStatesIntegerNegativeBinomial,self).clear_caches()
        self._binoms = None

    # TODO test
    @property
    def binoms(self):
        raise NotImplementedError, 'theres a bug here' # TODO
        if self._binoms is None:
            self._binoms = []
            for D in self.dur_distns:
                if 0 < D.p < 1:
                    arr = stats.binom.pmf(np.arange(D.r),D.r,1.-D.p)
                    arr[-1] += stats.binom.pmf(D.r,D.r,1.-D.p)
                else:
                    arr = np.zeros(D.r)
                    arr[0] = 1
                self._binoms.append(arr)
        return self._binoms

    # TODO test
    @property
    def pi_0(self):
        if not self.left_censoring:
            rs = self.rs
            return self.hsmm_pi_0.repeat(rs) * np.concatenate(self.binoms)
        else:
            return util.top_eigenvector(self.trans_matrix)

    # TODO test
    @property
    def trans_matrix(self):
        if self._hmm_trans is None:
            rs = self.rs
            ps = np.array([d.p for d in self.dur_distns])

            trans_matrix = np.zeros((rs.sum(),rs.sum()))
            trans_matrix += np.diag(np.repeat(ps,rs))
            trans_matrix += np.diag(np.repeat(1.-ps,rs)[:-1],k=1)
            for z in rs[:-1].cumsum():
                trans_matrix[z-1,z] = 0

            ends = rs.cumsum()
            starts = np.concatenate(((0,),rs.cumsum()[:-1]))
            binoms = self.binoms
            for (i,j), v in np.ndenumerate(self.hsmm_trans_matrix * (1-ps)[:,na]):
                if i != j:
                    trans_matrix[ends[i]-1,starts[j]:ends[j]] = v * binoms[j]

            self._hmm_trans = trans_matrix

        return self._hmm_trans

    # matrix structure-exploiting methods

    def maxsum_messages_backwards(self):
        global eigen_path
        # these names are dumb
        hsmm_intnegbin_nonvariant_maxsum_messages_backwards_codestr = _get_codestr('hsmm_intnegbin_maxsum_messages_backwards')

        aBl = self.hsmm_aBl / self.temp if self.temp is not None else self.hsmm_aBl
        T,M = aBl.shape

        rs = self.rs
        crs = rs.cumsum()
        start_indices = np.concatenate(((0,),crs[:-1]))
        end_indices = crs-1
        rtot = int(crs[-1])
        ps = np.array([d.p for d in self.dur_distns])
        logps = np.log(ps)
        log1mps = np.log1p(-ps)
        Al = np.log(self.hsmm_trans_matrix) + log1mps[:,na]
        binoms = np.log(np.concatenate(self.binoms))

        scores = np.zeros((T,rtot))
        args = np.zeros((T,rtot),dtype=np.int32)

        scipy.weave.inline(hsmm_intnegbin_nonvariant_maxsum_messages_backwards_codestr,
                ['start_indices','end_indices','rtot','Al','aBl',
                    'logps','log1mps','scores','args','T','M','binoms'],
                headers=['<Eigen/Core>'],include_dirs=[eigen_path],
                extra_compile_args=['-O3','-DNDEBUG'])

        return scores, args

    def maximize_forwards(self,scores,args):
        global eigen_path
        hsmm_intnegbin_maximize_forwards_codestr = _get_codestr('hsmm_intnegbin_maximize_forwards')

        T,rtot = scores.shape

        rs = self.rs
        crs = rs.cumsum()
        start_indices = np.concatenate(((0,),crs[:-1]))
        themap = np.arange(self.state_dim).repeat(rs)

        stateseq = np.empty(T,dtype=np.int32)
        initial_hmm_state = (np.concatenate(self.binoms) + scores[0] + np.log(self.pi_0) + self.aBl[0]).argmax()
        stateseq[0] = themap[initial_hmm_state]

        scipy.weave.inline(hsmm_intnegbin_maximize_forwards_codestr,
                ['T','rtot','themap','scores','args','stateseq','initial_hmm_state'],
                headers=['<Eigen/Core>'],include_dirs=[eigen_path],
                extra_compile_args=['-O3','-DNDEBUG'])

        self.stateseq = stateseq # must have this line at end; it triggers stateseq_norep

#################
#  eigen stuff  #
#################

# TODO move away from weave, which is not maintained. numba? ctypes? cffi?
# cython? probably cython or ctypes.

import os
eigen_path = os.path.join(os.path.dirname(__file__),'../deps/Eigen3/')
eigen_code_dir = os.path.join(os.path.dirname(__file__),'cpp_eigen_code/')

codestrs = {}
def _get_codestr(name):
    if name not in codestrs:
        with open(os.path.join(eigen_code_dir,name+'.cpp')) as infile:
            codestrs[name] = infile.read()
    return codestrs[name]


####################
#  NEEDS UPDATING  #
####################

class HSMMStatesPossibleChangepoints(HSMMStatesPython): # TODO TODO update this class
    def __init__(self,model,changepoints,*args,**kwargs):
        warnings.warn("%s hasn't been used in a while; there may be some bumps"
                % self.__class__.__name__)

        self.changepoints = changepoints
        self.startpoints = np.array([start for start,stop in changepoints],dtype=np.int32)
        self.blocklens = np.array([stop-start for start,stop in changepoints],dtype=np.int32)
        self.Tblock = len(changepoints) # number of blocks
        super(HSMMStatesPossibleChangepoints,self).__init__(model,*args,**kwargs)

    ### generation

    def generate_states(self):
        # TODO TODO this method can probably call sample_forwards with dummy uniform
        # aBl/betal/betastarl, but that's just too complicated!
        Tblock = self.Tblock
        assert Tblock == len(self.changepoints)
        blockstateseq = np.zeros(Tblock,dtype=np.int32)

        tblock = 0
        nextstate_distr = self.pi_0
        A = self.trans_matrix

        while tblock < Tblock:
            # sample the state
            state = sample_discrete(nextstate_distr)

            # compute possible duration info (indep. of state)
            possible_durations = self.blocklens[tblock:].cumsum()

            # compute the pmf over those steps
            durprobs = self.dur_distns[state].pmf(possible_durations)
            # TODO censoring: the last possible duration isn't quite right
            durprobs /= durprobs.sum()

            # sample it
            blockdur = sample_discrete(durprobs) + 1

            # set block sequence
            blockstateseq[tblock:tblock+blockdur] = state

            # set up next iteration
            tblock += blockdur
            nextstate_distr = A[state]

        # convert block state sequence to full stateseq and stateseq_norep and
        # durations
        self.stateseq = np.zeros(self.T,dtype=np.int32)
        for state, (start,stop) in zip(blockstateseq,self.changepoints):
            self.stateseq[start:stop] = state
        self.stateseq_norep, self.durations_censored = util.rle(self.stateseq)

        return self.stateseq

    def generate(self): # TODO
        raise NotImplementedError

    ### caching

    def clear_caches(self):
        self._aBBl = None
        super(HSMMStatesPossibleChangepoints,self).clear_caches()

    @property
    def aBBl(self):
        if self._aBBl is None:
            aBl = self.aBl
            aBBl = self._aBBl = np.empty((self.Tblock,self.state_dim))
            for idx, (start,stop) in enumerate(self.changepoints):
                aBBl[idx] = aBl[start:stop].sum(0)
        return self._aBBl

    ### message passing

    def messages_backwards(self):
        aDl, Al = self.aDl, np.log(self.trans_matrix)
        Tblock = self.Tblock
        state_dim = Al.shape[0]
        trunc = self.trunc if self.trunc is not None else self.T

        betal = np.zeros((Tblock,state_dim),dtype=np.float64)
        betastarl = np.zeros_like(betal)

        for tblock in range(Tblock-1,-1,-1):
            possible_durations = self.blocklens[tblock:].cumsum() # could precompute these
            possible_durations = possible_durations[possible_durations < max(trunc,possible_durations[0]+1)]
            truncblock = len(possible_durations)
            normalizer = np.logaddexp.reduce(aDl[possible_durations-1],axis=0)

            np.logaddexp.reduce(betal[tblock:tblock+truncblock]
                    + self.block_cumulative_likelihoods(tblock,tblock+truncblock,possible_durations)
                    + aDl[possible_durations-1] - normalizer,axis=0,out=betastarl[tblock])
            # TODO TODO put censoring here, must implement likelihood_block
            np.logaddexp.reduce(betastarl[tblock] + Al, axis=1, out=betal[tblock-1])
        betal[-1] = 0.

        assert not np.isnan(betal).any()
        assert not np.isnan(betastarl).any()

        return betal, betastarl

    def block_cumulative_likelihoods(self,startblock,stopblock,possible_durations):
        return self.aBBl[startblock:stopblock].cumsum(0)[:possible_durations.shape[0]]

    def block_cumulative_likelihood_state(self,startblock,stopblock,state,possible_durations):
        return self.aBBl[startblock:stopblock,state].cumsum(0)[:possible_durations.shape[0]]

    ### Gibbs sampling

    def sample_forwards(self,betal,betastarl):
        aDl = self.aDl
        trunc = self.trunc

        Tblock = betal.shape[0]
        assert Tblock == len(self.changepoints)
        blockstateseq = np.zeros(Tblock,dtype=np.int32)

        tblock = 0
        nextstate_unsmoothed = self.pi_0
        A = self.trans_matrix
        trunc = trunc if trunc is not None else self.T

        while tblock < Tblock:
            # sample the state
            logdomain = betastarl[tblock] - np.amax(betastarl[tblock])
            nextstate_distr = np.exp(logdomain) * nextstate_unsmoothed
            state = sample_discrete(nextstate_distr)

            # compute possible duration info (indep. of state)
            # TODO TODO doesn't handle censoring quite correctly
            possible_durations = self.blocklens[tblock:].cumsum()
            possible_durations = possible_durations[possible_durations < max(trunc,possible_durations[0]+1)]
            truncblock = len(possible_durations)

            if truncblock > 1:
                # compute the next few log likelihoods
                loglikelihoods = self.block_cumulative_likelihood_state(tblock,tblock+truncblock,state,possible_durations)

                # compute pmf over those steps
                logpmf = aDl[possible_durations-1,state] + loglikelihoods + betal[tblock:tblock+truncblock,state] - betastarl[tblock,state]

                # sample from it
                blockdur = sample_discrete_from_log(logpmf)+1
            else:
                blockdur = 1

            # set block sequence
            blockstateseq[tblock:tblock+blockdur] = state

            # set up next iteration
            tblock += blockdur
            nextstate_unsmoothed = A[state]

        # convert block state sequence to full stateseq and stateseq_norep and
        # durations
        self.stateseq = np.zeros(self.T,dtype=np.int32)
        for state, (start,stop) in zip(blockstateseq,self.changepoints):
            self.stateseq[start:stop] = state
        self.stateseq_norep, self.durations_censored = util.rle(self.stateseq)

# NOTE: these only work with iid-like emissions with aBl (that includes
# autoregressive but not fancier things)

