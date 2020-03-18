import numpy as np
from functools import reduce

from pyhsmm.util.stats import sample_discrete
from pyhsmm.util.general import rle

class WeakLimitHDPHLMStatesPython(object):

    def __init__(self, model, data=None, trunc=None, generate=True, initialize_from_prior=False):
        self.model = model
        self.data = data
        self.T = T = len(data)
        self.trunc = trunc
        self._stateseq = np.zeros(T, dtype=np.int32)
        self._stateseq_norep = None
        self._durations_censored = None
        self._normalizer = None
        self._letter_stateseq = np.zeros(T, dtype=np.int32)
        self._kwargs = dict(trunc=trunc)
        if generate:
            if data is not None and not initialize_from_prior:
                self.resample()
            else:
                self.generate_states()
        self.clear_caches()

    def generate_states(self):
        raise NotImplementedError

    @property
    def stateseq(self):
        return self._stateseq

    @stateseq.setter
    def stateseq(self, stateseq):
        self._stateseq = stateseq

    @property
    def letter_stateseq(self):
        return self._letter_stateseq

    @letter_stateseq.setter
    def letter_stateseq(self, letter_stateseq):
        self._letter_stateseq = letter_stateseq

    @property
    def stateseq_norep(self):
        if self._stateseq_norep is None:
            self._stateseq_norep, self._durations_censored  = rle(self.stateseq)
        return self._stateseq_norep

    @property
    def durations_censored(self):
        if self._durations_censored is None:
            self._stateseq_norep, self._durations_censored = rle(self.stateseq)
        return self._durations_censored

    # Be care full!!!!
    # This method return the log likelihood which before resampling this model.
    def log_likelihood(self):
        if self._normalizer is None:
            _, _, normalizerl = self.messages_backwards()
            self._normalizer = normalizerl
        return self._normalizer

    @property
    def pi_0(self):
        return self.model.init_state_distn.pi_0

    @property
    def aDl(self):
        if self._aDl is None:
            aDl = np.empty((self.T,self.model.num_states))
            possible_durations = np.arange(1,self.T + 1,dtype=np.float64)
            for idx, dur_distn in enumerate(self.model.dur_distns):
                aDl[:,idx] = dur_distn.log_pmf(possible_durations)
            self._aDl = aDl
        return self._aDl

    @property
    def alDl(self):
        if self._alDl is None:
            alDl = np.empty((self.T,self.model.letter_num_states))
            possible_durations = np.arange(1,self.T + 1,dtype=np.float64)
            for idx, dur_distn in enumerate(self.model.letter_dur_distns):
                alDl[:,idx] = dur_distn.log_pmf(possible_durations)
            self._alDl = alDl
        return self._alDl

    @property
    def aBl(self):
        if self._aBl is None:
            aBl = np.empty((self.data.shape[0], self.model._letter_num_states))
            for idx, obs_distn in enumerate(self.model.letter_obs_distns):
                aBl[:,idx] = obs_distn.log_likelihood(self.data).ravel()
            aBl[np.isnan(aBl).any(1)] = 0.0
            self._aBl = aBl
        return self._aBl

    @property
    def trans_matrix(self):
        return self.model.trans_distn.trans_matrix

    @property
    def log_trans_matrix(self):
        if self._log_trans_matrix is None:
            self._log_trans_matrix = np.log(self.trans_matrix)
        return self._log_trans_matrix

    def resample(self):
        self.clear_caches()
        betal, betastarl, normalizerl = self.messages_backwards()
        self._normalizer = normalizerl
        self.sample_forwards(betal, betastarl)

    def messages_backwards(self):
        aDl = self.aDl
        log_trans_matrix = self.log_trans_matrix
        T = self.T
        pi_0 = self.pi_0
        trunc = self.trunc if self.trunc is not None else T
        betal = np.zeros((T, self.model.num_states), dtype=np.float64)
        betastarl = np.zeros((T, self.model.num_states), dtype=np.float64)

        return hlm_messages_backwards_log(self.cumulative_likelihoods, aDl, log_trans_matrix, pi_0, trunc, betal, betastarl)

    def cumulative_likelihoods(self, start, stop):
        T = min(self.T, stop)
        tsize = T - start
        cum_like = np.empty((tsize, self.model.num_states), dtype=np.float64)

        for state, word in enumerate(self.model.word_list):
            cum_like[:, state] = self.likelihood_block_word(start, stop, word)

        return cum_like

    def likelihood_block_word(self, start, stop, word):
        T = min(self.T, stop)
        tsize = T - start
        aBl = self.aBl[start:T]
        alDl = self.alDl[:tsize]
        L = len(word)
        alphal = np.ones((tsize, L), dtype=np.float64) * -np.inf

        return hlm_internal_hsmm_messages_forwards_log(aBl, alDl, word, alphal)[:, -1]

    def sample_forwards(self, betal, betastarl):
        T = self.T
        aD = np.exp(self.aDl)
        self._letter_stateseq[:] = -1
        stateseq, stateseq_norep, durations_censored = hlm_sample_forwards_log(
            self.likelihood_block_word, self.trans_matrix, self.pi_0, self.aDl, self.model.word_list,
            betal, betastarl,
            np.empty(T, dtype=np.int32),[], [])

        self._stateseq = stateseq
        self._stateseq_norep = stateseq_norep
        self._durations_censored = durations_censored

    def clear_caches(self):
        self._aBl = None
        self._aDl = None
        self._alDl = None
        self._log_trans_matrix = None

    def add_word_datas(self, **kwargs):
        s = self.stateseq_norep
        d = self.durations_censored
        dc = np.concatenate(([0], d)).cumsum()
        for i, word_idx in enumerate(s):
            self.model.add_word_data(self.data[dc[i]:dc[i+1]], hlmstate=self, word_idx=word_idx, d0=dc[i], d1=dc[i+1], **kwargs)

class WeakLimitHDPHLMStates(WeakLimitHDPHLMStatesPython):

    def messages_backwards(self):
        from pyhlm.internals.hlm_messages_interface import messages_backwards_log
        words = np.array(reduce(lambda a, b: a + b, self.model.word_list), dtype=np.int32)
        Ls = np.array([len(word) for word in self.model.word_list], dtype=np.int32)
        cLs = np.concatenate(([0], np.cumsum(Ls)[:-1])).astype(np.int32)
        Lmax = Ls.max()
        N = self.model.num_states
        T = self.T
        pi_0 = self.pi_0
        trunc = self.trunc if self.trunc is not None else T
        betal, betastarl = messages_backwards_log(
            self.aBl, self.aDl, self.alDl, self.log_trans_matrix,
            words, Ls, cLs, Lmax, trunc,
            np.zeros((T, self.model.num_states), dtype=np.float64),
            np.zeros((T, self.model.num_states), dtype=np.float64)
        )

        assert not np.isnan(betal).any()
        assert not np.isnan(betastarl).any()

        normalizerl = np.logaddexp.reduce(betastarl[0] + np.log(pi_0))

        return betal, betastarl, normalizerl

    def messages_backwards_python(self):
        return super(WeakLimitHDPHLMStates, self).messages_backwards()

    def likelihood_block_word(self, start, stop, word):
        from pyhlm.internals.internal_hsmm_messages_interface import internal_hsmm_messages_forwards_log
        T = min(self.T, stop)
        tsize = T - start
        aBl = self.aBl[start:T]
        alDl = self.alDl[:tsize]
        L = len(word)
        alphal = np.ones((tsize, L), dtype=np.float64) * -np.inf

        if tsize - L + 1 <= 0:
            return alphal[:, -1]

        return internal_hsmm_messages_forwards_log(aBl, alDl, np.array(word, dtype=np.int32), alphal)[:, -1]

    def likelihood_block_word_python(self, start, stop, word):
        return super(WeakLimitHDPHLMStates, self).likelihood_block_word(start, stop, word)

def hlm_internal_hsmm_messages_forwards_log(aBl, alDl, word, alphal):
    T = alphal.shape[0]
    L = alphal.shape[1]
    alphal[:] = -np.inf

    if T-L+1 <= 0:
        return alphal

    cumsum_aBl = np.empty(T-L+1, dtype=np.float64)
    alphal[:T-L+1, 0] = np.cumsum(aBl[:T-L+1, word[0]]) + alDl[:T-L+1, word[0]]
    cache_range = range(T - L + 1)
    for j, l in enumerate(word[1:]):
        cumsum_aBl[:] = 0.0
        for t in cache_range:
            cumsum_aBl[:t+1] += aBl[t+j+1, l]
            alphal[t+j+1, j+1] = np.logaddexp.reduce(cumsum_aBl[:t+1] + alDl[t::-1, l] + alphal[j:t+j+1, j])
    return alphal

def hlm_messages_backwards_log(cumulative_likelihoods_func, aDl, log_trans_matrix, pi_0, trunc, betal, betastarl):
    T = betal.shape[0]

    for t in range(T-1, -1, -1):
        betastarl[t] = np.logaddexp.reduce(
            betal[t:t+trunc] + cumulative_likelihoods_func(t, t+trunc) + aDl[:min(trunc, T-t)],
            axis=0
        )
        betal[t-1] = np.logaddexp.reduce(betastarl[t] + log_trans_matrix, axis=1)
    betal[-1] = 0.0
    normalizerl = np.logaddexp.reduce(betastarl[0] + np.log(pi_0))
    return betal, betastarl, normalizerl

def hlm_sample_forwards_log(likelihood_block_word_func, trans_matrix, pi_0, aDl, word_list, betal, betastarl, stateseq, stateseq_norep, durations_censored):
    stateseq[:] = -1
    T = betal.shape[0]
    t = 0
    aD = np.exp(aDl)
    nextstate_unsmoothed = pi_0
    while t < T:
        logdomain = betastarl[t] - betastarl[t].max()
        nextstate_dist = np.exp(logdomain) * nextstate_unsmoothed

        state = sample_discrete(nextstate_dist)

        durprob = np.random.random()
        cache_mess_term = np.exp(likelihood_block_word_func(t, T, word_list[state]) + betal[t:T, state] - betastarl[t, state])

        dur = 0
        while durprob > 0 and t+dur < T:
            # p_d_prior = aD[dur, state] if t + dur < T else 1.
            p_d_prior = aD[dur, state]
            assert not np.isnan(p_d_prior)
            assert p_d_prior >= 0

            p_d = cache_mess_term[dur] * p_d_prior
            assert not np.isnan(p_d)
            durprob -= p_d
            dur += 1

        assert dur > 0
        assert dur >= len(word_list[state])
        stateseq[t:t+dur] = state
        nextstate_unsmoothed = trans_matrix[state]
        t += dur

        stateseq_norep.append(state)
        durations_censored.append(dur)
    stateseq_norep = np.array(stateseq_norep, dtype=np.int32)
    durations_censored = np.array(durations_censored, dtype=np.int32)
    return stateseq, stateseq_norep, durations_censored
