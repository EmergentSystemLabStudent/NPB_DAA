import numpy as np
from pyhsmm.util.stats import sample_discrete
from pyhsmm.util.general import rle

class HSMMState(object):
    def __init__(self, data, model):
        self.data = np.asarray(data)
        self.model = model
        self.T = len(data)
        self.stateseq = []
        self.letters = []
        self.clear_caches()
        self.resample()
        self.betal

    def clear_caches(self):
        self._Al = None
        self._aDl = None
        self._aBl = None
        self._dl = None
        self._cache_alphal = None
        self._durations = None
        self._stateseq_norep = None
        self.state_ranges = None

    @property
    def stateseq_norep(self):
        if self._stateseq_norep is None:
            self._stateseq_norep, dur = rle(self.stateseq)

        return self._stateseq_norep

    @property
    def durations(self):
        if self._durations is None:
            self._letterseq_norep, self._durations = rle(self.letterseq)

        return self._durations

    @property
    def aDl(self):
        if self._aDl is None:
            aDl = self._aDl = np.empty((self.T, self.model.state_dim))
            possible_durations = np.arange(1, self.T + 1, dtype = np.float64)
            for idx, dist in enumerate(self.model.word_dur_dists):
                aDl[:, idx] = dist.log_likelihood(possible_durations)

        return self._aDl

    @property
    def aD(self):
        return np.exp(self.aDl)

    @property
    def A(self):
        return self.model.trans_dists.A

    @property
    def Al(self):
        if self._Al is None:
            self._Al = np.log(self.model.trans_dists.A)

        return self._Al

    @property
    def aBl(self):
        if self._aBl is None:
            self._aBl = aBl = np.empty((self.T, self.model.letter_dim))
            for idx, dist in enumerate(self.model.obs_distns):
                aBl[:, idx] = np.nan_to_num(dist.log_likelihood(self.data))

        return self._aBl

    @property
    def dl(self):
        if self._dl is None:
            self._dl = dl = np.empty((self.T, self.model.letter_dim))
            possible_durations = np.arange(1, self.T + 1, dtype = np.float64)
            for idx, dist in enumerate(self.model.dur_distns):
                dl[:, idx] = dist.log_likelihood(possible_durations)

        return self._dl

    def resample(self):
        self.clear_caches()
        betal, betastarl = self.messages_backwards()
        self.sample_forwards(betal, betastarl)

    def messages_backwards(self, trunc = 60):
        Al = self.Al
        aDl = self.aDl
        state_dim = self.model.state_dim

        self.betal = betal = np.zeros((self.T, state_dim), dtype = np.float64)
        self.betastarl = betastarl = np.zeros((self.T, state_dim), dtype = np.float64)
        T = self.T

        for t in range(T - 1, -1, -1):
            betastarl[t] = np.logaddexp.reduce(
                betal[t:t+trunc] + self.cumulative_likelihoods(t, t + trunc) + aDl[:min(trunc, T-t)],
                axis = 0
                )
            betal[t-1] = np.logaddexp.reduce(betastarl[t] + Al, axis = 1)

        betal[-1] = 0.0
        return betal, betastarl

    def sample_forwards(self, betal, betastarl):
        T = self.T
        A = self.A
        aD = self.aD
        stateseq = self.stateseq = np.zeros(T, dtype = np.int32)
        state_ranges = self.state_ranges = []
        idx = 0
        nextstate_unsmoothed = self.model.init_dist.pi_0
        while idx < T:
            logdomain = betastarl[idx] - np.amax(betastarl[idx])
            nextstate_dist = np.exp(logdomain) * nextstate_unsmoothed
            if (nextstate_dist == 0.).all():
                nextstate_dist = np.exp(logdomain)

            state = sample_discrete(nextstate_dist)
            durprob = np.random.random()
            word = self.model.word_list[state]
            dur = len(word) - 1

            self.likelihood_block_word(idx, T, word)
            cache_loglikelihood = self._cache_alphal[:, -1]
            cache_mess_term = np.exp(cache_loglikelihood + betal[idx:T, state] - betastarl[idx, state])

            while durprob > 0:
                p_d_prior = aD[dur, state] if dur < T else 1.
                assert not np.isnan(p_d_prior)
                assert p_d_prior >= 0

                if p_d_prior == 0:
                    dur += 1
                    continue

                if idx + dur < T:
                    p_d = cache_mess_term[dur] * p_d_prior
                    assert not np.isnan(p_d)
                    durprob -= p_d
                    dur += 1
                else:
                    dur += 1
                    break

            assert dur > 0
            assert dur >= len(word)
            stateseq[idx:idx+dur] = state
            state_ranges.append((state, (idx, idx + dur)))
            nextstate_unsmoothed = A[state]
            idx += dur


    def cumulative_likelihoods(self, start, stop):
        T = min(self.T, stop)
        tsize = T - start
        ret = np.ones((tsize, self.model.state_dim)) * -np.inf

        for state, word in enumerate(self.model.word_list):
            self.likelihood_block_word(start, stop, word)
            alphal = self._cache_alphal
            ret[:, state] = alphal[:, -1]

        return ret

    def likelihood_block_word(self, start, stop, word):
        T = min(self.T, stop)
        tsize = T - start
        aBl = self.aBl
        dl = self.dl
        rev_dl = dl[::-1]
        len_word = len(word)
        self._cache_alphal = alphal = np.ones((tsize, len_word)) * -np.inf

        if tsize-len_word+1 <= 0:
            return alphal[-1, -1]

        cumsum_aBl = np.empty(tsize-len_word+1)
        alphal[:tsize-len_word+1, 0] = np.cumsum(aBl[start:start+tsize-len_word+1, word[0]]) + dl[:tsize-len_word+1, word[0]]
        cache_range = range(tsize - len_word + 1)
        for j, l in enumerate(word[1:]):
            cumsum_aBl[:] = 0.0
            for t in cache_range:
                cumsum_aBl[:t+1] += aBl[start+t+j+1, l]
                alphal[t+j+1, j+1] = np.logaddexp.reduce(cumsum_aBl[:t+1] + rev_dl[-t-1:, l] + alphal[j:t+j+1, j])
        return alphal[-1, -1]
