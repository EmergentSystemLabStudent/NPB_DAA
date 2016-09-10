#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from multiprocessing import Process, Queue
from HDP_HSMM.basic.distributions import PoissonDuration
from HDP_HSMM.internals.transitions import HDPHMMTransitions
from HDP_HSMM.internals.states import InitialState
from HDP_HSMM.basic.util import sample_discrete, rle, engine_global_namespace


#########################################################
#  Word states class  #
#########################################################
class WordStates:
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
            while durprob > 0:
                p_d_prior = aD[dur, state] if dur < T else 1.
                assert not np.isnan(p_d_prior)
                assert p_d_prior >= 0
                if p_d_prior == 0:
                    dur += 1
                    continue
                if idx + dur < T:
                    loglikelihood = self.likelihood_block_word(idx, idx+dur+1, word)
                    mess_term = np.exp(loglikelihood + betal[idx+dur, state] - betastarl[idx, state])
                    p_d = mess_term * p_d_prior
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
        ret = np.zeros((tsize, self.model.state_dim))
        for state, word in enumerate(self.model.word_list):
            self.likelihood_block_word(start, stop, word)
            alphal = self._cache_alphal
            ret[:, state] = alphal[:, -1]
        return ret
    def likelihood_block_word(self, start, stop, word):
        T = min(self.T, stop)
        tsize = T - start
        aBl = self.aBl
        len_word = len(word)
        self._cache_alphal = alphal = np.ones((tsize, len_word)) * -np.inf
        for j, l in enumerate(word):
            for t in range(j, tsize - len_word + j + 1):
                if j == 0:
                    alphal[t, j] = np.sum(aBl[start:start+t+1, l]) + self.dl[t, l]
                else:
                    alphal[t, j] = np.logaddexp.reduce([
                        np.sum(aBl[start+t-d:start+t+1, l]) + \
                        self.dl[d, l] + \
                        alphal[t - d - 1, j - 1]
                        for d  in range(t + 1)
                    ])
        return alphal[-1, -1]


#########################################################
#  Language model class  #
#########################################################
class LanguageHSMMModel(object):
    def __init__(self, hypparams, letter_hsmm, length_dist, obs_dists, dur_dists, parallel = True):
        self.trans_dists = HDPHMMTransitions(**hypparams)
        self.letter_hsmm = letter_hsmm
        self.length_dist = length_dist
        self.state_dim = hypparams['state_dim']
        self.letter_dim = len(obs_dists)
        self.init_dist = InitialState(state_dim = self.state_dim, rho = 1.0)
        self.states_list = []
        self.parallel = parallel
        word_set = set()
        while len(word_set) < self.state_dim:
            word = self.generate_word()
            word_set.add(word)
        self.word_list = list(word_set)
        self.resample_dur_dists()
    @property
    def obs_distns(self):
        return self.letter_hsmm.obs_distns
    @property
    def dur_distns(self):
        return self.letter_hsmm.dur_distns
    def generate_word(self):
        size = self.length_dist.rvs() or 1
        return self.letter_hsmm.generate_word(size)
    def generate(self, limit_len = 3):
        nextstate_dist = self.init_dist.pi_0
        A = self.trans_dists.A
        state_list = []
        for _ in range(limit_len):
            state = sample_discrete(nextstate_dist)
            state_list.append(state)
            nextstate_dist = A[state]
        stateseq = []
        letseq = []
        obsseq = []
        for s in state_list:
            for l in self.word_list[s]:
                d = self.dur_distns[l].rvs() or 1
                o = self.obs_distns[l].rvs(size = d)
                obsseq.append(o)
                letseq.append([l] * d)
                stateseq.append([s] * d)
        return map(np.concatenate, (stateseq, letseq, obsseq))
    def add_data(self, data):
        from HDP_HSMM import parallel
        self.states_list.append(WordStates(data, self))
        # parallel = true なら下記がないとエラーになる
        if self.parallel:
            parallel.add_data(self.states_list[-1].data)
    def resample_model(self):
        if self.parallel:
            self.resample_states_parallel()
        else:
            self.resample_states()
        self.resample_letter_params()
        self.resample_dur_dists()
        self.resample_trans_dist()
        self.resample_init_dist()
    def resample_trans_dist(self):
        self.trans_dists.resample(np.array([[state for (state, _) in s.state_ranges] for s in self.states_list]))
    def resample_init_dist(self):
        self.init_dist.resample([s.stateseq[:1] for s in self.states_list])
    def resample_states(self):
        [s.resample() for s in self.states_list]
    def resample_states_parallel(self):
        from HDP_HSMM import parallel
        states = self.states_list
        self.states_list = []
        raw = parallel.map_on_each(
            self._states_sampler,
            [s.data for s in states],
            kwargss = self._get_parallel_kwargss(states),
            engine_globals = dict(global_model = self)
        )
        self.states_list = states
        # 下記の例では状態ラベルと状態持続範囲を結果に受け取っている
        """
        for s1, ret in zip(self.states_list, raw):
            s1.stateseq, s1.state_ranges = ret
        """
        for s1, ret in zip(self.states_list, raw):
            s1.stateseq = ret.stateseq
            s1.state_ranges = ret.state_ranges
            s1.betal = ret.betal
    @staticmethod
    @engine_global_namespace
    def _states_sampler(data):
        global_model.add_data(data = data)
        model = global_model.states_list.pop()
        # 下記の例では計算結果のモデル中の状態ラベルと状態持続範囲を取得している
        """
        return model.stateseq, model.state_ranges
        """
        return model
    def resample_letter_params(self):
        states_index = [0]
        hsmm = self.letter_hsmm
        hsmm.states_list = []
        for s in self.states_list:
            s.letterseq = np.ones(len(s.data), dtype = np.int64) * -1
        for state in range(self.state_dim):
            for s in self.states_list:
                for state2, (start, stop) in s.state_ranges:
                    if state == state2:
                        hsmm.add_data_parallel(s.data[start:stop])
                        hsmm.states_list[-1].letterseq = s.letterseq[start:stop]
            states_index.append(len(hsmm.states_list))
        hsmm.resample_states_parallel()
        likelihoods = hsmm.likelihoods()
        state_count = {}
        for state, bound in enumerate(zip(states_index[:-1], states_index[1:])):
            staff = range(*bound)
            if len(staff) == 0:
                self.word_list[state] = self.generate_word()
                continue
            candidates = []
            scores = []
            for idx in staff:
                rest = set(staff) - set([idx])
                word = hsmm.states_list[idx].stateseq_norep
                ## parallelize: nakashima edit
                def multi_bw(hsmm,word,s,q):
                    q.put(hsmm.states_list[s].likelihood_block_word(0, len(hsmm.states_list[s].data), word))
                q = Queue()
                pr_l = []
                for s in rest:
                    pr = Process(target=multi_bw, args=(hsmm,word,s,q))
                    pr_l.append(pr)
                    pr.start()
                for p in pr_l:
                    p.join()
                q_l = [q.get() for i in range(len(pr_l))]
                score = np.sum(q_l) + likelihoods[idx]
                ## -------------------------------------
                """
                score = np.sum([hsmm.states_list[s].likelihood_block_word(0, len(hsmm.states_list[s].data), word) for s in rest]) + likelihoods[idx]
                """
                scores.append(score)
                candidates.append(tuple(word))
            resample_state_flag = len(set(candidates)) > 1
            if resample_state_flag:
                word_idx = sample_discrete(np.exp(scores))
                sampleseq = candidates[word_idx]
            else:
                sampleseq = candidates[0]
            self.word_list[state] = tuple(sampleseq)
            for idx in staff:
                s = hsmm.states_list[idx]
                s.letterseq[:] = s.stateseq
                word = tuple(s.stateseq_norep)
        hsmm.resample_trans_distn()
        hsmm.resample_init_state_distn()
        hsmm.resample_dur_distns()
        hsmm.resample_obs_distns()
        self.resample_length_dist()
    def resample_length_dist(self):
        self.length_dist.resample(np.array(map(len, self.word_list)))
    def resample_dur_dists(self):
        self.word_dur_dists = [
            PoissonDuration(lmbda = np.sum([self.dur_distns[c].lmbda for c in w]))
            for w in self.word_list
        ]
    def _get_parallel_kwargss(self,states_objs):
        # this method is broken out so that it can be overridden
        return [{}]*len(states_objs)
