#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from pyhsmm.basic.distributions import PoissonDuration
from pyhsmm.internals.transitions import HDPHSMMTransitions, HDPHMMTransitions
from pyhsmm.internals.initial_state import InitialState
from pyhsmm.util import general
from pyhsmm.util.stats import sample_discrete
from states import HSMMState
import time

class DAHSMM(object):
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
        from pyhsmm import parallel
        self.states_list.append(HSMMState(data, self))
        #parallel=true nara koreganaito keyerror ninaru
        if self.parallel:
            parallel.add_data(self.states_list[-1].data)

    def resample_model(self):
        times = [0.0, 0.0, 0.0, 0.0, 0.0]
        if self.parallel:
            st = time.time()
            self.resample_states_parallel()
            times[0] = time.time() - st
        else:
            self.resample_states()

        st = time.time()
        self.resample_letter_params()
        times[1] = time.time() - st
        st = time.time()
        self.resample_dur_dists()
        times[2] = time.time() - st
        st = time.time()
        self.resample_trans_dist()
        times[3] = time.time() - st
        st = time.time()
        self.resample_init_dist()
        times[4] = time.time() - st

        #print time stamps.
        print "TS_resample:",str(times)[1:-1]


    def resample_trans_dist(self):
        self.trans_dists.resample(np.array([[state for (state, _) in s.state_ranges] for s in self.states_list]))

    def resample_init_dist(self):
        self.init_dist.resample([s.stateseq[:1] for s in self.states_list])

    def resample_states(self):
        [s.resample() for s in self.states_list]

    def resample_states_parallel(self):
        from pyhsmm import parallel
        states = self.states_list
        self.states_list = []
        raw = parallel.map_on_each(
            self._states_sampler,
            [s.data for s in states],
            kwargss = self._get_parallel_kwargss(states),
            engine_globals = dict(global_model = self)
        )
        self.states_list = states
        # ここでうけとってるんだけどこれだと状態と範囲？しかとれないから
        # for s1, ret in zip(self.states_list, raw):
        #     s1.stateseq, s1.state_ranges = ret
        for s1, ret in zip(self.states_list, raw):
            s1.stateseq = ret.stateseq
            s1.state_ranges = ret.state_ranges
            s1.betal = ret.betal
            # うけとりたいものを追加するでいけると思う

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
        likelihoods = np.array(hsmm.likelihoods())

        #parallelable procedure
        for state, bound in enumerate(zip(states_index[:-1], states_index[1:])):
            staff = range(*bound)
            print "word",state,"has",len(staff),"segments."

            if len(staff) == 0:
                self.word_list[state] = self.generate_word()
                continue
            elif len(staff) == 1:
                self.word_list[state] = tuple(hsmm.states_list[staff[0]].stateseq_norep)
                s = hsmm.states_list[staff[0]]
                s.letterseq[:] = s.stateseq
                continue

            words_list = [tuple(hsmm.states_list[s].stateseq_norep) for s in staff]
            words_unique = list(set(words_list))
            ref_array = [words_unique.index(word) for word in words_list]

            print "word",state,".   unique/all =",len(words_unique),"/",len(words_list)

            if len(words_unique) == 1:
                self.word_list[state] = words_unique[0]
                for idx in staff:
                    s = hsmm.states_list[idx]
                    s.letterseq[:] = s.stateseq
                continue

            candidates = words_list
            cache_scores = np.empty((len(words_unique), len(words_list)))

            #calculate likelihood
            st = time.time()
            for i, word in enumerate(words_unique):
                #parallelable procedure
                for j, s in enumerate(staff):
                    cache_scores[i, j] = hsmm.states_list[s].likelihood_block_word(0, len(hsmm.states_list[s].data), word)
            en = time.time()
            print "word",state,".   importance calc:",(en - st)
            cache_scores_matrix = cache_scores[ref_array]
            for i in range(len(staff)):
                cache_scores_matrix[i,i] = 0.0
            scores = np.sum(cache_scores_matrix, axis=1) + likelihoods[staff[0]:staff[-1]+1]

            word_idx = sample_discrete(np.exp(scores))
            sampleseq = candidates[word_idx]

            self.word_list[state] = tuple(sampleseq)
            for idx in staff:
                s = hsmm.states_list[idx]
                s.letterseq[:] = s.stateseq
                #word = tuple(s.stateseq_norep)

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

    @staticmethod
    @general.engine_global_namespace
    def _states_sampler(data):
        global_model.add_data(data = data)
        model = global_model.states_list.pop()
        # ここで計算結果のモデルを取得してる
        # 返してるのは状態列と何回?続いたかだったかな?
        # return model.stateseq, model.state_ranges
        return model


    def _get_parallel_kwargss(self,states_objs):
        # this method is broken out so that it can be overridden
        return [{}]*len(states_objs)
