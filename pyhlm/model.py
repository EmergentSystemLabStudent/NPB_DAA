import numpy as np

from pyhsmm.util.general import list_split
from pyhsmm.util.stats import sample_discrete_from_log
from pyhsmm.internals.transitions import WeakLimitHDPHMMTransitions
from pyhsmm.internals.initial_state import HMMInitialState

from pyhlm.internals import hlm_states

class WeakLimitHDPHLMPython(object):
    _states_class = hlm_states.WeakLimitHDPHLMStatesPython

    def __init__(self, num_states, alpha, gamma, init_state_concentration, letter_hsmm, dur_distns, length_distn):
        self._letter_hsmm = letter_hsmm
        self._length_distn = length_distn#Poisson(alpha_0=30, beta_0=10)
        self._dur_distns = dur_distns
        self._num_states = num_states
        self._letter_num_states = letter_hsmm.num_states
        self._init_state_distn = HMMInitialState(self, init_state_concentration=init_state_concentration)
        self._trans_distn = WeakLimitHDPHMMTransitions(num_states=num_states, alpha=alpha, gamma=gamma)
        self.states_list = []

        self.word_list = [None] * self.num_states
        for i in range(self.num_states):
            word = self.generate_word()
            while word in self.word_list[:i]:
                word = self.generate_word()
            self.word_list[i] = word
        self.resample_dur_distns()

    @property
    def num_states(self):
        return self._num_states

    @property
    def letter_num_states(self):
        return self._letter_num_states

    @property
    def letter_obs_distns(self):
        return self.letter_hsmm.obs_distns

    @property
    def dur_distns(self):
        return self._dur_distns

    @property
    def letter_dur_distns(self):
        return self.letter_hsmm.dur_distns

    @property
    def init_state_distn(self):
        return self._init_state_distn

    @property
    def trans_distn(self):
        return self._trans_distn

    @property
    def length_distn(self):
        return self._length_distn

    @property
    def letter_hsmm(self):
        return self._letter_hsmm

    @property
    def params(self):
        letter_hsmm_params = self.letter_hsmm.params
        bigram_params = {**self.init_state_distn.params, "trans_matrix": self.trans_distn.trans_matrix}
        length_params = self.length_distn.params
        word_dicts = {f"word({i})": np.array(word) for i, word in enumerate(self.word_list)}
        return {"num_states": self.num_states, "word_dicts": word_dicts, "letter_hsmm": letter_hsmm_params, "word_length": length_params, "bigram": bigram_params}

    @property
    def hypparams(self):
        letter_hsmm_hypparams = self.letter_hsmm.hypparams
        bigram_hypparams = self.init_state_distn.hypparams
        length_hypparams = self.length_distn.hypparams
        return {"letter_hsmm": letter_hsmm_hypparams, "word_length": length_hypparams, "bigram": bigram_hypparams}

    def log_likelihood(self):
        return sum(word_state.log_likelihood() for word_state in self.states_list)

    def word_counts(self):
        r = np.zeros(self.num_states, dtype=np.int32)
        for s in self.states_list:
            for i in s.stateseq_norep:
                r[i] += 1
        return r

    def generate_word(self):
        size = self.length_distn.rvs() or 1
        return self.letter_hsmm.generate_word(size)

    def add_data(self, data, **kwargs):
        self.states_list.append(self._states_class(self, data, **kwargs))

    def add_word_data(self, data, **kwargs):
        self.letter_hsmm.add_data(data, **kwargs)

    def resample_model(self, num_procs=0):
        self.letter_hsmm.states_list = []
        [state.add_word_datas(generate=False) for state in self.states_list]
        self.letter_hsmm.resample_states(num_procs=num_procs)
        [letter_state.reflect_letter_stateseq() for letter_state in self.letter_hsmm.states_list]
        self.resample_words(num_procs=num_procs)
        self.letter_hsmm.resample_parameters_by_sampled_words(self.word_list)
        self.resample_length_distn()
        self.resample_dur_distns()
        self.resample_trans_distn()
        self.resample_init_state_distn()
        self.resample_states(num_procs=num_procs)
        self._clear_caches()

    def resample_states(self, num_procs=0):
        if num_procs == 0:
            for state in self.states_list:
                state.resample()
        else:
            self._joblib_resample_states(self.states_list, num_procs)

    def _joblib_resample_states(self, states_list, num_procs):
        from joblib import Parallel, delayed
        from . import parallel

        # warn('joblib is segfaulting on OS X only, not sure why')

        if len(states_list) > 0:
            joblib_args = list_split(
                    [self._get_joblib_pair(s) for s in states_list],
                    num_procs)

            parallel.model = self
            parallel.args = joblib_args

            raw_stateseqs = Parallel(n_jobs=num_procs,backend='multiprocessing')\
                    (delayed(parallel._get_sampled_stateseq_norep_and_durations_censored)(idx)
                            for idx in range(len(joblib_args)))

            for s, (stateseq, stateseq_norep, durations_censored, log_likelihood) in zip(
                    [s for grp in list_split(states_list,num_procs) for s in grp],
                    [seq for grp in raw_stateseqs for seq in grp]):
                s.stateseq, s._stateseq_norep, s._durations_censored, s._normalizer = stateseq, stateseq_norep, durations_censored, log_likelihood

    def _get_joblib_pair(self,states_obj):
        return (states_obj.data, states_obj._kwargs)

    def resample_words(self, num_procs=0):
        if num_procs == 0:
            self.word_list = [self._resample_a_word(
                [letter_state for letter_state in self.letter_hsmm.states_list if letter_state.word_idx == word_idx]
            ) for word_idx in range(self.num_states)]
        else:
            from joblib import Parallel, delayed
            self.word_list = Parallel(n_jobs=num_procs, backend='multiprocessing')\
                ([delayed(self._resample_a_word)(
                    [letter_state for letter_state in self.letter_hsmm.states_list if letter_state.word_idx == word_idx]
                    ) for word_idx in range(self.num_states)]
                )
        # Merge same letter seq which has different id.
        for i, word in enumerate(self.word_list):
            if word in self.word_list[:i]:
                existed_id = self.word_list[:i].index(word)
                for word_state in self.states_list:
                    stateseq, stateseq_norep = word_state.stateseq, word_state.stateseq_norep
                    word_state.stateseq[stateseq == i] = existed_id
                    word_state.stateseq_norep[stateseq_norep == i] = existed_id
                    word_candi = self.generate_word()
                    while word_candi in self.word_list:
                        word_candi = self.generate_word()
                    self.word_list[i] = word_candi

    def _resample_a_word(self, hsmm_states):
        # hsmm_states = [letter_state for letter_state in self.letter_hsmm.states_list if letter_state.word_idx == word_idx]
        candidates = [tuple(letter_state.stateseq_norep) for letter_state in hsmm_states]
        unique_candidates = list(set(candidates))
        ref_array = np.array([unique_candidates.index(candi) for candi in candidates])
        if len(candidates) == 0:
            return self.generate_word()
        elif len(unique_candidates) == 1:
            return unique_candidates[0]
        cache_score = np.empty((len(unique_candidates), len(candidates)))
        likelihoods = np.array([letter_state.log_likelihood() for letter_state in hsmm_states])
        range_tmp = list(range(len(candidates)))

        for candi_idx, candi in enumerate(unique_candidates):
            tmp = range_tmp[:]
            if (ref_array == candi_idx).sum() == 1:
                tmp.remove(np.where(ref_array == candi_idx)[0][0])
            for tmp_idx in tmp:
                # print(hsmm_states[tmp_idx].likelihood_block_word(candi)[-1])
                cache_score[candi_idx, tmp_idx] = hsmm_states[tmp_idx].likelihood_block_word(candi)[-1]
        cache_scores_matrix = cache_score[ref_array]
        for i in range_tmp:
            cache_scores_matrix[i, i] = 0.0
        scores = cache_scores_matrix.sum(axis=1) + likelihoods

        sampled_candi_idx = sample_discrete_from_log(scores)
        return candidates[sampled_candi_idx]

    def resample_length_distn(self):
        self.length_distn.resample(np.array([len(word) for word in self.word_list]))

    def resample_dur_distns(self):#Do not resample!! This code only update the parameter of duration distribution of word.
        letter_lmbdas = np.array([letter_dur_distn.lmbda for letter_dur_distn in self.letter_dur_distns])
        for word, dur_distn in zip(self.word_list, self.dur_distns):
            dur_distn.lmbda = np.sum(letter_lmbdas[list(word)])

    def resample_trans_distn(self):
        self.trans_distn.resample([word_state.stateseq_norep for word_state in self.states_list])

    def resample_init_state_distn(self):
        self.init_state_distn.resample(np.array([word_state.stateseq_norep[0] for word_state in self.states_list]))

    def _clear_caches(self):
        for word_state in self.states_list:
            word_state.clear_caches()

class WeakLimitHDPHLM(WeakLimitHDPHLMPython):
    _states_class = hlm_states.WeakLimitHDPHLMStates
