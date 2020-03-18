import numpy as np

from pyhsmm.models import WeakLimitHDPHSMMPython
from pyhsmm.models import WeakLimitHDPHSMM
from pybasicbayes.distributions.poisson import Poisson
from pyhsmm.util.stats import sample_discrete
from pyhlm.internals.internal_hsmm_states import LetterHSMMStatesPython, LetterHSMMStatesEigen

class LetterHSMMPython(WeakLimitHDPHSMMPython):
    _states_class = LetterHSMMStatesPython

    def resample_trans_distn_by_sampled_words(self, word_list):
        self.trans_distn.resample([np.array(word) for word in word_list])
        self._clear_caches()

    def resample_init_state_distn_by_sampled_words(self, word_list):
        self.init_state_distn.resample([word[0] for word in word_list])
        self._clear_caches()

    def resample_parameters_by_sampled_words(self, word_list):
        self.resample_dur_distns()
        self.resample_obs_distns()
        self.resample_trans_distn_by_sampled_words(word_list)
        self.resample_init_state_distn_by_sampled_words(word_list)

    def generate_word(self, word_size):
        nextstate_distn = self.init_state_distn.pi_0
        A = self.trans_distn.trans_matrix
        word = [-1] * word_size
        for idx in range(word_size):
            word[idx] = sample_discrete(nextstate_distn)
            nextstate_distn = A[word[idx]]
        return tuple(word)

    @property
    def params(self):
        obs_params = {"obs_distn({})".format(idx): obs_distn.params for idx, obs_distn in enumerate(self.obs_distns)}
        dur_params = {"dur_distn({})".format(idx): dur_distn.params for idx, dur_distn in enumerate(self.dur_distns)}
        bigram_params = {**self.init_state_distn.params, "trans_matrix":self.trans_distn.trans_matrix}
        return {"num_states": self.num_states, "obs_distns": obs_params, "dur_distns": dur_params, "bigram": bigram_params}

    @property
    def hypparams(self):
        obs_hypparams = {"obs_distn({})".format(idx): obs_distn.hypparams for idx, obs_distn in enumerate(self.obs_distns)}
        dur_hypparams = {"dur_distn({})".format(idx): dur_distn.hypparams for idx, dur_distn in enumerate(self.dur_distns)}
        bigram_hypparams = self.init_state_distn.hypparams
        return {"obs_distns": obs_hypparams, "dur_distns": dur_hypparams, "bigram": bigram_hypparams}

class LetterHSMM(WeakLimitHDPHSMM, LetterHSMMPython):
    _states_class = LetterHSMMStatesEigen
