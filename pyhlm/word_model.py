import numpy as np

from pyhsmm.models import WeakLimitHDPHSMMPython
from pyhsmm.models import WeakLimitHDPHSMM
from pyhsmm.internals.hsmm_states import HSMMStatesPython
from pyhsmm.internals.hsmm_states import HSMMStatesEigen
from pybasicbayes.distributions.poisson import Poisson
from pyhsmm.util.stats import sample_discrete


class LetterHSMMStatesPython(HSMMStatesPython):

    def __init__(self, model, hlmstate=None, word_idx=-1, d0=-1, d1=-1, **kwargs):
        self._hlmstate = hlmstate
        self._word_idx = word_idx
        self._d0 = d0
        self._d1 = d1
        super(LetterHSMMStatesPython, self).__init__(model, **kwargs)

    @property
    def word_idx(self):
        return self._word_idx

    def likelihood_block_word(self, word):
        from pyhlm.internals.hlm_states import hlm_internal_hsmm_messages_forwards_log
        T = self.T
        aBl = self.aBl
        alDl = self.aDl
        L = len(word)
        alphal = np.empty((T, L), dtype=np.float64)

        return hlm_internal_hsmm_messages_forwards_log(aBl, alDl, word, alphal)[:, -1]

    def reflect_letter_stateseq(self):
        if self._hlmstate is not None:
            self._hlmstate.letter_stateseq[self._d0:self._d1] = self.stateseq

    def sample_forwards(self,betal,betastarl):
        from pyhsmm.internals.hsmm_messages_interface import sample_forwards_log
        if self.left_censoring:
            raise NotImplementedError
        caBl = np.vstack((np.zeros(betal.shape[1]), np.cumsum(self.aBl[:-1],axis=0)))
        self.stateseq = sample_forwards_log(
                self.trans_matrix, caBl, self.aDl, self.pi_0, betal, betastarl,
                np.empty(betal.shape[0],dtype='int32'))
        # assert not (0 == self.stateseq).all() #Remove this assertion.


class LetterHSMMStatesEigen(HSMMStatesEigen, LetterHSMMStatesPython):

    def likelihood_block_word(self, word):
        from pyhlm.internals.internal_hsmm_messages_interface import internal_hsmm_messages_forwards_log
        T = self.T
        aBl = self.aBl
        alDl = self.aDl
        L = len(word)
        alphal = np.ones((T, L), dtype=np.float64) * -np.inf

        if T - L + 1 <= 0:
            return alphal[:, -1]

        return internal_hsmm_messages_forwards_log(aBl, alDl, np.array(word, dtype=np.int32), alphal)[:, -1]

    def sample_forwards(self,betal,betastarl):
        from pyhsmm.internals.hsmm_messages_interface import sample_forwards_log
        if self.left_censoring:
            raise NotImplementedError
        caBl = np.vstack((np.zeros(betal.shape[1]), np.cumsum(self.aBl[:-1],axis=0)))
        self.stateseq = sample_forwards_log(
                self.trans_matrix, caBl, self.aDl, self.pi_0, betal, betastarl,
                np.empty(betal.shape[0],dtype='int32'))
        # assert not (0 == self.stateseq).all() #Remove this assertion.

    def likelihood_block_word_python(self, word):
        return super(LetterHSMMStatesEigen, self).likelihood_block_word(word)

class LetterHSMMPython(WeakLimitHDPHSMMPython):
    _states_class = LetterHSMMStatesPython

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
