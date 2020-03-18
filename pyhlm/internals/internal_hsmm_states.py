import numpy as np

from pyhsmm.internals.hsmm_states import HSMMStatesPython
from pyhsmm.internals.hsmm_states import HSMMStatesEigen

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
        caBl = np.vstack((np.zeros(self.num_states),np.cumsum(self.aBl,axis=0)))
        self.stateseq = sample_forwards_log(
                self.trans_matrix, caBl, self.aDl, self.pi_0, betal, betastarl,
                np.empty(betal.shape[0],dtype='int32'))
        # assert not (0 == self.stateseq).all() #Remove this assertion.

    def likelihood_block_word_python(self, word):
        return super(LetterHSMMStatesEigen, self).likelihood_block_word(word)
