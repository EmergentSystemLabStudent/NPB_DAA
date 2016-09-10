import numpy as np
from HDP_HSMM.models import HSMMEigen
from HDP_HSMM.internals.states import HSMMStatesEigen
from HDP_HSMM.basic.util import sample_discrete


#########################################################
#  Letter states class  #
#########################################################
class LetterStates(HSMMStatesEigen):
    def likelihood_block_word(self, start, stop, word):
        T = min(self.T, stop)
        tsize = T - start
        aBl = self.aBl
        aDl = self.aDl
        len_word = len(word)
        self._cache_alphal = alphal = np.ones((tsize, len_word)) * -np.inf
        for j, l in enumerate(word):
            for t in range(tsize - len_word + j +1):
                if j == 0:
                    alphal[t, j] = np.sum(aBl[start:start+t+1, l]) + aDl[t, l]
                elif j > t:
                    alphal[t, j] = -np.inf
                    continue
                else:
                    alphal[t, j] = np.logaddexp.reduce([
                        np.sum(aBl[start+t+1-d:start+t+1, l]) + \
                        aDl[d, l] + \
                        alphal[t - d, j - 1]
                        for d  in range(1, t + 1)
                    ])
        return alphal[-1, -1]


#########################################################
#  Word model class  #
#########################################################
class WordHSMMModel(HSMMEigen):
    _states_class = LetterStates
    def likelihoods(self):
        messages = np.vstack([s.messages_backwards()[1][0] + np.log(s.pi_0) for s in self.states_list])
        return np.logaddexp.reduce(messages, axis = 1)
    def generate_word(self, size):
        next_dist = self.init_state_distn.pi_0
        word = []
        for _ in range(size):
            letter = sample_discrete(next_dist)
            word.append(letter)
            next_dist = self.trans_distn.A[letter]
        return tuple(word)
