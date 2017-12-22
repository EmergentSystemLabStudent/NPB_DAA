from pyhsmm.models import HSMMEigen
from pyhsmm.internals.transitions import HDPHSMMTransitions
from pyhsmm.basic.distributions import PoissonDuration
from pyhsmm.util.stats import sample_discrete
from pyhsmm.internals.initial_state import InitialState
from pyhsmm.internals.states import HSMMStatesEigen
import numpy as np

class LLStates(HSMMStatesEigen):
    def likelihood_block_word(self, start, stop, word):
        T = min(self.T, stop)
        tsize = T - start
        aBl = self.aBl
        aDl = self.aDl
        rev_aDl = aDl[::-1]
        len_word = len(word)
        self._cache_alphal = alphal = np.ones((tsize, len_word)) * -np.inf

        if tsize-len_word+1 <= 0:
            return alphal[-1, -1]

        cumsum_aBl = np.empty(tsize-len_word+1)
        alphal[:tsize-len_word+1, 0] = np.cumsum(aBl[start:start+tsize-len_word+1, word[0]]) + aDl[:tsize-len_word+1, word[0]]
        cache_range = range(tsize - len_word + 1)
        for j, l in enumerate(word[1:]):
            cumsum_aBl[:] = 0.0
            for t in cache_range:
                cumsum_aBl[:t+1] += aBl[start+t+j+1, l]
                alphal[t+j+1, j+1] = np.logaddexp.reduce(cumsum_aBl[:t+1] + rev_aDl[-t-1:, l] + alphal[:t+1, j])
        return alphal[-1, -1]


class LLHSMM(HSMMEigen):
    _states_class = LLStates

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


class WordTransitions(HDPHSMMTransitions):
    def __init__(self, state_dim, alpha, gamma, beta = None, A = None, fullA = None):
        super(WordTransitions, self).__init__(state_dim, alpha, gamma, beta, A, fullA)

    def resample(self, states_noreps = []):
        augmented_data = self._augment_data(self._count_transitions(states_noreps))
        m = self._get_m(augmented_data)

        self._resample_beta(m)
        self._resample_A(augmented_data)

class WordModel(object):
    def __init__(self, letter_type, rho = 1.0, alpha = 1.0, gamma = 1.0, lmbda = 4.0):
        self.letter_dim = letter_type
        self.init_dist = InitialState(state_dim = letter_type, rho = rho)
        self.letter_trans = WordTransitions(state_dim = letter_type, alpha = alpha, gamma = gamma)
        self.letter_dur = PoissonDuration(lmbda = lmbda)

    def resample(self, data):
        self.letter_trans.resample(data)
        self.init_dist.resample([word[:1] for word in data])

    def generate(self):
        word_size = self.letter_dur.rvs() or 1
        next_state_dist = self.init_dist.pi_0
        ret = []

        for i in range(word_size):
            next_state = sample_discrete(next_state_dist)
            ret.append(next_state)
            next_state_dist = self.letter_trans.A[next_state]

        return tuple(ret)
