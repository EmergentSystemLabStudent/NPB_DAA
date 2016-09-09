from __future__ import division
import numpy as np

from pyhsmm import models as m, distributions as d

######################
#  likelihood tests  #
######################

def _random_variant_model():
    N = 4
    obs_dim = 2

    obs_hypparams = {'mu_0':np.zeros(obs_dim),
                    'sigma_0':np.eye(obs_dim),
                    'kappa_0':0.05,
                    'nu_0':obs_dim+5}

    obs_distns = [d.Gaussian(**obs_hypparams) for state in range(N)]

    dur_distns = \
            [d.NegativeBinomialIntegerRVariantDuration(
                np.r_[0,0,0,0,0,1.,1.,1.], # discrete distribution uniform over {6,7,8}
                alpha_0=9,beta_0=1, # average geometric success probability 1/(9+1)
                ) for state in range(N)]

    model  = m.HSMMIntNegBinVariant(
            init_state_concentration=10.,
            alpha=6.,gamma=6.,
            obs_distns=obs_distns,
            dur_distns=dur_distns)

    return model


def in_out_comparison_test():
    for i in range(2):
        yield _hmm_in_out_comparison_helper

def _hmm_in_out_comparison_helper():
    model = _random_variant_model()
    data, _ = model.generate(1000)

    like1 = model.log_likelihood()
    model.states_list = []
    like2 = model.log_likelihood(data)
    model.add_data(data)
    like3 = model.log_likelihood()

    assert np.isclose(like1,like2) and np.isclose(like2,like3)


def hmm_message_comparison_test():
    for i in range(2):
        yield _hmm_message_comparison_helper

def _hmm_message_comparison_helper():
    model = _random_variant_model()

    data, _ = model.generate(1000)

    likelihood_hsmmintnegbin_messages = model.log_likelihood(data)

    s = model.states_list[0]
    s.messages_backwards = None
    likelihood_hmm_messages = np.logaddexp.reduce(
            np.log(s.pi_0) + s.messages_backwards_hmm()[0] + s.aBl[0])

    assert np.isclose(likelihood_hmm_messages, likelihood_hsmmintnegbin_messages)

