import numpy as np
from HDP_HLM.language import LanguageHSMMModel
from HDP_HLM.word import WordHSMMModel
from HDP_HSMM.basic.distributions import ScalarGaussianNIX, PoissonDuration
from HDP_HSMM.basic.util import progprint_xrange
from HDP_HLM.util import get_result_dir, save_parameters_scalar, save_fig_title, Result
import glob, os
if "XDG_CACHE_HOME" in os.environ: del os.environ["XDG_CACHE_HOME"]


SAVE_PARAMS = '''ITER_N
LETTER_N
WORD_N
DATA_N
model_hypparams
word_model_params
obs_hypparams
dur_hypparams
filename'''.split('\n')

def main(result_dir = None):
    if result_dir == None:
        result_dir = get_result_dir(__file__)
    os.mkdir(result_dir)
    param_path = os.path.join(result_dir, 'parameter.json')
    fig_title_path = os.path.join(result_dir, 'fig_title.json')
    ITER_N = 100
    LETTER_N = 7
    WORD_N = 6
    DATA_N = 40
    model_hypparams = {'state_dim': WORD_N, 'alpha': 10.0, 'gamma': 10.0}
    word_model_params = {'letter_type': LETTER_N, 'rho': 10}
    obs_hypparams = {'mu_0': 0., 'sigmasq_0': 1.0, 'kappa_0': 0.01, 'nu_0': 1}
    dur_hypparams = {'alpha_0': 50.0, 'beta_0': 10.0}
    length_dist = PoissonDuration(alpha_0 = 30, beta_0 = 10, lmbda = 3)
    # setting
    obs_dists = [ScalarGaussianNIX(**obs_hypparams) for state in range(LETTER_N)]
    dur_dists = [PoissonDuration(**dur_hypparams) for state in range(LETTER_N)]
    letter_hsmm = WordHSMMModel(
        init_state_concentration = 10,
        alpha = 10.0,
        gamma = 10.0,
        dur_distns = dur_dists,
        obs_distns = obs_dists
    )
    model = LanguageHSMMModel(
        model_hypparams,
        letter_hsmm,
        length_dist,
        obs_dists,
        dur_dists
    )
    # adddata
    filename = []
    for f in glob.glob('./DATA/*.txt'):
        print f + " loading..."
        huhu=np.loadtxt(f)
        model.add_data(huhu[2])
        f = f.replace("./DATA/","")
        filename.append(f.replace(".txt",""))
    save_fig_title(fig_title_path, SAVE_PARAMS, locals())
    save_parameters_scalar(param_path, SAVE_PARAMS, locals())
    # estimation
    result = Result(result_dir, DATA_N)
    loglikelihood = []
    for idx in progprint_xrange(ITER_N, perline = 10):
        model.resample_model()
        loglikelihood.append(result.get_loglikelihood(model))
        result.save(model)
    result.save_loglikelihood(loglikelihood)

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--result-dir', default = None)
    args = parser.parse_args()
    main(**vars(args))
