import os
if "XDG_CACHE_HOME" in os.environ:
    del os.environ["XDG_CACHE_HOME"]
import pyhsmm
import pyhsmm.parallel
from pyhsmm import parallel
from pyhsmm.basic.pybasicbayes.distributions import Gaussian as Gaussian
from pyhsmm.util.text import progprint_xrange
from dahsmm.hsmm import DAHSMM
from dahsmm.word import LLHSMM
from dahsmm.util import get_result_dir, save_parameters, save_fig_title, Result
from dahsmm.states import HSMMState
import numpy as np
import glob
import multiprocessing
import pickle
import time
#--------------------------------------save paras--------------------------------------#
SAVE_PARAMS = '''ITER_N
LETTER_N
WORD_N
DATA_N
model_hypparams
word_model_params
obs_hypparams
dur_hypparams
filename'''.split('\n')
#--------------------------------------multi process function--------------------------------------#


def multi_dump_object(data, model, f, count):
    print f, " dumping..."
    fp = open("TMP/" + f + ".dump", 'w')
    pickle.dump(HSMMState(data, model), fp)
    fp.close()
    count.value = count.value + 1
    print f, 'dump finish-->count:', count.value
#--------------------------------------main function--------------------------------------#


def main(result_dir=None):
    if result_dir == None:
        result_dir = get_result_dir(__file__)
    os.mkdir(result_dir)
    param_path = os.path.join(result_dir, 'parameter.json')
    fig_title_path = os.path.join(result_dir, 'fig_title.json')
#initialize model params#
    # O(T*L_max*W_max^2*d_max^3)
    ITER_N = 10
    LETTER_N = 7
    WORD_N = 7
    DATA_N = 60
    obs_dim = 3
    model_hypparams = {'state_dim': WORD_N, 'alpha': 10.0, 'gamma': 10.0}
    word_model_params = {'letter_type': LETTER_N, 'rho': 10}
    obs_hypparams = {'mu_0': np.zeros(obs_dim), 'sigma_0': np.eye(
        obs_dim), 'kappa_0': 0.01, 'nu_0': obs_dim + 5}
    dur_hypparams = {'alpha_0': 50.0, 'beta_0': 10.0}
    length_dist = pyhsmm.distributions.PoissonDuration(
        alpha_0=30, beta_0=10, lmbda=3)
#setting#
    obs_dists = [Gaussian(**obs_hypparams) for state in range(LETTER_N)]
    dur_dists = [pyhsmm.distributions.PoissonDuration(
        **dur_hypparams) for state in range(LETTER_N)]
    letter_hsmm = LLHSMM(
        init_state_concentration=10,
        alpha=10.0,
        gamma=10.0,
        dur_distns=dur_dists,
        obs_distns=obs_dists
    )
    model = DAHSMM(
        model_hypparams,
        letter_hsmm,
        length_dist,
        obs_dists,
        dur_dists
    )
#dump_object in multi engines#
    #"""
    print "--------------------------------------dump process start--------------------------------------"
    count = multiprocessing.Value('i', 0)
    datatxt_names = glob.glob('./DATA/*.txt')
    datatxt_names.sort()
    for f in datatxt_names:
        input_mat = np.loadtxt(f)
        f = f.replace("./DATA/", "")
        f = f.replace(".txt", "")
        pr = multiprocessing.Process(
            target=multi_dump_object, args=(input_mat, model, f, count))
        pr.start()
        time.sleep(0.1)
    while (1):
        if count.value > 55:
            time.sleep(1)
            print "--------------------------------------dump process completed!!--------------------------------------"
            break
    #"""
#add_data in one engine#
    """
    filename = []
    for f in glob.glob('./DATA/*.txt'):
        model.add_data(np.loadtxt(f))
        f = f.replace("./DATA/","")
        filename.append(f.replace(".txt",""))
    """
#add_data in multi engines#
    print "--------------------------------------add_data process start--------------------------------------"
    filename = []
    datadmp_names = glob.glob('./TMP/*.dump')
    datadmp_names.sort()
    for f in datadmp_names:
        print f, " loading..."
        fp = open(f)
        f = f.replace("./TMP/", "")
        f = f.replace(".dump", "")
        filename.append(f)
        obj = pickle.load(fp)
        model.states_list.append(obj)
        if model.parallel:
            parallel.add_data(model.states_list[-1].data)
        fp.close()
    print "--------------------------------------add_data process completed!!--------------------------------------"
#save params&charm#
    save_fig_title(fig_title_path, SAVE_PARAMS, locals())
    save_parameters(param_path, SAVE_PARAMS, locals())
    obs_hypparams['sigma_0'] = np.eye(obs_dim)
    obs_hypparams['mu_0'] = np.zeros(obs_dim)
#estimation&result_write#
    print "--------------------------------------estimation process start--------------------------------------"
    result = Result(result_dir, DATA_N)
    loglikelihood = []
    for idx in progprint_xrange(ITER_N, perline=10):
        model.resample_model()
        loglikelihood.append(result.save_loglikelihood(model))
        result.save(model)
    result.write_loglikelihood(loglikelihood)
    print "--------------------------------------estimation process completed!!--------------------------------------"
#--------------------------------------direct execution function--------------------------------------#
if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--result-dir', default=None)
    args = parser.parse_args()
    main(**vars(args))
