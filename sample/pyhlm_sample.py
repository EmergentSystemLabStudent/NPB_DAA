import os
import numpy as np
from pyhlm.model import WeakLimitHDPHLM, WeakLimitHDPHLMPython
from pyhlm.internals.hlm_states import WeakLimitHDPHLMStates
from pyhlm.word_model import LetterHSMM, LetterHSMMPython
import pyhsmm
import warnings
from tqdm import trange
warnings.filterwarnings('ignore')
import time
from argparse import ArgumentParser
from util.config_parser import ConfigParser_with_eval

#%% parse arguments
default_hypparams_model = "hypparams/model.config"
default_hypparams_letter_duration = "hypparams/letter_duration.config"
default_hypparams_letter_hsmm = "hypparams/letter_hsmm.config"
default_hypparams_letter_observation = "hypparams/letter_observation.config"
default_hypparams_pyhlm = "hypparams/pyhlm.config"
default_hypparams_word_length = "hypparams/word_length.config"
default_hypparams_superstate = "hypparams/superstate.config"

parser = ArgumentParser()
parser.add_argument("--model", default=default_hypparams_model, help=f"hyper parameters of model, default is [{default_hypparams_model}]")
parser.add_argument("--letter_duration", default=default_hypparams_letter_duration, help=f"hyper parameters of letter duration, default is [{default_hypparams_letter_duration}]")
parser.add_argument("--letter_hsmm", default=default_hypparams_letter_hsmm, help=f"hyper parameters of letter HSMM, default is [{default_hypparams_letter_hsmm}]")
parser.add_argument("--letter_observation", default=default_hypparams_letter_observation, help=f"hyper parameters of letter observation, default is [{default_hypparams_letter_observation}]")
parser.add_argument("--pyhlm", default=default_hypparams_pyhlm, help=f"hyper parameters of pyhlm, default is [{default_hypparams_pyhlm}]")
parser.add_argument("--word_length", default=default_hypparams_word_length, help=f"hyper parameters of word length, default is [{default_hypparams_word_length}]")
parser.add_argument("--superstate", default=default_hypparams_superstate, help=f"hyper parameters of superstate, default is [{default_hypparams_superstate}]")
args = parser.parse_args()

hypparams_model = args.model
hypparams_letter_duration = args.letter_duration
hypparams_letter_hsmm = args.letter_hsmm
hypparams_letter_observation = args.letter_observation
hypparams_pyhlm = args.pyhlm
hypparams_word_length = args.word_length
hypparams_superstate = args.superstate

#%%
def load_config(filename):
    cp = ConfigParser_with_eval()
    cp.read(filename)
    return cp

#%%
def load_datas():
    data = []
    names = np.loadtxt("files.txt", dtype=str)
    files = names
    for name in names:
        data.append(np.loadtxt("DATA/" + name + ".txt"))
    return data

def unpack_durations(dur):
    unpacked = np.zeros(dur.sum())
    d = np.cumsum(dur[:-1])
    unpacked[d-1] = 1.0
    return unpacked

def save_stateseq(model):
    # Save sampled states sequences.
    names = np.loadtxt("files.txt", dtype=str)
    for i, s in enumerate(model.states_list):
        with open("results/" + names[i] + "_s.txt", "a") as f:
            np.savetxt(f, s.stateseq)
        with open("results/" + names[i] + "_l.txt", "a") as f:
            np.savetxt(f, s.letter_stateseq)
        with open("results/" + names[i] + "_d.txt", "a") as f:
            np.savetxt(f, unpack_durations(s.durations_censored))

def save_params(itr_idx, model):
    with open("parameters/ITR_{0:04d}.txt".format(itr_idx), "w") as f:
        f.write(str(model.params))

def save_loglikelihood(model):
    with open("summary_files/log_likelihood.txt", "a") as f:
        f.write(str(model.log_likelihood()) + "\n")

def save_resample_times(resample_time):
    with open("summary_files/resample_times.txt", "a") as f:
        f.write(str(resample_time) + "\n")

#%%
if not os.path.exists('results'):
    os.mkdir('results')

if not os.path.exists('parameters'):
    os.mkdir('parameters')

if not os.path.exists('summary_files'):
    os.mkdir('summary_files')

#%% config parse
config_parser = load_config(hypparams_model)
section         = config_parser["model"]
thread_num      = section["thread_num"]
pretrain_iter   = section["pretrain_iter"]
train_iter      = section["train_iter"]
word_num        = section["word_num"]
letter_num      = section["letter_num"]
observation_dim = ["observation_dim"]

hlm_hypparams = load_config(hypparams_pyhlm)["pyhlm"]

config_parser = load_config(hypparams_letter_observation)
obs_hypparams = [config_parser[f"{i+1}_th"] for i in range(letter_num)]

config_parser = load_config(hypparams_letter_duration)
dur_hypparams = [config_parser[f"{i+1}_th"] for i in range(letter_num)]

len_hypparams = load_config(hypparams_word_length)["word_length"]

letter_hsmm_hypparams = load_config(hypparams_letter_hsmm)["letter_hsmm"]

superstate_config = load_config(hypparams_superstate)

#%% make instance of distributions and model
letter_obs_distns = [pyhsmm.distributions.Gaussian(**hypparam) for hypparam in obs_hypparams]
letter_dur_distns = [pyhsmm.distributions.PoissonDuration(**hypparam) for hypparam in dur_hypparams]
dur_distns = [pyhsmm.distributions.PoissonDuration(lmbda=20) for _ in range(word_num)]
length_distn = pyhsmm.distributions.PoissonDuration(**len_hypparams)

letter_hsmm = LetterHSMM(**letter_hsmm_hypparams, obs_distns=letter_obs_distns, dur_distns=letter_dur_distns)
model = WeakLimitHDPHLM(**hlm_hypparams, letter_hsmm=letter_hsmm, dur_distns=dur_distns, length_distn=length_distn)

#%%
files = np.loadtxt("files.txt", dtype=str)
datas = load_datas()

#%% Pre training.
for data in datas:
    letter_hsmm.add_data(data, **superstate_config["DEFAULT"])
for t in trange(pretrain_iter):
    letter_hsmm.resample_model(num_procs=thread_num)
letter_hsmm.states_list = []

#%%
print("Add datas...")
for name, data in zip(files, datas):
    model.add_data(data, **superstate_config[name], generate=False)
model.resample_states(num_procs=thread_num)
# # or
# for name, data in zip(files, datas):
#     model.add_data(data, **superstate_config[name], initialize_from_prior=False)
print("Done!")

#%% Save init params
save_params(0, model)
save_loglikelihood(model)

#%%
for t in trange(train_iter):
    st = time.time()
    model.resample_model(num_procs=thread_num)
    resample_model_time = time.time() - st
    save_stateseq(model)
    save_loglikelihood(model)
    save_params(t+1, model)
    save_resample_times(resample_model_time)
    print(model.word_list)
    print(model.word_counts())
    print(f"log_likelihood:{model.log_likelihood()}")
    print(f"resample_model:{resample_model_time}")
