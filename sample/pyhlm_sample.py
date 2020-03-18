import numpy as np
from pyhlm.model import WeakLimitHDPHLM, WeakLimitHDPHLMPython
from pyhlm.internals.hlm_states import WeakLimitHDPHLMStates
from pyhlm.word_model import LetterHSMM, LetterHSMMPython
import pyhsmm
from tqdm import trange
import warnings
warnings.filterwarnings('ignore')
import time
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from util.config_parser import ConfigParser_with_eval
from pathlib import Path

#%% parse arguments
hypparams_model = "hypparams/model.config"
hypparams_letter_duration = "hypparams/letter_duration.config"
hypparams_letter_hsmm = "hypparams/letter_hsmm.config"
hypparams_letter_observation = "hypparams/letter_observation.config"
hypparams_pyhlm = "hypparams/pyhlm.config"
hypparams_word_length = "hypparams/word_length.config"
hypparams_superstate = "hypparams/superstate.config"

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

parser.add_argument("--model", default=hypparams_model, help="hyper parameters of model")
parser.add_argument("--letter_duration", default=hypparams_letter_duration, help="hyper parameters of letter duration")
parser.add_argument("--letter_hsmm", default=hypparams_letter_hsmm, help="hyper parameters of letter HSMM")
parser.add_argument("--letter_observation", default=hypparams_letter_observation, help="hyper parameters of letter observation")
parser.add_argument("--pyhlm", default=hypparams_pyhlm, help="hyper parameters of pyhlm")
parser.add_argument("--word_length", default=hypparams_word_length, help="hyper parameters of word length")
parser.add_argument("--superstate", default=hypparams_superstate, help="hyper parameters of superstate")

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
    d = np.cumsum(dur)
    unpacked[d-1] = 1.0
    return unpacked

def save_stateseq(model):
    # Save sampled states sequences.
    names = np.loadtxt("files.txt", dtype=str)
    for i, s in enumerate(model.states_list):
        with open("results/" + names[i] + "_s.txt", "a") as f:
            np.savetxt(f, s.stateseq, fmt="%d")
        with open("results/" + names[i] + "_l.txt", "a") as f:
            np.savetxt(f, s.letter_stateseq, fmt="%d")
        with open("results/" + names[i] + "_d.txt", "a") as f:
            np.savetxt(f, unpack_durations(s.durations_censored), fmt="%d")

def save_params_as_text(itr_idx, model):
    with open("parameters/ITR_{0:04d}.txt".format(itr_idx), "w") as f:
        f.write(str(model.params))

def save_params_as_file(iter_idx, model):
    params = model.params
    root_dir = Path("parameters/ITR_{0:04d}".format(iter_idx))
    root_dir.mkdir(exist_ok=True)
    save_json(root_dir, params)

def save_json(root_dir, json_obj):
    for keyname, subjson in json_obj.items():
        type_of_subjson = type(subjson)
        if type_of_subjson == dict:
            dir = root_dir / keyname
            dir.mkdir(exist_ok=True)
            save_json(dir, json_obj[keyname])
        else:
            savefile = root_dir / f"{keyname}.txt"
            if type_of_subjson == np.ndarray:
                if subjson.dtype in [np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64]:
                    np.savetxt(savefile, subjson, fmt="%d")
                else:
                    np.savetxt(savefile, subjson)
            else:
                savefile.write_text(str(subjson))

def save_params_as_npz(iter_idx, model):
    params = model.params
    flatten_params = flatten_json(params)
    # flatten_params = copy_flatten_json(flatten_params)
    np.savez(f"parameters/ITR_{iter_idx:04d}.npz", **flatten_params)

def flatten_json(json_obj, keyname_prefix=None, dict_obj=None):
    if dict_obj is None:
        dict_obj = {}
    if keyname_prefix is None:
        keyname_prefix = ""
    for keyname, subjson in json_obj.items():
        if type(subjson) == dict:
            prefix = f"{keyname_prefix}{keyname}/"
            flatten_json(subjson, keyname_prefix=prefix, dict_obj=dict_obj)
        else:
            dict_obj[f"{keyname_prefix}{keyname}"] = subjson
    return dict_obj

def unflatten_json(flatten_json_obj):
    dict_obj = {}
    for keyname, value in flatten_json_obj.items():
        current_dict = dict_obj
        splitted_keyname = keyname.split("/")
        for key in splitted_keyname[:-1]:
            if key not in current_dict:
                current_dict[key] = {}
            current_dict = current_dict[key]
        current_dict[splitted_keyname[-1]] = value
    return dict_obj

def copy_flatten_json(json_obj):
    new_json = {}
    for keyname, subjson in json_obj.items():
        type_of_subjson = type(subjson)
        if type_of_subjson in [int, float, complex, bool]:
            new_json[keyname] = subjson
        elif type_of_subjson in [list, tuple]:
            new_json[keyname] = subjson[:]
        elif type_of_subjson == np.ndarray:
            new_json[keyname] = subjson.copy()
        else:
            raise NotImplementedError(f"type :{type_of_subjson} can not copy. Plz implement here!")
    return new_json

def save_loglikelihood(model):
    with open("summary_files/log_likelihood.txt", "a") as f:
        f.write(str(model.log_likelihood()) + "\n")

def save_resample_times(resample_time):
    with open("summary_files/resample_times.txt", "a") as f:
        f.write(str(resample_time) + "\n")

#%%
Path("results").mkdir(exist_ok=True)
Path("parameters").mkdir(exist_ok=True)
Path("summary_files").mkdir(exist_ok=True)

#%% config parse
config_parser = load_config(hypparams_model)
section         = config_parser["model"]
thread_num      = section["thread_num"]
pretrain_iter   = section["pretrain_iter"]
train_iter      = section["train_iter"]
word_num        = section["word_num"]
letter_num      = section["letter_num"]
observation_dim = section["observation_dim"]

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
# save_params_as_text(0, model)
# save_params_as_file(0, model)
save_params_as_npz(0, model)
save_loglikelihood(model)

#%%
for t in trange(train_iter):
    st = time.time()
    model.resample_model(num_procs=thread_num)
    resample_model_time = time.time() - st
    save_stateseq(model)
    save_loglikelihood(model)
    # save_params_as_text(t+1, model)
    # save_params_as_file(t+1, model)
    save_params_as_npz(t+1, model)
    save_resample_times(resample_model_time)
    print(model.word_list)
    print(model.word_counts())
    print(f"log_likelihood:{model.log_likelihood()}")
    print(f"resample_model:{resample_model_time}")
