#%%
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.cm as cm
from tqdm import trange, tqdm

from sklearn.metrics import adjusted_rand_score
from argparse import ArgumentParser
from util.config_parser import ConfigParser_with_eval

#%% parse arguments
def arg_check(value, default):
    return value if value else default

default_hypparams_model = "hypparams/model.config"

parser = ArgumentParser()
parser.add_argument("--model", help=f"hyper parameters of model, default is [{default_hypparams_model}]")
args = parser.parse_args()

hypparams_model = arg_check(args.model, default_hypparams_model)

#%%
def load_config(filename):
    cp = ConfigParser_with_eval()
    cp.read(filename)
    return cp

#%%
def get_names():
    return np.loadtxt("files.txt", dtype=str)

def get_datas_and_length(names):
    datas = [np.loadtxt("DATA/" + name + ".txt") for name in names]
    length = [len(d) for d in datas]
    return datas, length

def get_results_of_word(names, length):
    return _joblib_get_results(names, length, "s")

def get_results_of_letter(names, length):
    return _joblib_get_results(names, length, "l")

def get_results_of_duration(names, length):
    return _joblib_get_results(names, length, "d")

def _get_results(names, lengths, c):
    return [np.loadtxt("results/" + name + "_" + c + ".txt").reshape((-1, l)) for name, l in zip(names, lengths)]

def _joblib_get_results(names, lengths, c):
    from joblib import Parallel, delayed
    def _component(name, length, c):
        return np.loadtxt("results/" + name + "_" + c + ".txt").reshape((-1, length))
    return Parallel(n_jobs=-1)([delayed(_component)(n, l, c) for n, l in zip(names, lengths)])

def _plot_discreate_sequence(feature, title, sample_data, cmap=None):
    ax = plt.subplot2grid((2, 1), (0, 0))
    plt.sca(ax)
    ax.plot(feature)
    ax.set_xlim((0, feature.shape[0]-1))
    plt.ylabel('Feature')
    #label matrix
    ax = plt.subplot2grid((2, 1), (1, 0))
    plt.suptitle(title)
    plt.sca(ax)
    ax.matshow(sample_data, aspect = 'auto', cmap=cmap)
    #write x&y label
    plt.xlabel('Frame')
    plt.ylabel('Iteration')
    plt.xticks(())

#%%
if not os.path.exists("figures"):
    os.mkdir("figures")

if not os.path.exists("summary_files"):
    os.mkdir("summary_files")

#%% config parse
print("Loading model config...")
config_parser = load_config(hypparams_model)
section = config_parser["model"]
word_num = section["word_num"]
letter_num = section["letter_num"]
print("Done!")

#%%
print("Loading results....")
names = get_names()
datas, length = get_datas_and_length(names)

l_results = get_results_of_letter(names, length)
w_results = get_results_of_word(names, length)
d_results = get_results_of_duration(names, length)

log_likelihood = np.loadtxt("summary_files/log_likelihood.txt")
resample_times = np.loadtxt("summary_files/resample_times.txt")
print("Done!")

train_iter = l_results[0].shape[0]

#%%
lcolors = ListedColormap([cm.tab20(float(i)/letter_num) for i in range(letter_num)])
wcolors = ListedColormap([cm.tab20(float(i)/word_num) for i in range(word_num)])

#%%
print("Plot results...")
for i, name in enumerate(tqdm(names)):
    plt.clf()
    _plot_discreate_sequence(datas[i], name + "_l", l_results[i], cmap=lcolors)
    plt.savefig("figures/" + name + "_l.png")
    plt.clf()
    _plot_discreate_sequence(datas[i], name + "_s", w_results[i], cmap=wcolors)
    plt.savefig("figures/" + name + "_s.png")
    plt.clf()
    _plot_discreate_sequence(datas[i], name + "_d", d_results[i], cmap=cm.binary)
    plt.savefig("figures/" + name + "_d.png")
print("Done!")

#%%
plt.clf()
plt.title("Log likelihood")
plt.plot(range(train_iter+1), log_likelihood, ".-")
plt.savefig("figures/Log_likelihood.png")

#%%
plt.clf()
plt.title("Resample times")
plt.plot(range(train_iter), resample_times, ".-")
plt.savefig("figures/Resample_times.png")

#%%
with open("summary_files/Sum_of_resample_times.txt", "w") as f:
    f.write(str(np.sum(resample_times)))
