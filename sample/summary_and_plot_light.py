#%%
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.cm as cm
from tqdm import trange, tqdm
from sklearn.metrics import adjusted_rand_score, f1_score
from argparse import ArgumentParser
from util.config_parser import ConfigParser_with_eval
import warnings
warnings.filterwarnings('ignore')

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

def get_letter_labels(names):
    return _get_labels(names, "lab")

def get_word_labels(names):
    return _get_labels(names, "lab2")

def _get_labels(names, ext):
    return [np.loadtxt("LABEL/" + name + "." + ext) for name in names]


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

def _convert_label(truth, predict, N):
    converted_label = np.full_like(truth, N)
    for true_lab in range(N):
        counted = [np.sum(predict[truth == true_lab] == pred) for pred in range(N)]
        pred_lab = np.argmax(counted)
        converted_label[predict == pred_lab] = true_lab
    return converted_label

def calc_f1_score(truth, predict, N, **kwargs):
    converted_predict = _convert_label(truth, predict, N)
    return f1_score(truth, converted_predict, labels=np.unique(converted_predict), **kwargs )

#%%
if not os.path.exists("figures"):
    os.mkdir("figures")

if not os.path.exists("summary_files"):
    os.mkdir("summary_files")

#%% config parse
print("Loading model config...")
config_parser = load_config(hypparams_model)
section = config_parser["model"]
train_iter = section["train_iter"]
word_num = section["word_num"]
letter_num = section["letter_num"]
print("Done!")

#%%
print("Loading results....")
names = get_names()
datas, length = get_datas_and_length(names)
l_labels = get_letter_labels(names)
w_labels = get_word_labels(names)
concat_l_l = np.concatenate(l_labels, axis=0)
concat_w_l = np.concatenate(w_labels, axis=0)

l_results = get_results_of_letter(names, length)
w_results = get_results_of_word(names, length)
d_results = get_results_of_duration(names, length)

concat_l_r = np.concatenate(l_results, axis=1)
concat_w_r = np.concatenate(w_results, axis=1)

log_likelihood = np.loadtxt("summary_files/log_likelihood.txt")
resample_times = np.loadtxt("summary_files/resample_times.txt")
print("Done!")

#%%
letter_ARI = np.zeros(train_iter)
letter_macro_f1_score = np.zeros(train_iter)
letter_micro_f1_score = np.zeros(train_iter)
word_ARI = np.zeros(train_iter)
word_macro_f1_score = np.zeros(train_iter)
word_micro_f1_score = np.zeros(train_iter)

#%% calculate ARI
print("Calculating ARI...")
for t in trange(train_iter):
    letter_ARI[t] = adjusted_rand_score(concat_l_l, concat_l_r[t])
    letter_macro_f1_score[t] = calc_f1_score(concat_l_l, concat_l_r[t], letter_num, average="macro")
    letter_micro_f1_score[t] = calc_f1_score(concat_l_l, concat_l_r[t], letter_num, average="micro")
    word_ARI[t] = adjusted_rand_score(concat_w_l, concat_w_r[t])
    word_macro_f1_score[t] = calc_f1_score(concat_w_l, concat_w_r[t], word_num, average="macro")
    word_micro_f1_score[t] = calc_f1_score(concat_w_l, concat_w_r[t], word_num, average="micro")
print("Done!")

#%% plot ARIs.
plt.clf()
plt.title("Letter ARI")
plt.plot(range(train_iter), letter_ARI, ".-")
plt.savefig("figures/Letter_ARI.png")

#%%
plt.clf()
plt.title("Word ARI")
plt.plot(range(train_iter), word_ARI, ".-")
plt.savefig("figures/Word_ARI.png")

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
np.savetxt("summary_files/Letter_ARI.txt", letter_ARI)
np.savetxt("summary_files/Letter_macro_F1_score.txt", letter_macro_f1_score)
np.savetxt("summary_files/Letter_micro_F1_score.txt", letter_micro_f1_score)
np.savetxt("summary_files/Word_ARI.txt", word_ARI)
np.savetxt("summary_files/Word_macro_F1_score.txt", word_macro_f1_score)
np.savetxt("summary_files/Word_micro_F1_score.txt", word_micro_f1_score)
with open("summary_files/Sum_of_resample_times.txt", "w") as f:
    f.write(str(np.sum(resample_times)))
