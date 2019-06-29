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

def get_labels(names):
    letter_labels = [np.loadtxt("LABEL/" + name + ".lab") for name in names]
    word_labels = [np.loadtxt("LABEL/" + name + ".lab2") for name in names]
    return letter_labels, word_labels

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

def _boundary(label):
    diff = np.diff(label)
    diff[diff!=0] = 1
    return np.concatenate((diff, [0]))

def _plot_discreate_sequence(true_data, title, sample_data, cmap=None, cmap2=None, label_cmap=None):
    ax = plt.subplot2grid((10, 1), (1, 0))
    plt.sca(ax)
    if label_cmap is None:
        label_cmap = cmap
    ax.matshow([true_data], aspect='auto', cmap=label_cmap)
    plt.ylabel('Truth Label')
    #label matrix
    ax = plt.subplot2grid((10, 1), (2, 0), rowspan = 8)
    plt.suptitle(title)
    plt.sca(ax)
    if cmap2 is not None:
        cmap = cmap2
    ax.matshow(sample_data, aspect='auto', cmap=cmap)
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
l_labels, w_labels = get_labels(names)
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

train_iter = l_results[0].shape[0]

#%%
letter_ARI = np.zeros(train_iter)
letter_macro_f1_score = np.zeros(train_iter)
letter_micro_f1_score = np.zeros(train_iter)
word_ARI = np.zeros(train_iter)
word_macro_f1_score = np.zeros(train_iter)
word_micro_f1_score = np.zeros(train_iter)

#%%
lcolors = ListedColormap([cm.tab20(float(i)/letter_num) for i in range(letter_num)])
wcolors = ListedColormap([cm.tab20(float(i)/word_num) for i in range(word_num)])

#%%
print("Plot results...")
for i, name in enumerate(tqdm(names)):
    plt.clf()
    # _plot_discreate_sequence(l_labels[i], name + "_l", l_results[i], cmap=lcolors)
    _plot_discreate_sequence(_boundary(l_labels[i]), name + "_l", l_results[i], cmap=lcolors, label_cmap=cm.binary)
    plt.savefig("figures/" + name + "_l.png")
    plt.clf()
    # _plot_discreate_sequence(w_labels[i], name + "_s", w_results[i], cmap=wcolors)
    _plot_discreate_sequence(_boundary(w_labels[i]), name + "_s", w_results[i], cmap=wcolors, label_cmap=cm.binary)
    plt.savefig("figures/" + name + "_s.png")
    plt.clf()
    # _plot_discreate_sequence(w_labels[i], name + "_d", d_results[i], cmap=wcolors, cmap2=cm.binary)
    _plot_discreate_sequence(_boundary(w_labels[i]), name + "_d", d_results[i], cmap2=cm.binary, label_cmap=cm.binary)
    plt.savefig("figures/" + name + "_d.png")
print("Done!")

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

#%%
plt.title("Word ARI")
plt.clf()
plt.plot(range(train_iter), word_ARI, ".-")

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
