#%%
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pathlib import Path
import re

#%%
parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("--result_dir", type=Path, required=True)
args = parser.parse_args()

#%%
dirs = [dir for dir in args.result_dir.iterdir() if dir.is_dir() and re.match(r"^[0-9]+$", dir.stem)]
dirs.sort(key=lambda dir: dir.stem)

#%%
Path("figures").mkdir(exist_ok=True)
Path("summary_files").mkdir(exist_ok=True)

#%%
print("Initialize variables....")
N = len(dirs)
tmp = np.loadtxt(dirs[0] / "summary_files/resample_times.txt")
T = tmp.shape[0]

resample_times  = np.empty((N, T))
log_likelihoods = np.empty((N, T+1))
print("Done!")

#%%
print("Loading results....")
for i, dir in enumerate(dirs):
    resample_times[i] = np.loadtxt(dir / "summary_files/resample_times.txt")
    log_likelihoods[i] = np.loadtxt(dir / "summary_files/log_likelihood.txt")
print("Done!")

#%%
print("Ploting...")
plt.clf()
plt.errorbar(range(T), resample_times.mean(axis=0), yerr=resample_times.std(axis=0))
plt.xlabel("Iteration")
plt.ylabel("Execution time [sec]")
plt.title("Transitions of the execution time")
plt.savefig("figures/summary_of_execution_time.png")

plt.clf()
plt.errorbar(range(T+1), log_likelihoods.mean(axis=0), yerr=log_likelihoods.std(axis=0))
plt.xlabel("Iteration")
plt.ylabel("Log likelihood")
plt.title("Transitions of the log likelihood")
plt.savefig("figures/summary_of_log_likelihood.png")
print("Done!")

#%%
print("Save npy files...")
np.save("summary_files/resample_times.npy", resample_times)
np.save("summary_files/log_likelihoods.npy", log_likelihoods)
print("Done!")
