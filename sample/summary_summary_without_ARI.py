#%%
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

#%%
label = sys.argv[1]
dirs = sys.argv[2:]
dirs.sort()

#%%
if not os.path.exists("figures"):
    os.mkdir("figures")

if not os.path.exists("summary_files"):
    os.mkdir("summary_files")

#%%
print("Initialize variables....")
N = len(dirs)
tmp = np.loadtxt(label + "/" +  dirs[0] + "/summary_files/resample_times.txt")
T = tmp.shape[0]

resample_times  = np.empty((N, T))
log_likelihoods = np.empty((N, T+1))
print("Done!")

#%%
print("Loading results....")
for i, path in enumerate(dirs):
    resample_times[i] = np.loadtxt(label + "/" + path + "/summary_files/resample_times.txt")
    log_likelihoods[i] = np.loadtxt(label + "/" + path + "/summary_files/log_likelihood.txt")
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
