import json
import os
from paver.easy import pushd
import numpy as np
import matplotlib
matplotlib.use('Agg') # in the case of perform on server
import matplotlib.pyplot as plt
import pickle
import csv
from sklearn import metrics

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('directory')
    # opts = parser.parse_args()
    summary = Summary()
    figs_dir = 'summary_figs'
    os.path.exists(figs_dir) or os.mkdir(figs_dir)
    with pushd(figs_dir):
        summary.a_rand_index(summary.sample_letters,summary.input_data,'l')
        summary.a_rand_index(summary.sample_states,summary.input_data2,'s')
        with open('word_list.txt',"w") as f:
            for num, key in enumerate(summary.word_list):
                f.write("iter%d:: " % num)
                for num2, key2 in enumerate(key):
                    f.write("%d:" % num2 + str(key2) + " ")
                f.write("\n")
        # plot sample states and letters
        for idx in range(summary.data_size):
            summary.plot_states(idx)
            plt.savefig('sample_states_%d.png' % idx)
            summary.plot_state_boundaries(idx)
            plt.savefig('state_boundary_%d.png' % idx)
            summary.plot_letters(idx)
            plt.savefig('sample_letters_%d.png' % idx)
            plt.clf()

class Summary(object):
    def __init__(self, dirpath = '.'):
        with open('parameter.json') as f:
            params = self.params = json.load(f)
        with open('fig_title.json') as f2:
            fig_title = self.fig_title = json.load(f2)
        with open('sample_word_list.txt') as f3:
            self.word_list = pickle.load(f3)
        self.data_size = params['DATA_N']
        self.input_data=[]
        self.input_data2=[]
        for i in fig_title:
            data_l = np.loadtxt(i + ".txt")
            data_l2 = np.loadtxt(i + ".lab")
            self.input_data.append(data_l[0])
            self.input_data2.append(data_l2)
        self.sample_states = [np.loadtxt('sample_states_%d.txt' % i)for i in range(params['DATA_N'])]
        self.sample_letters = [np.loadtxt('sample_letters_%d.txt' % i)for i in range(params['DATA_N'])]
        self.state_ranges = []
        for i in range(params['DATA_N']):
            with open('state_ranges_%d.txt' % i) as f:
                self.state_ranges.append(pickle.load(f))
        llist = np.loadtxt("loglikelihood.txt").tolist()
        self.maxlikelihood = (max(llist), llist.index(max(llist)))

    def a_rand_index(self,sample_data,true_data,char):
        RIs=[]
        for idx in range(len(sample_data[0])):
            true=[]
            sample=[]
            for key,key2 in zip(sample_data,true_data):
                sample.extend(key[idx])
                true.extend(key2)
            ris=metrics.adjusted_rand_score(true, sample)
            RIs.append(ris)
        np.savetxt("aRIs_"+char+".txt",RIs)
        true=[]
        sample=[]
        for key,key2 in zip(sample_data,true_data):
            sample.extend(key[99])
            true.extend(key2)
        ri=metrics.adjusted_rand_score(true, sample)
        str="max_adjusted_rand_index_"+char+".txt"
        f = open(str,'w')
        writer = csv.writer(f)
        writer.writerow(["adjusted_rand_score",ri])

    def _plot_discreate_sequence(self, true_data, title, sample_data, label = u'', plotopts = {}):
        ax = plt.subplot2grid((10, 1), (1, 0))
        plt.sca(ax)
        ax.matshow([true_data], aspect = 'auto')
        plt.ylabel('Truth Label')

        # label matrix
        ax = plt.subplot2grid((10, 1), (2, 0), rowspan = 8)
        plt.suptitle(title)
        plt.sca(ax)
        ax.matshow(sample_data, aspect = 'auto', **plotopts)

        plt.xlabel('Frame')
        plt.ylabel('Iteration')
        plt.xticks(())

    def _plot_label_boundary(self, true_data, title, sample_data, label = u''):
        boundaries = [[stop for state, (start, stop) in r] for r in sample_data]
        size = boundaries[0][-1]
        data = np.zeros((len(sample_data), size))
        for i, b in enumerate(boundaries):
            for x in b[:-1]:
                data[i, x] = 1.0

        self._plot_discreate_sequence(true_data, title, data, label, plotopts = {'cmap': 'Greys'})

    def plot_letters(self, idx):
        self._plot_discreate_sequence(
            self.input_data[idx],
            self.fig_title[idx],
            self.sample_letters[idx],
            label=self.sample_letters[idx]
        )

    def plot_states(self, idx):
        self._plot_discreate_sequence(
            self.input_data2[idx],
            self.fig_title[idx],
            self.sample_states[idx],
            label=self.sample_states[idx]
        )

    def plot_state_boundaries(self, idx):
        self._plot_label_boundary(
            self.input_data2[idx],
            self.fig_title[idx],
            self.state_ranges[idx],
            label=self.sample_states[idx]
        )

if __name__ == '__main__':
    main()
