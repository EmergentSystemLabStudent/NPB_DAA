# -*- coding: utf-8 -*-
import numpy as np
from sklearn import metrics
import matplotlib
matplotlib.use('Agg') #in the case of perform on server
import pickle
import json
#-------------------------------

class Summary(object):
    def __init__(self, dirpath = '.'):
        print(dirpath)

        with open(dirpath+'/'+'parameter.json') as f:
            self.params = json.load(f)
        with open(dirpath+'/'+'fig_title.json') as f2:
            self.fig_title = json.load(f2)
        with open(dirpath+'/'+'sample_word_list.txt') as f3:
            self.word_list = pickle.load(f3)
        self.data_size = self.params['DATA_N']
        print (self.data_size)
        self.input_data = [np.loadtxt("./LABEL/" + i + ".lab") for i in self.fig_title]#label change here
        self.input_data2 = [np.loadtxt("./LABEL/" + i + ".lab2") for i in self.fig_title]#label change here
        self.sample_states = [np.loadtxt(dirpath+'/'+'sample_states_%d.txt' % i) for i in range(self.params['DATA_N'])]
        self.sample_letters = [np.loadtxt(dirpath+'/'+'sample_letters_%d.txt' % i) for i in range(self.params['DATA_N'])] #%iをループで記述している．
        self.state_ranges = []

        for i in range(self.params['DATA_N']):#ファイル読み込み
            with open('state_ranges_%d.txt' % i) as f:
                self.state_ranges.append(pickle.load(f))
        llist = np.loadtxt("loglikelihood.txt").tolist()
        print llist
        self.maxlikelihood = (max(llist), llist.index(max(llist)))
        # manipulation part ラベル保管用
        self.l_label_dic = {}
        self.s_label_dic = {}
        # manipulation part end
        # --------------------------------------letter&state confused matrix function--------------------------------------#
        def letter_confused_matrix(self):
            #change here
            a = []
            i = []
            for key, key2 in zip(self.sample_letters, self.input_data):

        # --------------------------------------write result_graph--------------------------------------#