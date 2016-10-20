# -*- coding: utf-8 -*-
import numpy as np
from sklearn import metrics
import matplotlib
matplotlib.use('Agg') #in the case of perform on server
import pickle
import json
import csv

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
        self.input_data = [np.loadtxt("./LABEL/" + i + ".lab") for i in self.fig_title]#label change here　ファイルのテキスト
        self.input_data2 = [np.loadtxt("./LABEL/" + i + ".lab2") for i in self.fig_title]#label change here
        self.sample_states = [np.loadtxt(dirpath+'/'+'sample_states_%d.txt' % i) for i in range(self.params['DATA_N'])]
        self.sample_letters = [np.loadtxt(dirpath+'/'+'sample_letters_%d.txt' % i) for i in range(self.params['DATA_N'])] #%iをループで記述している．
        self.state_ranges = []

        for i in range(self.params['DATA_N']):#ファイル読み込み
            with open(dirpath+'/'+'state_ranges_%d.txt' % i) as f:
                self.state_ranges.append(pickle.load(f))
        llist = np.loadtxt(dirpath+'/'+"loglikelihood.txt").tolist()
        self.maxlikelihood = (max(llist), llist.index(max(llist)))#maxが入っている配列取得

        # manipulation part ラベル保管用
        self.l_label_dic = {}
        self.s_label_dic = {}
        # manipulation part end

        # --------------------------------------letter&state confused matrix function--------------------------------------#
    def letter_confused_matrix(self):
        #change here
        a = []
        i = []
        u=[]
        e=[]
        o=[]

        #ラベル付けしたときの番号を振る。inudataはラベル　
        for key, key2 in zip(self.sample_letters, self.input_data):
           #print key
           #print key[self.maxlikelihood[1]]#最大尤度のself.sample_lettersを抜き取っている
           for key3, key4 in zip(key2,key[self.maxlikelihood[1]]):#ここのkey2でリストを分解するのでラベル中の値が入る
                if key3 == 0:#ラベルづけで設定した値
                    a.append(key4)#sample_lettesの最大尤度のインデックスの値のラベル番号　推定値
                    #print a
                elif key3 == 1:
                    i.append(key4)
                elif key3 == 2:
                    u.append(key4)
                elif key3 == 3:
                    e.append(key4)
                elif key3 == 4:
                    o.append(key4)

        l_max = max(a + i + u + e + o)# 全リストの中から最大値つまり、ラベルの最後の番号
        a_count = []
        i_count = []
        u_count = []
        e_count = []
        o_count = []

        for num in range(int(l_max)+1):
            a_count.append(a.count(num))#引数で指定したオブジェクトが持つ値がいくつの要素に含まれているのかを返します。
            print a_count
            i_count.append(i.count(num))
            u_count.append(u.count(num))
            e_count.append(e.count(num))
            o_count.append(o.count(num))

        #confused_matrix_l.csvにはどれだけ間違ったが記述されていることになる
        f = open('confused_matrix_l.csv','w')
        writer = csv.writer(f)
        writer.writerow(["phone|letter_label"]+range(int(l_max)+1))#音素列の範囲
        writer.writerow(["a"]+a_count)
        writer.writerow(["i"]+i_count)
        writer.writerow(["u"]+u_count)
        writer.writerow(["e"]+e_count)
        writer.writerow(["o"]+o_count)
        writer.writerow([])
        writer.writerow(["a_label:"+str(a_count.index(max(a_count))),"i_label:"+str(i_count.index(max(i_count))),"u_label:"+str(u_count.index(max(u_count))),"e_label:"+str(e_count.index(max(e_count))),"o_label:"+str(o_count.index(max(o_count)))])
        self.l_label_dic[a_count.index(max(a_count))]="a"
        self.l_label_dic[i_count.index(max(i_count))]="i"
        self.l_label_dic[u_count.index(max(u_count))]="u"
        self.l_label_dic[e_count.index(max(e_count))]="e"
        self.l_label_dic[o_count.index(max(o_count))]="o"
        #confused_matrix_l.csvで認識がちゃんと行われているかがわかる

        # --------------------------------------write result_graph--------------------------------------#