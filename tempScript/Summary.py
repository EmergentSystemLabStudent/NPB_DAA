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
        #また，同じラベルだと判定した場合，最後のラベルとしてにんしきする．

    def state_confused_matrix(self):
        #大体同じである．
        aioi = []
        aue = []
        ao = []
        ie = []
        uo = []

        for key, key2 in zip(self.sample_states, self.input_data2):
            for key3, key4 in zip(key[self.maxlikelihood[1]], key2):
                if key4 == 0:
                    aioi.append(key3)
                elif key4 == 1:
                    aue.append(key3)
                elif key4 == 2:
                    ao.append(key3)
                elif key4 == 3:
                    ie.append(key3)
                elif key4 == 4:
                    uo.append(key3)
                    l_max = max(aioi + aue + ao + ie + uo)
                    aioi_count = []
                    aue_count = []
                    ao_count = []
                    ie_count = []
                    uo_count = []

                    for num in range(int(l_max) + 1):
                        aioi_count.append(aioi.count(num))
                        aue_count.append(aue.count(num))
                        ao_count.append(ao.count(num))
                        ie_count.append(ie.count(num))
                        uo_count.append(uo.count(num))

                    f = open('confused_matrix_s.csv', 'w')
                    writer = csv.writer(f)
                    writer.writerow(["word|state_label"] + range(int(l_max) + 1))
                    writer.writerow(["aioi"] + aioi_count)
                    writer.writerow(["aue"] + aue_count)
                    writer.writerow(["ao"] + ao_count)
                    writer.writerow(["ie"] + ie_count)
                    writer.writerow(["uo"] + uo_count)
                    writer.writerow([])
                    writer.writerow(["aioi_label:" + str(aioi_count.index(max(aioi_count))),
                                     "aue_label:" + str(aue_count.index(max(aue_count))),
                                     "ao_label:" + str(ao_count.index(max(ao_count))),
                                     "ie_label:" + str(ie_count.index(max(ie_count))),
                                     "uo_label:" + str(uo_count.index(max(uo_count)))])
                    self.s_label_dic["aioi"] = aioi_count.index(max(aioi_count))
                    self.s_label_dic["aue"] = aue_count.index(max(aue_count))
                    self.s_label_dic["ao"] = ao_count.index(max(ao_count))
                    self.s_label_dic["ie"] = ie_count.index(max(ie_count))
                    self.s_label_dic["uo"] = uo_count.index(max(uo_count))

    # --------------------------------------culculate PER and WER function--------------------------------------#
    def _levenshtein_distance(self, a, b):
        m = [[0] * (len(b) + 1) for i in range(len(a) + 1)]
        for i in xrange(len(a) + 1):
            m[i][0] = i
        for j in xrange(len(b) + 1):
            m[0][j] = j
        for i in xrange(1, len(a) + 1):
            for j in xrange(1, len(b) + 1):
                if a[i - 1] == b[j - 1]:
                    x = 0
                else:
                    x = 1
                m[i][j] = min(m[i - 1][j] + 1, m[i][j - 1] + 1, m[i - 1][j - 1] + x)
        return m[-1][-1]

    def culPER(self):
        #文字列での評価している所
        str_letter = []
        print "--------------------------------------culPER function--------------------------------------"
        print "P_DIC: ", self.l_label_dic#音素ラベル

        #このループは音素列を最大尤度での配列を一つづ見ていく．隣同士で違うおんそならmojiの配列に代入

        for key in self.sample_letters:
            moji = []
            for count, key2 in enumerate(key[self.maxlikelihood[1]]):#最尤度のインデックスをしようして
                try:
                    if key2 != key[self.maxlikelihood[1]][count + 1]:#keymaxlikehoodの音素と次のkeymaxlikehoodの音素との比較
                        moji.append(self.l_label_dic[key2])#もし次のラベルが違うならば，それまでのラベルを追加
                except IndexError:
                    try:
                        moji.append(self.l_label_dic[key2])
                    except KeyError:
                        moji.append("*")
                except KeyError:#わからんかったら*いれとけ
                    moji.append("*")

            str_letter.append("".join(map(str, moji)))
        str_true = []
        print(self.fig_title)

        for key in self.fig_title:
            key = key.replace("2", "")#labe2の2を消す
            key = key.replace("_", "")#_を消す
            str_true.append(key)#同じやつが2つでてくる

        """
        # aioi_ie notokidake #iが二回続いているからから変更している.
        where = np.where(np.array(str_true) == "aioiie")#aioieのインデックスならそのインデックスを取得 二つのインデックス番号がある

        for key in where[0].tolist():#[0]に入っているリストを保管 tolistを配列に変える
            print(str_letter[key])
            str_letter[key] = str_letter[key][:-1] + "ie"
            print(str_letter[key])
        # aioi_ie notokidake end
        """
        #正解ラベルの表示
        print "TRUE: ",str_true
        #今回のletterの推定値
        print "SAMP: ",str_letter
        print "--------------------------------------culPER function end--------------------------------------"
        score = []

        for p, p2 in zip(str_true, str_letter):
            score.append(float(self._levenshtein_distance(p, p2)) / len(p))#これがわからない
        np.savetxt("PERandWER.txt", ["PER," + str(np.average(score))], fmt="%s")

    def culWER(self):
        #単語評価
        str_word = []
        print "--------------------------------------culWER function--------------------------------------"
        print "W_DIC: ", self.s_label_dic

        #こっち推定ラベルいじる
        #self.state_ranges ロードファイル
        for key in self.state_ranges:
            moji = []
            for key2 in key[self.maxlikelihood[1]]:
                moji.append(key2[0])
            str_word.append("".join(map(str, moji)))

        #こっち正解ラベルいじる
        str_true = []
        #2とを無くして，_ splitで２つの配列に文化る
        for key in self.fig_title:
            key = key.replace("2", "")
            wl = key.split("_")#2次配列
            twl = []

            for key2 in wl:
                twl.append(self.s_label_dic[key2])#key2が文字列番号になり，合った数字を返す.
            #twlなんの文字列のラベルかを現している.
            str_true.append("".join(map(str, twl)))#連結して文字列に変換してオワリ

        print "TRUE: ", str_true
        print "SAMP: ", str_word
        print "--------------------------------------culWER function end--------------------------------------"
        score = []
        for w, w2 in zip(str_true, str_word):
            score.append(float(self._levenshtein_distance(w, w2)) / len(w))
        with open('PERandWER.txt', 'a') as f_handle:
            np.savetxt(f_handle, ["WER," + str(np.average(score))], fmt="%s")
        # <<<<<<<<<<<<<<<<<<<<<manipulation functions end!!!>>>>>>>>>>>>>>>>>>>>>#

         # --------------------------------------compute adjusted rand index--------------------------------------#
        def a_rand_index(self, sample_data, true_data, char):
            #ラベルと推定データを引数
            RIs = []
            #
            for idx in range(len(sample_data[0])):#フレーム数のfor文
                true = []
                sample = []
                for key, key2 in zip(sample_data, true_data):#ラベルと推定データの比較のfor文
                    sample.extend(key[idx])#extendは連結する関数 どの状態でも
                    true.extend(key2)
                ris = metrics.adjusted_rand_score(true, sample)
                #ARIスコア エスティメーションでの結果が出力される．ここが一番大事かも
                #ARI= (RI - Expected_RI)/(最大（RI ) - Expected_RI )
                #ARIの意味 Rand Index
                RIs.append(ris)

            np.savetxt("aRIs_" + char + ".txt", RIs)
            true = []
            sample = []
            #ここまで
            for key, key2 in zip(sample_data, true_data):
                sample.extend(key[self.maxlikelihood[1]])
                true.extend(key2)
            ri = metrics.adjusted_rand_score(true, sample)
            str = "maxLk_adjusted_rand_index_" + char + ".txt"
            f = open(str, 'w')
            writer = csv.writer(f)
            writer.writerow(["adjusted_rand_score", ri])



                # --------------------------------------write result_graph--------------------------------------#