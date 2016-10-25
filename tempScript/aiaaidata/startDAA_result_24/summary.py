import numpy as np
from sklearn import metrics
import matplotlib
matplotlib.use('Agg') #in the case of perform on server
import matplotlib.pyplot as plt
import json, os, pickle, csv, argparse, multiprocessing, time
from paver.easy import pushd
#--------------------------------------multi process function--------------------------------------#
def multi_plot_object(summary,idx):
    print summary.fig_title[idx], " plotting..."
    summary.plot_states(idx)
    plt.savefig('sample_states_%d.png' % idx)
    summary.plot_state_boundaries(idx)
    plt.savefig('state_boundary_%d.png' % idx)
    summary.plot_letters(idx)
    plt.savefig('sample_letters_%d.png' % idx)
    plt.clf()
    print summary.fig_title[idx], 'plot finish!!!'
#--------------------------------------main function--------------------------------------#
def main():
#result_file make#
    parser = argparse.ArgumentParser()
    parser.add_argument('directory')
    #opts = parser.parse_args()
    figs_dir = 'summary_figs'
    os.path.exists(figs_dir) or os.mkdir(figs_dir)
    summary = Summary()
#evaluation_result save#
    with pushd(figs_dir):
        #gen confused matrix
        summary.letter_confused_matrix()
        summary.state_confused_matrix()
        #gen PER and WER
        summary.culPER()
        summary.culWER()
        #gen adjusted rand index
        summary.a_rand_index(summary.sample_letters,summary.input_data,'l')
        summary.a_rand_index(summary.sample_states,summary.input_data2,'s')
        #gen word list
        with open('WordList.txt',"w") as f:
            for num, key in enumerate(summary.word_list):
                f.write("iter%d:: " % num)
                for num2, key2 in enumerate(key):
                    f.write("%d:" % num2 + str(key2) + " ")
                f.write("\n")
#multi plot sample states and letters#
        print "--------------------------------------plot process start--------------------------------------"
        pr_l = []
        for idx in range(summary.data_size):
            pr = multiprocessing.Process(target=multi_plot_object, args=(summary,idx))
            pr_l.append(pr)
            pr.start()
        for p in pr_l:
            p.join()
        print "--------------------------------------plot process completed!!--------------------------------------"
#=====================summary(main process?) class=====================#
class Summary(object):
#--------------------------------------init paras--------------------------------------#
    def __init__(self, dirpath = '.'):
        with open('parameter.json') as f:
            params = self.params = json.load(f)
        with open('fig_title.json') as f2:
            fig_title = self.fig_title = json.load(f2)
        with open('sample_word_list.txt') as f3:
            self.word_list = pickle.load(f3)
        self.data_size = params['DATA_N']
        self.input_data = [np.loadtxt("../LABEL/"+ i + ".lab") for i in fig_title]
        self.input_data2 = [np.loadtxt("../LABEL/"+ i + ".lab2") for i in fig_title]
        self.sample_states = [np.loadtxt('sample_states_%d.txt' % i)for i in range(params['DATA_N'])]
        self.sample_letters = [np.loadtxt('sample_letters_%d.txt' % i)for i in range(params['DATA_N'])]
        self.state_ranges = []
        for i in range(params['DATA_N']):
            with open('state_ranges_%d.txt' % i) as f:
                self.state_ranges.append(pickle.load(f))
        llist = np.loadtxt("loglikelihood.txt").tolist()
        self.maxlikelihood = (max(llist), llist.index(max(llist)))
        #manipulation part
        self.l_label_dic={}
        self.s_label_dic={}
        #manipulation part end
#--------------------------------------write result_graph--------------------------------------#
#base_graph function#
    def _plot_discreate_sequence(self, true_data, title, sample_data, label = u'', plotopts = {}):
        ax = plt.subplot2grid((10, 1), (1, 0))
        plt.sca(ax)
        ax.matshow([true_data], aspect = 'auto')
        plt.ylabel('Truth Label')
        #label matrix
        ax = plt.subplot2grid((10, 1), (2, 0), rowspan = 8)
        plt.suptitle(title)
        plt.sca(ax)
        ax.matshow(sample_data, aspect = 'auto', **plotopts)
        #write per 10 iterations(max_likelihood) label
        """
        for i in range(label.shape[0]):
            for j in range(label.shape[1]):
                if i%10==0 or i==99 or i==self.maxlikelihood[1]:
                    if i==self.maxlikelihood[1]:
                        ax.text(j, i+1.5, int(label[i][j]), ha='center', va='bottom', color = 'red', fontsize=8)
                    else:
                        ax.text(j, i+1.5, int(label[i][j]), ha='center', va='bottom', color = 'black', fontsize=8)
                    ax.text(j, i+1.5, int(label[i][j]), ha='center', va='bottom', color = 'black', fontsize=8)
        """
        #write x&y label
        plt.xlabel('Frame')
        plt.ylabel('Iteration')
        plt.xticks(())
#plot letter_result graph#
    def plot_letters(self, idx):
        self._plot_discreate_sequence(
            self.input_data[idx],
            self.fig_title[idx],
            self.sample_letters[idx],
            label=self.sample_letters[idx]
        )
#plot state_result graph#
    def plot_states(self, idx):
        self._plot_discreate_sequence(
            self.input_data2[idx],
            self.fig_title[idx],
            self.sample_states[idx],
            label=self.sample_states[idx]
        )
#plot boundary graph#
    def _plot_label_boundary(self, true_data, title, sample_data, label = u''):
        boundaries = [[stop for state, (start, stop) in r] for r in sample_data]
        size = boundaries[0][-1]
        data = np.zeros((len(sample_data), size))
        for i, b in enumerate(boundaries):
            for x in b[:-1]:
                data[i, x] = 1.0
        self._plot_discreate_sequence(true_data, title, data, label, plotopts = {'cmap': 'Greys'})
    def plot_state_boundaries(self, idx):
        self._plot_label_boundary(
            self.input_data2[idx],
            self.fig_title[idx],
            self.state_ranges[idx],
            label=self.sample_states[idx]
        )
#--------------------------------------compute adjusted rand index--------------------------------------#
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
            sample.extend(key[self.maxlikelihood[1]])
            true.extend(key2)
        ri=metrics.adjusted_rand_score(true, sample)
        str="maxLk_adjusted_rand_index_"+char+".txt"
        f = open(str,'w')
        writer = csv.writer(f)
        writer.writerow(["adjusted_rand_score",ri])
#<<<<<<<<<<<<<<<<<<<<<manipulation functions...>>>>>>>>>>>>>>>>>>>>>#
#--------------------------------------letter&state confused matrix function--------------------------------------#
    def letter_confused_matrix(self):
        a=[]
        i=[]
        u=[]
        e=[]
        o=[]
        for key,key2 in zip(self.sample_letters,self.input_data):
            for key3,key4 in zip(key[self.maxlikelihood[1]],key2):
                if key4 == 0:
                    a.append(key3)
                elif key4 == 1:
                    i.append(key3)
                elif key4 == 2:
                    u.append(key3)
                elif key4 == 3:
                    e.append(key3)
                elif key4 == 4:
                    o.append(key3)
        l_max=max(a+i+u+e+o)
        a_count=[]
        i_count=[]
        u_count=[]
        e_count=[]
        o_count=[]
        for num in range(int(l_max)+1):
            a_count.append(a.count(num))
            i_count.append(i.count(num))
            u_count.append(u.count(num))
            e_count.append(e.count(num))
            o_count.append(o.count(num))
        f = open('confused_matrix_l.csv','w')
        writer = csv.writer(f)
        writer.writerow(["phone|letter_label"]+range(int(l_max)+1))
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
    def state_confused_matrix(self):
        aioi=[]
        aue=[]
        ao=[]
        ie=[]
        uo=[]
        for key,key2 in zip(self.sample_states,self.input_data2):
            for key3,key4 in zip(key[self.maxlikelihood[1]],key2):
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
        l_max=max(aioi+aue+ao+ie+uo)
        aioi_count=[]
        aue_count=[]
        ao_count=[]
        ie_count=[]
        uo_count=[]
        for num in range(int(l_max)+1):
            aioi_count.append(aioi.count(num))
            aue_count.append(aue.count(num))
            ao_count.append(ao.count(num))
            ie_count.append(ie.count(num))
            uo_count.append(uo.count(num))
        f = open('confused_matrix_s.csv','w')
        writer = csv.writer(f)
        writer.writerow(["word|state_label"]+range(int(l_max)+1))
        writer.writerow(["aioi"]+aioi_count)
        writer.writerow(["aue"]+aue_count)
        writer.writerow(["ao"]+ao_count)
        writer.writerow(["ie"]+ie_count)
        writer.writerow(["uo"]+uo_count)
        writer.writerow([])
        writer.writerow(["aioi_label:"+str(aioi_count.index(max(aioi_count))),"aue_label:"+str(aue_count.index(max(aue_count))),"ao_label:"+str(ao_count.index(max(ao_count))),"ie_label:"+str(ie_count.index(max(ie_count))),"uo_label:"+str(uo_count.index(max(uo_count)))])
        self.s_label_dic["aioi"]=aioi_count.index(max(aioi_count))
        self.s_label_dic["aue"]=aue_count.index(max(aue_count))
        self.s_label_dic["ao"]=ao_count.index(max(ao_count))
        self.s_label_dic["ie"]=ie_count.index(max(ie_count))
        self.s_label_dic["uo"]=uo_count.index(max(uo_count))
#--------------------------------------culculate PER and WER function--------------------------------------#
    def _levenshtein_distance(self, a, b):
        m = [ [0] * (len(b) + 1) for i in range(len(a) + 1) ]
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
                m[i][j] = min(m[i - 1][j] + 1, m[i][ j - 1] + 1, m[i - 1][j - 1] + x)
        return m[-1][-1]
    def culPER(self):
        str_letter = []
        print "--------------------------------------culPER function--------------------------------------"
        print "P_DIC: ",self.l_label_dic
        for key in self.sample_letters:
            moji=[]
            for count, key2 in enumerate(key[self.maxlikelihood[1]]):
                try:
                    if key2 != key[self.maxlikelihood[1]][count+1]:
                        moji.append(self.l_label_dic[key2])
                except IndexError:
                    try:
                        moji.append(self.l_label_dic[key2])
                    except KeyError:
                        moji.append("*")
                except KeyError:
                    moji.append("*")
            str_letter.append("".join(map(str, moji)))
        str_true = []
        for key in self.fig_title:
            key=key.replace("2", "")
            key=key.replace("_", "")
            str_true.append(key)
        #aioi_ie notokidake
        where = np.where(np.array(str_true)=="aioiie")
        for key in where[0].tolist():
            str_letter[key] = str_letter[key][:-1]+"ie"
        #aioi_ie notokidake end
        print "TRUE: ",str_true
        print "SAMP: ",str_letter
        print "--------------------------------------culPER function end--------------------------------------"
        score=[]
        for p,p2 in zip(str_true,str_letter):
            score.append(float(self._levenshtein_distance(p,p2))/len(p))
        np.savetxt("PERandWER.txt", ["PER,"+str(np.average(score))], fmt="%s")
    def culWER(self):
        str_word = []
        print "--------------------------------------culWER function--------------------------------------"
        print "W_DIC: ",self.s_label_dic
        for key in self.state_ranges:
            moji = []
            for key2 in key[self.maxlikelihood[1]]:
                moji.append(key2[0])
            str_word.append("".join(map(str, moji)))
        str_true = []
        for key in self.fig_title:
            key = key.replace("2", "")
            wl = key.split("_")
            twl = []
            for key2 in wl:
                twl.append(self.s_label_dic[key2])
            str_true.append("".join(map(str, twl)))
        print "TRUE: ",str_true
        print "SAMP: ",str_word
        print "--------------------------------------culWER function end--------------------------------------"
        score=[]
        for w,w2 in zip(str_true,str_word):
            score.append(float(self._levenshtein_distance(w,w2))/len(w))
        with open('PERandWER.txt', 'a') as f_handle:
            np.savetxt(f_handle, ["WER,"+str(np.average(score))], fmt="%s")
#<<<<<<<<<<<<<<<<<<<<<manipulation functions end!!!>>>>>>>>>>>>>>>>>>>>>#
#--------------------------------------direct execution function--------------------------------------#
if __name__ == '__main__':
    main()
