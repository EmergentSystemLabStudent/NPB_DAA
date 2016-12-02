# -*- coding: utf-8 -*-

import argparse,os,sys
import Summary
from paver.easy import pushd
import shutil
import matplotlib.pyplot as plt
import multiprocessing
import time

#並列処理
def multi_plot_object(summary,idx,count):
    print(summary.fig_title[idx], " plotting...") #タイトル
    summary.plot_states(idx)
    plt.savefig('sample_states_%d.png' % idx)
    summary.plot_state_boundaries(idx)
    plt.savefig('state_boundary_%d.png' % idx)
    #summary.plot_letters(idx)
    #plt.savefig('sample_letters_%d.png' % idx)
    plt.clf()
    count.value = count.value + 1
    print(summary.fig_title[idx], 'plot finish-->count:', count.value)#その状態のときの終了出力



def main():
    parser = argparse.ArgumentParser(description='DATA of NPB_DAA analysis script')
    parser.add_argument('-d',dest='resultdata',help='SummaryData of NPB_DAA')
    #parser.add_argument('-r', required=True) # このオプションは必須です
    parser.add_argument('--version', action='version', version='%(prog)s 1.0') # version
    args = parser.parse_args() # コマンドラインの引数を解釈します
    print(args.resultdata)
    figs_dir = 'summary_figs'
    if(os.path.exists(figs_dir)==True):
        print(figs_dir+" exited.Can I overwrite ?")
        var = raw_input("y or n? >")
        if var =='n':
            print("this program finish")
            sys.exit()
        shutil.rmtree(figs_dir)
    os.mkdir(figs_dir)
    summary = Summary.Summary("./"+args.resultdata)
    #evaluation_result save
    with pushd(figs_dir):

            #gen confused matrix
            #summary.letter_confused_matrix()
            summary.state_confused_matrix()

            # gen PER and WER
            #summary.culPER()
            summary.culWER()

            # gen adjusted rand index
            #summary.a_rand_index(summary.sample_letters, summary.input_data, 'l')
            summary.a_rand_index(summary.sample_states, summary.input_data2, 's')

            # gen word list
            with open('WordList.txt', "w") as f:
                for num, key in enumerate(summary.word_list):#sample_word
                    f.write("iter%d:: " % num)
                    for num2, key2 in enumerate(key):
                        f.write("%d:" % num2 + str(key2) + " ")#key2は単語がどんな音素を持っているかがわかる
                    f.write("\n")

            # multi plot sample states and letters#
            print("--------------------------------------plot process start--------------------------------------")
            count = multiprocessing.Value('i', 0)
            for idx in range(summary.data_size):
                    pr = multiprocessing.Process(target=multi_plot_object, args=(summary, idx, count))
                    pr.start()#並列処理スタート
                    time.sleep(0.1)  # charm...!!!(koreganaito loop karanukenai)
            while (1):
                    if count.value > 55:
                        time.sleep(1)
                        print("--------------------------------------plot process completed!!--------------------------------------")
                        break



if __name__ == '__main__':
    main()
