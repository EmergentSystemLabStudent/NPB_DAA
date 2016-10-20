# -*- coding: utf-8 -*-

import numpy as np
import argparse,os,sys
import Summary
from paver.easy import pushd
##parse する
#=====================summary　class=====================#




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
        os.rmdir(figs_dir)
    os.mkdir(figs_dir)
    summary = Summary.Summary("./"+args.resultdata)
    #evaluation_result save
    with pushd(figs_dir):
            #gen confused matrix
            summary.letter_confused_matrix()
            summary.state_confused_matrix()

if __name__ == '__main__':
    main()
