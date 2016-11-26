# -*- coding: utf-8 -*-

import numpy as np
import argparse,os,sys
from paver.easy import pushd
import shutil


def main():

    #setting 読み込み
    fname= "setting"
    summaryname = "./DSAEDATA/8.txt"
    resultsDir = "8DDSAE"
    dataname = []
    summarydata = []

    if not os.path.isdir(resultsDir):
        os.makedirs(resultsDir)

    with open(fname + ".txt") as f:
        print f.readline()
        for n, line in enumerate(f.readlines()):
            line = line.rstrip()
            line = line.split("\t")
            dataname.append(line)
    dataname=np.array(dataname)

    # データ読み込み8DSAE.DATA
    #その名前でデータ渡す

    with open(summaryname) as fp:
        print fp.readline()
        for n, line in enumerate(fp.readlines()):
            line = line.rstrip()
            line = line.split("\t")
            summarydata.append(line)
    #print summarydata[0]]
    #summarydata = np.array(summarydata).astype(np.float64)
    print summarydata

    print summarydata[0]
    length = 0


    for i ,l in zip(dataname[:,0],dataname[:,1]):
        #print i
        #print l
        with open(resultsDir +"/"+str(i)+ ".txt", "w") as fs:
            for k in range(int(l)):
                print (str(summarydata[k + length]).rstrip('[]').replace("\', \'","\t"))+"\n"
                #fs.write((str(summarydata[k+length,:]).strip('[]')).replace("\n","").replace("\n","").replace(" ","\t")+"\n")
                fs.write((str(summarydata[k + length]).strip('[]\'').replace("\', \'","\t"))+"\n")
            length += k


if __name__ == '__main__':
    main()