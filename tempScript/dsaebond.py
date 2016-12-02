#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import numpy as np
import os
import fnmatch

def main():
    foldername= "/Users/pro11/Desktop/DSAE/an/WAVDATA/TEXTDATANOPOWER"
    filenames = fnmatch.filter(os.listdir(foldername),"*.txt")
    print filenames
    #ファイルを読み込んで
    #ある結合ファイルに保管
    resultfilename= "filename"
    Summary = 0
    with open('BondWaveDataNopower.txt', mode='w') as fp,open('settingNopower.txt',mode='w') as fp2:
        fp2.write("name" + "\t" + "line:" + "\n")
        for filename in filenames:
            with open(foldername+"/"+filename) as f:
                for n, line in enumerate(f.readlines()):
                        fp.write(line)
                Summary += n+1
                fp2.write(str(filename).rstrip(".txt")+"\t"+str(n+1)+"\n")
    print Summary

if __name__ == '__main__':
    main()