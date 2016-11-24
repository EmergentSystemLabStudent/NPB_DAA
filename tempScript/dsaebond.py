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
    foldername= "/Users/pro11/Desktop/DSAE/Linux/TEXTDATA"
    filenames = fnmatch.filter(os.listdir(foldername),"*.txt")
    print filenames
    #ファイルを読み込んで
    #ある結合ファイルに保管
    resultfilename= "filename"

    with open('new.txt', mode='w') as fp,open('setting.txt',mode='w') as fp2:
        for filename in filenames:
            with open(foldername+"/"+filename) as f:
                for n, line in enumerate(f.readlines()):
                        fp.write(line)
                fp2.write("name"+":"+str(filename)+" line:"+str(n+1)+"\n")


if __name__ == '__main__':
    main()