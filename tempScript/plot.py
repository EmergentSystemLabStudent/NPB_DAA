#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import numpy as np
import os
import fnmatch
import sys


if __name__ == '__main__':
    filename = "./dev0_1/" + sys.argv[1] + ".txt"
    results_dir = "Artificialfiguredata"
    dataset =[]
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    with open(filename) as f:
        for line in f.readlines():
            line = line.rstrip()
            dataset.append(line)
    print dataset
    fig = plt.figure()
    length=len(dataset)
    x = range(length)
    plt.scatter(x,dataset,label = "PF", color = "b")
    plt.savefig(results_dir + "/" +sys.argv[1]+ ".png")
