#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

def main():
    # データ・セット読み込み#
    filename = "DSAEValue"
    # 36フレーム

    fig = plt.figure()
    data = fig.add_subplot(111)  # データの確保するための宣言
    color = ["grey","y"]
    namesnum=[["1a","1b"],["2a","2b"],["3a","3b"],["4a","4b"]]

    for l,names in enumerate(namesnum):
        for i,name in enumerate(names):
            print(name)
            dataset = []
            with open("./DATA/"+str(name)+".txt") as f:
                 for line in f.readlines():
                    line = line.rstrip()
                    line = line.split("	")
                    #print(type(line))
                    line= np.array(line,dtype=float)
                    dataset.append(line)
            #print(dataset[0][0])
            dataset=np.array(dataset,dtype=float)
            #data.scatter(X_pca[:,0],X_pca[:,1], c="pink",edgecolors="red")

            #data.plot(frame, dataset[:,0], label="PF", color="grey")
            #data.plot(frame, dataset[:, 1], label="PF", color="green")
            #data.plot(frame, dataset[:, 2], label="PF", color="m")
            #data.plot(frame, dataset[:, 3], label="PF", color="r")
            #data.plot(frame, dataset[:, 4], label="PF", color="b")
            #data.plot(frame, dataset[:, 5], label="PF", color="k")
            #data.plot(frame, dataset[:, 6], label="PF", color="y")
            #data.plot(frame, dataset[:, 7], label="PF", color="c")
            print(dataset[:,0])
            frame = np.arange(0, dataset.shape[0], 1)
            data.plot(frame, dataset[:,0], label="PF", color=color[i])
            data.plot(frame, dataset[:, 1], label="PF", color=color[i])
            data.plot(frame, dataset[:, 2], label="PF", color=color[i])
            data.plot(frame, dataset[:, 3], label="PF", color=color[i])
            data.plot(frame, dataset[:, 4], label="PF", color=color[i])
            data.plot(frame, dataset[:, 5], label="PF", color=color[i])
            data.plot(frame, dataset[:, 6], label="PF", color=color[i])
            data.plot(frame, dataset[:, 7], label="PF", color=color[i])

        data.set_xlabel('frame')
        data.set_ylabel('FeatureValue')
        #plt.show()
        plt.savefig(filename+str(names[0].rstrip("a"))+".png")

if __name__ == '__main__':
    main()