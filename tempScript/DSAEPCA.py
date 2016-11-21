#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import os

# =================
#  主成分分析 (PCA)
# =================

#aからzを全部まとめてからPCA

def main():
    # データ・セット読み込み#
    dataset =[]
    colorlist = ["r", "g", "b", "c", "m", "y", "k", "indigo","darkgreen","pink","gray"]
    indcolorlist=["r","b"]
    namesnums = [["1a", "1b"], ["2a", "2b"], ["3a", "3b"], ["4a", "4b"], ["5a", "5b"], ["6a", "6b"], ["7a", "7b"],
                ["8a", "8b"], ["9a", "9b"], ["za", "zb"], ["oa", "ob"]]

    datalen=[[]for i in range(11)]

    results_dir = "summaryDSAEPCA"

    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    for l, names in enumerate(namesnums):
        for i, name in enumerate(names):
            with open("./DATA/"+name+".txt") as f:
                for n,line in enumerate(f.readlines()):
                    line = line.rstrip()
                    line = line.split("	")
                    #print(line)
                    dataset.append(line)
                print(l)
                datalen[l].append(n+1)#0も追加
        #print(dataset[0][0])

    #print dataset
    #print( datalen)

    pca = PCA(n_components=2)
    #dataset = map(float, dataset)
    #print(type(dataset[0][0]))
    dataset=np.array(dataset,dtype=float)
    pca.fit(dataset)
    #print pca.get_covariance()#共分散行列
    #print pca.components_
    X_pca = pca.transform(dataset) # データに対して削減後のベクトルを生成
    #print(X_pca)
    fig = plt.figure()
    indfig = plt.figure()
    data = fig.add_subplot(111)#データの確保するための宣言

    sum = 0
    datalen[0][0] = datalen[0][0] - 1

    for i,samedatas in enumerate(datalen):
        print(i)
        inddata = indfig.add_subplot(111)  # データの確保するための宣言
        for l,setnumber in enumerate(samedatas):
            print(colorlist[i])
            print(X_pca[sum:sum+setnumber,0])
            if l == 0 :
                data.scatter(X_pca[sum:sum+setnumber,0],X_pca[sum:sum+setnumber,1], c=colorlist[i],edgecolors=colorlist[i],label=(namesnums[i][0]).rstrip("a"))
            else:
                data.scatter(X_pca[sum:sum + setnumber, 0], X_pca[sum:sum + setnumber, 1], c=colorlist[i],edgecolors=colorlist[i])

            inddata.scatter(X_pca[sum:sum + setnumber, 0], X_pca[sum:sum + setnumber, 1], c=indcolorlist[l],
                         edgecolors=indcolorlist[l],label=namesnums[i][l])
            sum = sum + setnumber
        inddata.set_xlabel('first')
        inddata.set_ylabel('Second')
        inddata.legend(loc="lower right")
        indfig.savefig(results_dir+"/"+"ind"+str(i+1)+ ".png")
        indfig.clf()

    data.set_xlabel('first')
    data.set_ylabel('Second')
    #plt.show()
    data.legend(loc="lower right")
    fig.savefig(results_dir+"/""DSAE8PCA"+".png")


if __name__ == '__main__':
    main()