#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

# =================
#  主成分分析 (PCA)
# =================

#aからzを全部まとめてからPCA

def main():
    # データ・セット読み込み#
    dataset =[]
    colorlist = ["r", "g", "b", "c", "m", "y", "k", "indigo","darkgreen","pink","gray"]
    namesnums = [["1a", "1b"], ["2a", "2b"], ["3a", "3b"], ["4a", "4b"], ["5a", "5b"], ["6a", "6b"], ["7a", "7b"],
                ["8a", "8b"], ["9a", "9b"], ["za", "zb"], ["oa", "ob"]]
    datalen=[[]for i in range(11)]

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
    data = fig.add_subplot(111)#データの確保するための宣言
    sum = 0
    datalen[0][0] = datalen[0][0] - 1

    for i,samedatas in enumerate(datalen):
        print(i)
        for setnumber in samedatas:
            print(colorlist[i])
            print(X_pca[sum:sum+setnumber,0])
            data.scatter(X_pca[sum:sum+setnumber,0],X_pca[sum:sum+setnumber,1], c=colorlist[i],edgecolors=colorlist[i])
            sum = sum + setnumber

    data.set_xlabel('first')
    data.set_ylabel('Second')
    plt.show()


if __name__ == '__main__':
    main()