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

def main():
    # データ・セット読み込み#
    dataset =[]
    i="1a"
    with open("./DATA/"+i+".txt") as f:
         for line in f.readlines():
            line = line.rstrip()
            line = line.split("	")
            #print(line)
            dataset.append(line)
    #print(dataset[0][0])
    pca = PCA(n_components=2)
    #dataset = map(float, dataset)
    #print(type(dataset[0][0]))
    dataset=np.array(dataset,dtype=float)
    pca.fit(dataset)
    #print pca.get_covariance()#共分散行列
    #print pca.components_
    X_pca = pca.transform(dataset) # データに対して削減後のベクトルを生成
    print(X_pca)
    fig = plt.figure()
    data = fig.add_subplot(111)#データの確保するための宣言
    data.scatter(X_pca[:,0],X_pca[:,1], c="pink",edgecolors="red")
    data.set_xlabel('first')
    data.set_ylabel('Second')
    plt.show()

if __name__ == '__main__':
    main()