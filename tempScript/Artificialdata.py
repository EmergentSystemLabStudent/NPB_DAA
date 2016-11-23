# -*- coding: utf-8 -*-

import numpy as np

import pandas as pd

#人工データを作成するプログラムです
#同じ音は同じくらいの持続長



def main():
    loop = 1
    while loop < 3:
        mean = 5
        deviation =0.1
        a = list(np.random.normal(mean, deviation,7))#最後は乱数の数
        i = list(np.random.normal(mean+1, deviation,8))
        u = list(np.random.normal(mean+2, deviation,4))
        e = list(np.random.normal(mean+3, deviation,5))
        o = list(np.random.normal(mean+4, deviation,6))
        #letterlist = pd.DataFrame([[1, a], [2, i], [3, u], [4, e], [5, o]], columns=['ID', 'data'])
        ai = a + i
        ie = i + e
        ou = o + u
        aue = a + u + e
        ao = a + o


        wordlist = pd.DataFrame([[0,"ai", ai], [1,"ie", ie], [2,"ou", ou], [3,"aue", aue], [4,"ao", ao]], columns=['ID','word','data'])

        #print wordlist[wordlist.ID==1].word.tolist()
        #print wordlist['word'][0]こっちの方がいい


        for j in range(5):
            for k in range(5):
                with open("./"+"dev"+str(deviation)+"/"+wordlist['word'][j]+'_'+wordlist['word'][k]+str(loop)+".txt", "w") as fp:
                    for l in range(len(wordlist['data'][j])):
                        fp.write(str((wordlist['data'][j])[l])+"\n")
                    for l in range(+len(wordlist['data'][k])):
                        fp.write(str((wordlist['data'][k])[l]) + "\n")
        loop += 1


if __name__ == '__main__':
    main()