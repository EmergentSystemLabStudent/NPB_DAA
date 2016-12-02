# -*- coding: utf-8 -*-

import numpy as np
import os
import pandas as pd

#人工データを作成するプログラムです
#同じ音は同じくらいの持続長

def Makeletter(str,mean,deviation):

    if (str == "a"):
        return list(np.random.normal(mean, deviation,20))#２次元データようにデータ長を現在倍に
    elif(str == "i"):
        return list(np.random.normal(mean+1, deviation,22))
    elif(str == "u"):
        return list(np.random.normal(mean+2, deviation,14))
    elif (str == "e"):
        return list(np.random.normal(mean+3, deviation,16))
    elif (str == "o"):
        return list(np.random.normal(mean+4, deviation,26))
    print ("ない音素です")

def Makedataframe(mean,deviation):
    aioi = Makeletter("a", mean, deviation) + Makeletter("i", mean, deviation) + \
           Makeletter("o", mean, deviation) + Makeletter("i", mean, deviation)
    aue = Makeletter("a", mean, deviation) + Makeletter("u", mean, deviation) + Makeletter("e", mean, deviation)
    ao = Makeletter("a", mean, deviation) + Makeletter("o", mean, deviation)
    ie = Makeletter("i", mean, deviation) + Makeletter("e", mean, deviation)
    uo = Makeletter("u", mean, deviation) + Makeletter("o", mean, deviation)

    aioi2 = Makeletter("a", mean, deviation) + Makeletter("i", mean, deviation) + \
            Makeletter("o", mean, deviation) + Makeletter("i", mean, deviation)
    aue2 = Makeletter("a", mean, deviation) + Makeletter("u", mean, deviation) + Makeletter("e", mean, deviation)
    ao2 = Makeletter("a", mean, deviation) + Makeletter("o", mean, deviation)
    ie2 = Makeletter("i", mean, deviation) + Makeletter("e", mean, deviation)
    uo2 = Makeletter("u", mean, deviation) + Makeletter("o", mean, deviation)

    # ループ外の文
    # uo aue ie ,ie ie uo ,aue ao ie ,ao ie ao,aioi uo ie
    return pd.DataFrame([[0, "aioi", aioi], [1, "aue", aue], [2, "ao", ao], [3, "ie", ie], [4, "uo", uo], \
                             [5, "aioi", aioi2], [6, "aue", aue2], [7, "ao", ao2], [8, "ie", ie2], [9, "uo", uo2]],
                            columns=['ID', 'word', 'data'])

def main():
    mean = -2
    deviation = 0.1
    results_dir = "dev"+str(deviation)
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    #letterlist = pd.DataFrame([[1, a], [2, i], [3, u], [4, e], [5, o]], columns=['ID', 'data'])
    loop = 1

    j= 2

    while loop < 10:
        #for j in range(5):
            wordlist = Makedataframe(mean, deviation)
            with open("./" + "dev" + str(deviation) + "/" + wordlist['word'][j] + '_' + wordlist['word'][j] + str(
                    loop) + ".txt", "w") as fp:
                # 同じときの例外処理
                if (wordlist['word'][j] == wordlist['word'][j]):
                    #print str(wordlist['word'][j])
                    print wordlist['data'][j]
                    for l in range(0, len(wordlist['data'][j]), 2):
                        fp.write(str((wordlist['data'][j])[l]) + "\t" + str((wordlist['data'][j])[l + 1]) + "\n")
                    for l in range(0, len(wordlist['data'][j + 5]), 2):
                        fp.write(str((wordlist['data'][j + 5])[l]) + "\t" + str((wordlist['data'][j + 5])[l + 1]) + "\n")
                else:
                    for l in range(0, len(wordlist['data'][j]), 2):
                        fp.write(str((wordlist['data'][j])[l]) + "\t" + str((wordlist['data'][j])[l + 1]) + "\n")
                    for l in range(0, len(wordlist['data'][j]), 2):
                        fp.write(str((wordlist['data'][j])[l]) + "\t" + str((wordlist['data'][j])[l + 1]) + "\n")
            loop += 1

    """
    loop = 1
    while loop < 3:
        for j in range(5):
            for k in range(5):

                #letterlist = pd.DataFrame([[1, a], [2, i], [3, u], [4, e], [5, o]], columns=['ID', 'data'])


                wordlist=Makedataframe(mean,deviation)
                #print wordlist[wordlist.ID==1].word.tolist()
                #print wordlist['word'][0]こっちの方がいい

                with open("./"+"dev"+str(deviation)+"/"+wordlist['word'][j]+'_'+wordlist['word'][k]+str(loop)+".txt", "w") as fp:
                    #同じときの例外処理
                    if(wordlist['word'][j]==wordlist['word'][k]):
                        print str(wordlist['word'][j])
                        for l in range(0,len(wordlist['data'][j]),2):
                            fp.write(str((wordlist['data'][j])[l])+"\t"+str((wordlist['data'][j])[l+1]) + "\n")
                        for l in range(0,len(wordlist['data'][k+5]),2):
                            fp.write(str((wordlist['data'][k+5])[l])+"\t"+str((wordlist['data'][k+5])[l+1]) + "\n")
                    else:
                        for l in range(0,len(wordlist['data'][j]),2):
                            fp.write(str((wordlist['data'][j])[l])+"\t"+str((wordlist['data'][j])[l+1])+"\n")
                        for l in range(0,len(wordlist['data'][k]),2):
                            fp.write(str((wordlist['data'][k])[l])+"\t"+str((wordlist['data'][k])[l+1]) + "\n")

        # uo aue ie ,ie ie uo ,aue ao ie ,ao ie ao,aioi uo ie ここだけべた書き(｢・ω・)｢ｶﾞｵｰ
        wordlist = Makedataframe(mean, deviation)
        for k in [[4,1,3],[3,8,9],[1,2,3],[2,3,2],[0,4,8]]:
            with open("./" + "dev" + str(deviation) + "/" + wordlist['word'][k[0]] + '_' + wordlist['word'][k[1]] +"_"+wordlist['word'][k[2]]+str(loop) + ".txt", "w") as fp:

                for l in range(0,len(wordlist['data'][k[0]]),2):
                    fp.write(str((wordlist['data'][k[0]])[l])+"\t"+str((wordlist['data'][k[0]])[l+1]) + "\n")
                for l in range(0,len(wordlist['data'][k[1]]),2):
                    fp.write(str((wordlist['data'][k[1]])[l])+"\t"+str((wordlist['data'][k[1]])[l+1]) + "\n")
                for l in range(0,len(wordlist['data'][k[2]]),2):
                    fp.write(str((wordlist['data'][k[2]])[l])+"\t"+str((wordlist['data'][k[2]])[l+1]) + "\n")

        loop += 1
        """







if __name__ == '__main__':
    main()