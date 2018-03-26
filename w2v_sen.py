# -*- coding: utf-8 -*-

from gensim.models import word2vec
import sys
import numpy as np
import random
from operator import itemgetter
#import numba

def insert_list(lis,x,in_lis):
    #リストに値を入れるための関数
    lis.insert(x,in_lis)
    return lis

#@numba.jit('f8(f8[:],f8[:])', nopython=True)
def cos_sim(v1, v2):
#コサイン類似度を返す関数
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def main():
    model = word2vec.Word2Vec.load(sys.argv[1])
    wakati = open(sys.argv[2],"r")
    no_wakati = open(sys.argv[3],"r")

    line_wakati = wakati.readlines()
    line_origin = no_wakati.readlines()
    all_vec_point = []

    wakati.close()
    no_wakati.close()

    """
    単語ベクトル→文章ベクトルに変換
    @param line_wakati わかち書きされた文章のリスト (一文目,二文目,...)
    @param line_word 形態素解析された単語のリスト(文中の一つ目の単語,二つ目の単語,...)
    @param vec_point 単語ベクトルの現在の値
    @param model[str] strの単語ベクトルを出力
    @param all_vec_point 単語ベクトルを加算した文章ベクトルの値のリスト(一文目のvec,二文目のvec,...)
    """
    for i in range(len(line_wakati)):
        vec_point = np.zeros(100) #100次元ベクトル
        line_word = (str(line_wakati[i])).split(" ") #形態素ごとに分解
        for j in range(len(line_word)):
            try:
                vec_point = vec_point + model[str(line_word[j])]
            except:
                pass

        insert_list(all_vec_point,len(all_vec_point),vec_point)

    #ランダムで基準(先頭)となる文章を決定
    random_sen = random.randint(0,len(all_vec_point))
    cos_list = []
    sorted_cos_list = []

    for k in range(len(all_vec_point)):
        if k != random_sen: #基準となる文章以外とのcos類似度を計算
            cos_simi = cos_sim(all_vec_point[random_sen],all_vec_point[k])
            insert_list(cos_list,len(cos_list),[line_origin[k].replace('\n','').replace('\u3000',''),cos_simi])
            #表示上見づらくなる要素を排除

    #昇順→降順へ
    sorted_cos_list = sorted(cos_list,key=itemgetter(1),reverse=True)

    #表示形態
    print(line_origin[random_sen])
    for i in range(10):
        print(sorted_cos_list[i][0])
        #print(all_vec_point[i])

    #print(str(len(sorted_cos_list)))

if __name__ == '__main__':
    main()
