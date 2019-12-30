import sys,os,json
import numpy as np
#from sklearn.manifold import TSNE
#from sklearn.metrics import calinski_harabasz_score as ch_score
import math
from collections import Counter


'''
[Direct]
Layer 0 -> CH Score 1.00
Layer 1 -> CH Score 4.84
Layer 2 -> CH Score 4.03
Layer 3 -> CH Score 10.37
Layer 4 -> CH Score 62.02
Layer 5 -> CH Score 165.99
Layer 6 -> CH Score 341.42
Layer 7 -> CH Score 331.05
Layer 8 -> CH Score 428.36
Layer 9 -> CH Score 544.28
Layer 10 -> CH Score 684.09
Layer 11 -> CH Score 1190.24
Layer 12 -> CH Score 3812.81

[TSNE]
---- Layer 0 -----
CH Score 2.87
---- Layer 1 -----
CH Score 3.11
---- Layer 2 -----
CH Score 0.27
---- Layer 3 -----
CH Score 0.34
---- Layer 4 -----
CH Score 0.65
---- Layer 5 -----
CH Score 21.32
---- Layer 6 -----
CH Score 1107.35
---- Layer 7 -----
CH Score 1171.58
---- Layer 8 -----
CH Score 1581.22
---- Layer 9 -----
CH Score 1507.65
---- Layer 10 -----
CH Score 1511.02
---- Layer 11 -----
CH Score 4202.30
---- Layer 12 -----
CH Score 7042.32

'''



s1 = [2.87, 3.11, 0.27, 0.34, 0.65, 21.32, 1107.35, 1171.58, 1581.22, 1507.65, 1511.02, 4202.30, 7042.32]
s2 = [1.00, 4.84, 4.03, 10.37, 62.02, 165.99, 341.42, 331.05, 428.36, 544.28, 684.09, 1190.24, 3812.81]

global best_record
best_record = [1e9, (0)]

def dfs(p, n, L, record):
    global best_record
    if n == 0:
        record.append(len(L)-1)
        score = np.array([L[record[i]][1] / L[record[i-1]][1] for i in range(len(record))]).std()
        if best_record[0] > score:
            best_record = [score, record]
        return
    for i in range(p+1, len(L)):
        dfs(i, n-1, L, record+[i])


def draft():
    #print(len(s1), len(s2))
    L = [[0, s2[0]]]
    for i in range(1, 13):
        while s2[i] < L[-1][1]:
            L.pop(-1)
        L.append([i, s2[i]])
    n = 4
    dfs(0, n-1, L, [0])
    for i in best_record[1]:
        print(L[i][0], L[i][1])
    print('------------')
    for i in range(1, len(L)):
        print(L[i][0], L[i][1] / L[i-1][1])


    # 2 / 4.03; 4 / 5.98; 5 / 2.67; 12 / 3.20;
    # 算与前面均值的比值
    # 1      4     5      12    (0-3-4-11)   2 5 6 11 (1-4-5-10)     
    # 4.84   5.98  2.67   3.20
    #assert best_record[0] == np.array([L[record[i]][1] / L[record[i-1]][1] for i in range(len(record))]).std()

global combs, map_layers
map_layers = [ [1, 2, 3], [4], [5], [6,7,8], [9,10], [11,12]]
combs = []

def dfs2(idx, record):
    global combs, map_layers
    if len(record) == 4:
        combs.append(record)
        return
    if idx >= len(map_layers):
        return
    for layer_id in map_layers[idx]:
        dfs2(idx+1, record + [layer_id])
    if idx != 0:
        dfs2(idx+1, record)

def draft2():
    dfs2(0, [])
    del_list = []
    for item in combs:
        if item[-1] not in [10,11,12]:
            del_list.append(item)
        elif (4 not in item) and (5 not in item) and (6 not in item):
            del_list.append(item)
    for item in del_list:
        combs.remove(item)
    with open('exps.txt', 'w', encoding='utf-8') as f:
        for item in combs:
            f.write('%s\n'%('-'.join([str(i-1) for i in item])))

if __name__=='__main__':
    with open('exps.txt', 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f]
    
    with open('exp2', 'w', encoding='utf-8') as f:
        p = 0
        while p < len(lines):
            out = '\'%s\''%lines[p]
            for i in range(1, 24):
                if p+i >= len(lines):
                    break
                out += ' ,\'%s\''%lines[p+i]
            f.write(out+'\n')
            p += 24
