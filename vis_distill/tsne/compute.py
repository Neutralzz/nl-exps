import sys,os,json
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics import calinski_harabasz_score as ch_score
import math

data_dir = '/mnt/unilm/hanbao/exp/bert_kd/TEACHER-SST-2'

def process_for_one_layer(data, layer_id):
    X = np.array(data)
    return TSNE(n_jobs=-1).fit_transform(X)


def load_data(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)

    processed_data = {'y': []}
    for i in range(13):
        processed_data['x_%d'%i] = []

    for item in data:
        processed_data['y'].append(item['label'])
        for i in range(13):
            processed_data['x_%d'%i].append(item['layer-%d'%i]) 

    print('load data finished.')
    return processed_data


def dump_processed_data():
    data = load_data(os.path.join(data_dir, 'vectors_with_label.json'))
    #data = load_data(os.join(data_dir, 'vectors_with_label_MM.json'))
    json.dump(data['y'], open('mnli_dev_y.json', 'w', encoding='utf-8'))
    for i in range(13):
        print('processing layer %d'%i)
        X_2 = process_for_one_layer(data['x_%d'%i], i).tolist()
        json.dump(X_2, open('mnli_dev_x2_%d.json'%i, 'w', encoding='utf-8') )

def compute_for_one_layer(x, y, layer_id):
    print('---- Layer %d -----'%layer_id)
    grouped_x = [[],[],[]]
    for item, label in zip(x, y):
        grouped_x[label].append(item)
    record = []
    for i in range(3):
        x_item = np.array(grouped_x[i])
        x_mean = x_item.mean(axis=0)
        x_std  = x_item.std(axis=0)
        bias = math.sqrt(x_std[0]**2 + x_std[1]**2)
        #print('Label %d :'%i, x_mean, bias)
        record.append((x_mean, bias))
    sum_loss = 0
    for i in range(3):
        for j in range(i+1,3):
            p1,p2 = record[i][0], record[j][0]
            d = math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
            print('%d - %d : dist: %.2f loss: %.2f'%(i, j, d, (record[i][1]+record[j][1]) / d))
            sum_loss += (record[i][1]+record[j][1]) / d
    print('Sum: %.2f'%sum_loss)

    #for i in range(3):
    #    print(np.array(grouped_x[i]).mean(), np.array(grouped_x[i]).std())
def compute2_for_one_layer(x, y, layer_id):
    # Calinski-Harabasz
    print('---- Layer %d -----'%layer_id)
    print('CH Score %.2f'%ch_score(np.array(x),np.array(y)))

def direct_compute_ch_score():
    data = load_data(os.path.join(data_dir, 'vectors_with_label.json'))
    x = []
    for i in range(13):
        cs = ch_score(np.array(data['x_%d'%i]), np.array(data['y']))
        x.append(cs)
        print('Layer %d -> CH Score %.2f'%(i, cs))
    for i in range(1, 13):
        print(i-1, max(x[i]/x[i-1], x[i-1]/x[i]))



def main():
    x2=[]
    for i in range(13):
        x2.append(json.load(open('/mnt/v-zhli7/data/mnli_dev_x2_%d.json'%i, 'r', encoding='utf-8')))
    y = json.load(open('/mnt/v-zhli7/data/mnli_dev_y.json', 'r', encoding='utf-8'))
    for i in range(13):
        compute2_for_one_layer(x2[i], y, i)


if __name__=='__main__':
    #main()
    #dump_processed_data()
    direct_compute_ch_score()
