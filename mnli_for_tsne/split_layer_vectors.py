import os,sys,json

data_dir='/home/aistudio/work/output/vecs'

def split(filename):
	data = json.load(open(os.path.join(data_dir, filename), 'r'))
	N = len(data)
	for layer in range(13):
		l_data = []
		for i in range(N):
			l_data.append({
				'label'  : data[i]['label'],
				'vector' : data[i]['layer-%d'%layer]
				})
		json.dump(l_data, open(os.path.join(data_dir, filename.split('.')[0], '%d.json'%layer), 'w', encoding='utf-8'))

if __name__=='__main__':
	split('vectors_with_label_M.json')
	split('vectors_with_label_MM.json')
	
