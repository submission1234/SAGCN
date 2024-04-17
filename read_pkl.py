import pickle

# rb是2进制编码文件，文本文件用r
f = open(r'coco_adj.pkl', 'rb')
data = pickle.load(f)
print(data)
