# confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
# confusion_matrix = np.array(  # gcn
#     [[194, 0,  0,  0,  0,   0,  2,  0, 0, 0],
#      [0, 151,  4,   0, 0,  0, 0, 0, 0, 7],
#      [0, 5,  162,   2, 1,  0, 1, 0, 0, 16],
#      [13, 0,  0,   224, 0,  0, 19, 0, 0, 0],
#      [0, 0,  0,   0, 220,  9, 0, 0, 4, 0],
#      [0, 0,  0,   0, 0,  193, 0, 69, 12, 0],
#      [7, 0,  0,   3, 0,  0, 212, 4, 0, 0],
#      [3, 0,  0,   0, 0,  12, 6, 256, 1, 0],
#      [0, 0,  0,   2, 0,  6, 0, 1, 239, 0],
#      [0, 44,  4,   2, 7,  0, 0, 0, 0, 151],
#      ], dtype=np.int)  # 输入特征矩阵
confusion_matrix = np.array(  # TF-GCN
    [[196, 0,  0,  0,  0,   0,  2,  0, 0, 0],
     [0, 213,  0,   0, 0,  0, 0, 0, 0, 14],
     [0, 0,  212,   0, 1,  0, 0, 0, 0, 0],
     [0, 0,  0,   253, 0,  0, 3, 0, 0, 0],
     [0, 0,  0,   0, 272,  7, 0, 0, 0, 0],
     [0, 0,  0,   0, 0,  212, 0, 62, 0, 0],
     [0, 0,  0,   0, 0,  0, 245, 0, 1, 0],
     [0, 3,  0,   0, 0,  0, 0, 275, 0, 0],
     [0, 0,  0,   2, 0,  0, 0, 0, 244, 0],
     [0, 53,  0,   0, 0,  0, 0, 0, 0, 121],
     ], dtype=np.int)  # 输入特征矩阵

# confusion_matrix = np.array(  # concate
#     [[196, 0,  0,  0,  0,   0,  2,  0, 0, 0],
#      [0, 213,  0,   0, 0,  0, 0, 0, 0, 14],
#      [0, 0,  212,   0, 1,  0, 0, 0, 0, 0],
#      [0, 0,  0,   253, 0,  0, 3, 0, 0, 0],
#      [0, 0,  0,   0, 272,  7, 0, 0, 0, 0],
#      [0, 0,  0,   0, 0,  212, 0, 62, 0, 0],
#      [0, 0,  0,   0, 0,  0, 245, 0, 1, 0],
#      [0, 3,  0,   0, 0,  0, 0, 275, 0, 0],
#      [0, 0,  0,   2, 0,  0, 0, 0, 244, 0],
#      [0, 53,  0,   0, 0,  0, 0, 0, 0, 121],
#      ], dtype=np.int)  # 输入特征矩阵
proportion = []
length = len(confusion_matrix)
print(length)
for i in confusion_matrix:
    for j in i:
        temp = j / (np.sum(i))
        proportion.append(temp)
# print(np.sum(confusion_matrix[0]))
# print(proportion)
pshow = []
for i in proportion:
    pt = "%.2f%%" % (i * 100)
    pshow.append(pt)
proportion = np.array(proportion).reshape(length, length)  # reshape(列的长度，行的长度)
pshow = np.array(pshow).reshape(length, length)
# print(pshow)
config = {
    "font.family": 'Times New Roman',  # 设置字体类型
}
rcParams.update(config)
plt.imshow(proportion, interpolation='nearest', cmap=plt.cm.Blues)  # 按照像素显示出矩阵
# (改变颜色：'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds','YlOrBr', 'YlOrRd',
# 'OrRd', 'PuRd', 'RdPu', 'BuPu','GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn')
# plt.title('confusion_matrix')
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, fontsize=12)
plt.yticks(tick_marks, classes, fontsize=12)

thresh = confusion_matrix.max() / 2.
# iters = [[i,j] for i in range(len(classes)) for j in range((classes))]

iters = np.reshape([[[i, j] for j in range(length)] for i in range(length)], (confusion_matrix.size, 2))
for i, j in iters:
    if (i == j):
        plt.text(j, i - 0.12, format(confusion_matrix[i, j]), va='center', ha='center', fontsize=10, color='white',
                 weight=5)  # 显示对应的数字
        plt.text(j, i + 0.14, pshow[i, j], va='center', ha='center', fontsize=8, color='white')
    else:
        plt.text(j, i - 0.12, format(confusion_matrix[i, j]), va='center', ha='center', fontsize=10)  # 显示对应的数字
        plt.text(j, i + 0.14, pshow[i, j], va='center', ha='center', fontsize=6)

plt.ylabel('True label', fontsize=10)
plt.xlabel('Predict label', fontsize=10)
plt.tight_layout()
plt.show()
plt.savefig('audio_vggish.png')


