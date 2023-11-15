#生成数据和可视化
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
iris_1 = load_iris()
#加载鸢尾花数据集
x = iris_1['data']
y = iris_1['target']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)
import pandas as pd
y_train=y_train.reshape(-1,1)

y_train=pd.DataFrame(y_train)
y_train.columns=['type']
y_train1 = pd.get_dummies(y_train.type, prefix='type')
y_train=np.array(y_train1)

NUM_OF_DATA = 150 
CLASIFICATION_TYPE ='ring'

def tag_entry(iris_data):
   print(iris_data[-1])
   """  if CLASIFICATION_TYPE == 'circle':
        if x**2+y**2 > 1: # 如果距离原点半径大于某值，则标为1
            tag = 1
        else:# 小于某值则标为0
            tag = 0
    
    if CLASIFICATION_TYPE == 'ring':
        if x**2+y**2 > 1 and x**2+y**2 < 2: #
            tag = 1
        else:# 
            tag = 0    
    elif CLASIFICATION_TYPE == 'line':
        if x > 0: # 如果距离原点半径大于某值，则标为1
            tag = 1
        else:# 小于某值则标为0
            tag = 0
            
    elif CLASIFICATION_TYPE == 'cross':
        if x*y > 0: # 如果距离原点半径大于某值，则标为1
            tag = 1
        else:# 小于某值则标为0
            tag = 0   
    return tag

def creat_data(num_of_data):
    entry_list = []
    for i in range(num_of_data):
        x = random.uniform(-2, 2)
        y = random.uniform(-2, 2)
        tag = tag_entry(x, y)
        entry = [x, y, tag]
        entry_list.append(entry)   
    return np.array(entry_list)

#---------可视化-----------------
def plot_data(data, title):
    color = []
    for i in data[:, 2]:
        if i == 0:
            color.append("orange")
        else:
            color.append("blue")
    
    plt.scatter(data[:, 0], data[:, 1], c=color)
    plt.title(title)
    plt.show()

# -----------------------
if __name__ == "__main__":
    data = creat_data(NUM_OF_DATA)
    print(data)
    plot_data(data, 'Demo') """
