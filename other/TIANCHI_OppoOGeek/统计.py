# coding:utf-8
import pickle as pkl
import numpy as np
import tqdm
import matplotlib.pyplot as plt
def make_dataset():
    f = open('oppo_round1_train_20180929.txt', 'r', encoding="utf-8")

    success = {} # 统计以搜索词作为key，最终得到的点击率
    total = {}
    for line in f:  # 使用迭代器打开避免一开始开很多内存大
        # print(line)
        line_ = line.split('\t')  # 只要第二个概率数据
        name = line_[0]
        y = int(line_[-1])
        #print(name,y)
        try:
            success[name] += y
            total[name] += 1

        except:
            success[name] = y
            total[name] = 1

    for i in total:
        total[i] = success[i]/total[i]

    with open('train_experience.pkl', "wb") as f:
        pkl.dump(total,f)






def load_and_predict():
    with open('train_experience.pkl', "rb") as f:
        data = pkl.load(f)
    f = open('oppo_round1_test_A_20180929.txt', 'r', encoding="utf-8")
    x = []
    for line in f:  # 使用迭代器打开避免一开始开很多内存大
        # print(line)
        line_ = line.split('\t')  # 只要第二个概率数据
        name = line_[0]
        x.append(name)
    answer = []
    for name in x:
        try:
            if data[name] >= 0.5: # 统计高于0.5，判为正例
                answer.append(1)
            else:
                answer.append(0)
        except:
            answer.append(0)
    string = ''
    for i in answer:
        string += "%s\n" % i
    with open('answer.csv', "w") as f:
        f.write(string)
def valid():
    with open('train_experience.pkl', "rb") as f:
        data = pkl.load(f)
    f = open('oppo_round1_vali_20180929.txt', 'r', encoding="utf-8")
    x = []
    y = []
    for line in f:  # 使用迭代器打开避免一开始开很多内存大
        # print(line)
        line_ = line.split('\t')  # 只要第二个概率数据
        name = line_[0]
        _y = int(line_[-1])
        x.append(name)
        y.append(_y)
    answer = []
    for name in x:
        try:
            if data[name] >= 0.5: # 统计高于0.5，判为正例
                answer.append(1)
            else:
                answer.append(0)
        except:
            answer.append(0)
    equal = np.array(answer) == np.array(y)
    print(np.sum(equal)/len(answer))


#make_dataset()
#load_and_predict()
valid()


