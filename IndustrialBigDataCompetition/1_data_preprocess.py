# 1 将原始数据备份

# 2 设置训练集路径常量

X1_TRAIN = "train/15/15_data.csv"
Y1_TRAIN_0 = "train/15/15_failureInfo.csv"
Y1_TRAIN_1 = "train/15/15_normalInfo.csv"

X2_TRAIN = "train/21/21_data.csv"
Y2_TRAIN_0 = "train/21/21_failureInfo.csv"
Y2_TRAIN_1 = "train/21/21_normalInfo.csv"

# 3 读取数据集，进行初步预处理

import pandas as pd
import numpy as np
import tqdm
a = pd.read_csv

# x1_train = a(X1_TRAIN)
# y1_train_0 = a(Y1_TRAIN_0)
# y1_train_1 = a(Y1_TRAIN_1)
#
# x2_train = a(X2_TRAIN)
# y2_train_0 = a(Y2_TRAIN_0)
# y2_train_1 = a(Y2_TRAIN_1)


# 3-1 对时间进行格式化，写入*_time.csv文件中
# 完成封装
def time_2_int(pd_data):
    pdr=pd_data
    time_l = pdr["time"]#时间列
    num = time_l.shape[0]
    new_time_l = []
    for i in range(num):
        data,time = time_l[i].split(" ")
        y,mouth,d = data.split("-")
        h,m,s = time.split(":")
        time_s = int(y+mouth+d+h+m+s)
        new_time_l.append(time_s)#存储int格式的时间
    data = pd_data.drop(labels="time",axis=1)#删除原来的时间列
    data.insert(0,"time",pd.Series(new_time_l))
    return data

# 3-1-1 对x_train数据进行转换

#time_2_int(x1_train).to_csv(path_or_buf=X1_TRAIN+"_time.csv",index=False)
#time_2_int(x2_train).to_csv(path_or_buf=X2_TRAIN+"_time.csv",index=False)
# 完成封装
def time_2_int_4_y(pd_data):
    pdr = pd_data
    time_l1 = pdr["startTime"]
    time_l2 = pdr["endTime"]
    num1 = time_l1.shape[0]
    #num2 = time_l2.shape[0]
    new_time_l1 = []
    new_time_l2 = []
    for i in range(num1):
        data, time = time_l1[i].split(" ")
        y, mouth, d = data.split("-")
        h, m, s = time.split(":")
        time_s = int(y + mouth + d + h + m + s)
        new_time_l1.append(time_s)

        data, time = time_l2[i].split(" ")
        y, mouth, d = data.split("-")
        h, m, s = time.split(":")
        time_s = int(y + mouth + d + h + m + s)
        new_time_l2.append(time_s)
    data = pd_data.drop(labels="startTime", axis=1).drop(labels="endTime", axis=1)
    data.insert(0, "endTime", pd.Series(new_time_l2))
    data.insert(0, "startTime", pd.Series(new_time_l1))
    return data
# 3-1-2 对y_train数据进行转换

#time_2_int_4_y(y1_train_0).to_csv(path_or_buf=Y1_TRAIN_0+"_time.csv",index=False)
#time_2_int_4_y(y1_train_1).to_csv(path_or_buf=Y1_TRAIN_1+"_time.csv",index=False)
#time_2_int_4_y(y2_train_0).to_csv(path_or_buf=Y2_TRAIN_0+"_time.csv",index=False)
#time_2_int_4_y(y2_train_1).to_csv(path_or_buf=Y2_TRAIN_1+"_time.csv",index=False)


# 3-2 根据y_train的内容对x_train进行提取，得到标注了normal(0)和failure(1)的数据集，


# 完成封装
def get_data_with_label(x_train,start_time_l,end_time_l,label):#输入数据集和startTime，endTime的list，返回start和end之间的数据
    assert len(start_time_l) == len(end_time_l)
    data_l =[]
    for i in range(len(start_time_l)):#对于每一个start, end time列，筛选并存储
        temp = x_train[ (x_train["time"]>start_time_l[i]) &
                               (x_train["time"]<end_time_l[i]) ] #按值筛选
        print(temp.shape)
        data_l.append(temp)

    data = pd.concat(data_l)#按行合并数据
    print(data.shape)
    data.insert(data.shape[1],"label",label)#在行末尾添加label

    return data


# 8天后：重构：直接在符合的数据后边加上label，而不是先筛选出来再合并
# 完成封装
def set_data_with_label():

    train_data1 = pd.read_csv(X1_TRAIN+"_time.csv",dtype="float")
    train_data1_fail = pd.read_csv(Y1_TRAIN_0 + "_time.csv",dtype="float")
    train_data1_normal = pd.read_csv(Y1_TRAIN_1+"_time.csv")

    train_data2 = pd.read_csv(X2_TRAIN+"_time.csv",dtype="float")
    train_data2_fail = pd.read_csv(Y2_TRAIN_0 + "_time.csv",dtype="float")
    train_data2_normal = pd.read_csv(Y2_TRAIN_1 + "_time.csv",dtype="float")

    data1_fail_start = train_data1_fail["startTime"]
    data1_fail_end = train_data1_fail["endTime"]

    data1_normal_start = train_data1_normal["startTime"]
    data1_normal_end = train_data1_normal["endTime"]

    data2_fail_start = train_data2_fail["startTime"]
    data2_fail_end = train_data2_fail["endTime"]

    data2_normal_start = train_data2_normal["startTime"]
    data2_normal_end = train_data2_normal["endTime"]

    data1_time = list(train_data1["time"])
    data2_time = list(train_data2["time"])

    # 用list方法存储，再替换

    label1 = [3  for _ in range(train_data1.shape[0])]
    label2 = [3 for _ in range(train_data2.shape[0])]

    for i in tqdm.trange(len(data1_fail_start)):
        for j in range(data1_time.__len__()):
            if (data1_time[j] >float(data1_fail_start[i])) and (
                        data1_time[j] < float(data1_fail_end[i])):
                label1[j] = 1

    for i in tqdm.trange(len(data1_normal_start)):
                for j in range(data1_time.__len__()):
                    if (data1_time[j] > float(data1_normal_start[i])) and (
                                data1_time[j] < float(data1_normal_end[i])):
                        label1[j] = 0

    for i in tqdm.trange(len(data2_fail_start)):
                        for j in range(data2_time.__len__()):
                            if (data2_time[j] > float(data2_fail_start[i])) and (
                                data2_time[j] < float(data2_fail_end[i])):
                                label2[j] = 1

    for i in tqdm.trange(len(data2_normal_start)):
                        for j in range(data2_time.__len__()):
                            if (data2_time[j] > float(data2_normal_start[i])) and (
                                data2_time[j] < float(data2_normal_end[i])):
                                label2[j] = 0

    label1 = pd.Series(label1)
    label2 = pd.Series(label2)

    train_data1.insert(train_data1.shape[1],"label",label1)
    train_data2.insert(train_data2.shape[1], "label", label2)





    # for i in range(len(data1_fail_start)):
    #
    #
    #
    #
    #
    #     print('\033[1;31;40m',"替换前" , "'\033[0m'")
    #
    #     # 检查type， 确定替换的格式
    #     print(type(train_data1[(train_data1["time"] > float(data1_fail_start[i])) &
    #                       (train_data1["time"] < float(data1_fail_end[i]))]["label"]))
    #
    #     print(train_data1[ (train_data1["time"]>float(data1_fail_start[i])) &
    #                  (train_data1["time"]<float(data1_fail_end[i])) ]["label"])
    #
    #     print('\033[1;31;40m', "》》》》》》》》》》》》》》》》》》》", "'\033[0m'")
    #
    #     train_data1[ (train_data1["time"]>float(data1_fail_start[i])) &
    #                  (train_data1["time"]<float(data1_fail_end[i])) ]["label"].replace(3,1)# 将3替换为1
    #
    #
    #
    #     print('\033[1;31;40m', "替换后", "'\033[0m'")
    #
    #     print(train_data1[(train_data1["time"] > float(data1_fail_start[i])) &
    #                       (train_data1["time"] < float(data1_fail_end[i]))]["label"])
    #
    #     print('\033[1;31;40m str(Exception):\t', "》》》》》》》》》》》》》》》》》》》", "'\033[0m'")
    #
    # for i in range(len(data1_normal_start)):
    #     train_data1[ (train_data1["time"]>float(data1_normal_start[i])) &
    #                  (train_data1["time"]<float(data1_normal_end[i])) ]["label"].replace(3,0)
    #
    # for i in range(len(data2_fail_start)):
    #     train_data2[(train_data2["time"] > float(data2_fail_start[i])) &
    #                 (train_data2["time"] < float(data2_fail_end[i]))]["label"].replace(3,1)  # 将3替换为1
    #
    # for i in range(len(data2_normal_start)):
    #     train_data2[(train_data2["time"] > float(data2_normal_start[i])) &
    #                 (train_data2["time"] < float(data2_normal_end[i]))]["label"].replace(3,0)



    train_data1.to_csv("x1_train_set.csv",index=False)
    train_data2.to_csv("x2_train_set.csv",index=False)

#set_data_with_label()

def delete_label_3():
    data1 = pd.read_csv("x1_train_set.csv")
    data2 = pd.read_csv("x2_train_set.csv")

    data1 = data1[ data1["label"]!= 3 ]
    data2 =  data2[ data2["label"]!= 3 ]


    data1.to_csv("x1_train_set_new.csv",index=False)
    data2.to_csv("x2_train_set_new.csv", index=False)

# 删除label为3的数据
#delete_label_3()

#分别4次载入x1的fail，normal，x2的fail，normal，最后合并，存储
# 完成封装
def run_3_2(dev=False):

    x_train = pd.read_csv(X1_TRAIN+"_time.csv")
    y_train_0 = pd.read_csv(Y1_TRAIN_0+"_time.csv")
    start_time_l = y_train_0["startTime"]
    end_time_l = y_train_0["endTime"]

    x1_train_fail=get_data_with_label(
        x_train=x_train,
        start_time_l=start_time_l,
        end_time_l=end_time_l,
        label=1)


    x_train = pd.read_csv(X1_TRAIN+"_time.csv")
    y_train_0 = pd.read_csv(Y1_TRAIN_1+"_time.csv")
    start_time_l = y_train_0["startTime"]
    end_time_l = y_train_0["endTime"]

    x1_train_normal=get_data_with_label(
        x_train=x_train,
        start_time_l=start_time_l,
        end_time_l=end_time_l,
        label=0)


    x_train = pd.read_csv(X2_TRAIN+"_time.csv")
    y_train_0 = pd.read_csv(Y2_TRAIN_0+"_time.csv")
    start_time_l = y_train_0["startTime"]
    end_time_l = y_train_0["endTime"]

    x2_train_fail=get_data_with_label(
        x_train=x_train,
        start_time_l=start_time_l,
        end_time_l=end_time_l,
        label=1)


    x_train = pd.read_csv(X2_TRAIN+"_time.csv")
    y_train_0 = pd.read_csv(Y2_TRAIN_1+"_time.csv")
    start_time_l = y_train_0["startTime"]
    end_time_l = y_train_0["endTime"]

    x2_train_normal=get_data_with_label(
        x_train=x_train,
        start_time_l=start_time_l,
        end_time_l=end_time_l,
        label=0)


    if dev:#是否要将两个不同文件的数据集分开？（为了LSTM一个作训练，一个作预测）
        pd.concat([x1_train_fail,x1_train_normal]).to_csv("x1_train_set.csv",index=False)
        pd.concat([x2_train_fail, x2_train_normal]).to_csv("x2_train_set.csv", index=False)


    else: pd.concat([x1_train_fail,x1_train_normal,x2_train_fail,x2_train_normal]).to_csv("train_set.csv",index=False)

#run_3_2(dev=True)




# 4 数据预处理和可视化


# 4-1 对数据进行标准化(normalization)并存储结果

import matplotlib.pyplot  as plt
from pandas.tools.plotting import parallel_coordinates
from sklearn import preprocessing
import numpy as np
# 完成封装
def normalization(file_path,save_path):
    data_t = pd.read_csv(file_path)
    data = data_t.drop(labels="time",axis=1)
    new_data = []
    out = ""
    for i in range(27):  # 对于除了label以外的列
        print(i)
        new_data.append(preprocessing.scale(np.array(data.icol(i))))  # 对于每一列，将指标正则化
    new_data.append(np.array(data.icol(27)).astype(int))  # 加上最后一个label指标
    new_data = np.transpose(new_data)

    np.savetxt(fname=save_path,X=np.array(new_data),delimiter=',',newline='\n')

#normalization("train_set.csv","train_set_normalization.csv")


#normalization("x1_train_set_new.csv","x1_train_set_normalization.csv")
#normalization("x2_train_set_new.csv","x2_train_set_normalization.csv")




# 4-2 利用平行坐标进行可视化，可视化所有failure，以及同等数量的normal,采用随机采样得到数据点，避免因为分组造成的偏差
import random


def visualization(file_path,fig_save_path):

    SAMPLE_NUMBER = 3000#作图采用的随机点的个数
    data = pd.read_csv(file_path)
    set1 = data[data.icol(27) >= 0.5]
    set2 = data[data.icol(27) <= 0.5]

    random1 = random.sample(list(range(set1.shape[0])),SAMPLE_NUMBER)#随机采集样本的index
    random2 = random.sample(list(range(set2.shape[0])),SAMPLE_NUMBER)

    result = []
    for i in range(len(random1)):
        result.append(set1[random1[i]:random1[i]+1])#采用[a:b]才能选择行
        result.append(set2[random2[i]:random2[i]+1])
    set = pd.concat(result)
    print(set)
    fig = plt.figure(figsize=(20, 10))
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.axis([0,28,-8,8])#scale : [xmin xmax ymin ymax]
    try:
        parallel_coordinates(set,"1.000000000000000000e+00",alpha=0.3,ax=ax1)#以label作为分类作图,第二个参数需要表头的标签，但是现在表头变成了值，只能用值索引
    except:
        parallel_coordinates(set, "0.000000000000000000e+00", alpha=0.3,
                             ax=ax1)
    plt.savefig(fig_save_path,dpi=150)
    plt.show()

#visualization("train_set_normalization.csv","visualization1.png")


