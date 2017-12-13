import Geohash as gh
import pandas as pd
import numpy as np
import pickle
import tqdm
import math
import random
# 写入原有的geohash转码前的结果





def get_geohash():

    x = []
    y = []

    data1 = pd.read_csv("train.csv")
    data2 = pd.read_csv("test.csv")

    hash1 = list(data1["geohashed_start_loc"])
    hash2 = list(data1["geohashed_end_loc"])
    hash3 = list(data2["geohashed_start_loc"])

    #print(hash1[:100],"\n",hash2[:100],"\n",hash3[:100],"\n",(len(hash1)+len(hash2)+len(hash3)))
    _hash = []
    _hash.extend(hash1)
    _hash.extend(hash2)
    _hash.extend(hash3)
    print(_hash.__len__())

    _hash = list(set(_hash))

    print("hash:",len(_hash))

    # 这里遇到的问题：一个经纬度竟然对应多个geohash值，以经纬度作为键，得到的结果是长度为5890的字典，
    # 而以geohash作为键，得到的是长度为11万左右的字典
    # 原因是解码后精度不够高！
    # 也就是说，实际上在一开始数据处理的时候就应该使用精度更高的geohash_exactly!
    # 但是现在没有必要去改，如果为了更加提高精度，就需要去更改
    # TODO：将数据集采用gh.decode_exactly得到精度更高的结果重新处理一下，但是对于原始数据不行，因为之后要贝叶斯分类
    # TODO：所以推荐的方法是只改变将start loc 和end loc变成精度更高的数据进行训练


    for i in _hash:
        t = gh.decode_exactly(i)
        x.append(t[0])
        y.append(t[1])

    x.sort()
    y.sort()

    print(x[0:100])
    print(y[0:100])

    between_x  = []
    between_y = []


    # 这里获得原始具有高精度的geo数据，两两相减，为的就是获取最小的间隔，之后在预测值左右的间隔上获取编码
    for i in range(len(x)-1):
        between_x.append(x[i]-x[i+1])
        between_y.append(y[i] - y[i + 1])
    print(between_x[0:100])
    print(between_y[0:100])
    print(np.mean(np.array(between_x)))
    print(np.mean(np.array(between_y)))



         #  建立经纬度-hashcode字典，这么做是为了保证得出的结果都是在原本的数据集上有的



    #with open("location_to_geohash.pickle","wb") as f:
     #   pickle.dump(location_hash_dic, f)


#get_geohash()

#TODO: 得到数据后，按照原来decode得到的数据的最小间隔，在数据之间取相近点进行encode，得到三个结果

# 这个方案是可以实现的，但是需要500多个小时，采用另一种
def prediction_to_geohash():



    with open("location_to_geohash.pickle","rb") as f:
        dict = pickle.load(f)
    #print(dict,"\n",len(dict))

    predict = np.load("prediction.npy")

    distance = []

    label1 = []
    label2 = []
    label3 = []

    loc = dict.keys()

    x = []
    y = []

    d1 = 100
    d2 = 100
    d3 = 100

    index_d1 = 0
    index_d2 = 0
    index_d3 = 0

    for i in loc:
        print(i)
        temp = i.split("#")
        x.append(temp[0])
        y.append(temp[1])
    x = np.array(x)
    y = np.array(y)

    for i in tqdm.trange(len(predict)):
        _predict = predict[i]
        _x = _predict[0]
        _y = _predict[1]

        for j in range(x.shape[0]):
            distance =  (float(x[j])-float(_x))*(float(x[j])-float(_x)) + \
                        (float(y[j])-float(_y))*(float(y[j])-float(_y))
            if distance < min(d1,d2,d3):
                # 用于求取最小的3个距离，同时也记下最小3个距离的index
                # 存储原来的值
                a = d1
                b = d2
                index_a = index_d1
                index_b = index_d2

                d1 = distance
                index_d1 = j
                d2 = a
                index_d2 = index_a
                d3 = b
                index_d3 = index_b

            elif  distance < min(d2,d3):
                a = d2
                index_a = index_d2

                d2 = distance
                index_d2 = j

                d3 = a
                index_d3 = index_a

            elif distance < d3:
                d3 = distance
                index_d3 = j

        label1.append(dict["{}#{}".format(x[index_d1],y[index_d1])])
        label2.append(dict["{}#{}".format(x[index_d2], y[index_d2])])
        label3.append(dict["{}#{}".format(x[index_d3], y[index_d3])])


        d1 = 100
        d2 = 100
        d3 = 100

        index_d1 = 0
        index_d2 = 0
        index_d3 = 0
        print(label1,"\n",label2,"\n",label3)


# 直接将结果进行geohash code，每列标签全部写一样的，先上传数据看一看
def prediction_to_geohash2():
    predict = np.load("prediction.npy")

    data = pd.read_csv("test.csv")
    data = list(data["orderid"])

    print(len(data))

    print(predict.shape[0])

    label = []
    num = predict.shape[0]
    #num = 100
    for i in tqdm.trange(num):
        temp = predict[i]
        label.append(gh.encode(temp[0],temp[1],precision=7))
    result = ''
    for i in tqdm.trange(len(label)):
        a = random.randint(a=0,b=len(label)-1)
        b = random.randint(a=0,b=len(label)-1)


        # 这里只是尝试，由于lable不允许重复，所以添加随机
        #TODO：尝试用pandas写入而不要用字符串写入文件

        result = result+str(data[i])+","+label[i]+","+label[a]+","+label[b]+"\n"




    with open("submission.csv","w") as f:
        f.write(result)


# 将预测结果加上误差，然后encode，输出结果
def prediction_to_geohash3(data_shift):

    predict = np.load("prediction.npy")
    data = pd.read_csv("test.csv")
    data = list(data["orderid"])


    num = predict.shape[0]

    print(num,len(data))

    label1 = []
    label2 = []
    label3 = []

    for i in tqdm.trange(num):
        temp = predict[i]
        label1.append(gh.encode(temp[0],temp[1],precision=7))
        label2.append(gh.encode(temp[0]+data_shift, temp[1]+data_shift, precision=7))
        label3.append(gh.encode(temp[0] - data_shift, temp[1] - data_shift, precision=7))

    result = pd.DataFrame()

    data = pd.read_csv("test.csv")
    data = list(data["orderid"])

    print(label1[0:100])
    print(label2[0:100])
    print(label3[0:100])

    result.insert(0,"orderid",data)
    result.insert(result.shape[1], "end_location1", pd.Series(label1))
    result.insert(result.shape[1], "end_location2", pd.Series(label1))
    result.insert(result.shape[1], "end_location3", pd.Series(label1))

    result.to_csv("result.csv",index=False)

    # 之后需要删除第一行表头，不要用excel删，否则会有数据损失


prediction_to_geohash3(0.01)







