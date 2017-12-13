import geohash as gh
import numpy as np
import pandas as pd
import tqdm
import profile
import threading


def geohash_decode(hashed_code_l):
    result = []
    for i in hashed_code_l:
        temp = gh.decode_exactly(i)
        #result.append([temp[0],temp[1]])
        result.append([round(temp[0],4), round(temp[1],4)])# 采用4位小数
    return result







# 用于获取最近距离的三个值，输入是location，x和y，以及所有的location标签，输入是前3近的标签
def get_closest_distance(aim_loc, all_location_dict_hash,min_number):

    aim_location = aim_loc
    number = len(all_location_dict_hash)


    location = geohash_decode(all_location_dict_hash)

    #print(location[:100])

    #result = location - aim_loc#注意之后加上绝对值




    #print(result[:100])

    #result = np.sum(result,axis=1)# 按照一维距离相减，得到距离差值



    result = []

    a = len(location)

    #a = 10

    for i in range(a):
        location_ =  [round(location[i][0],4),round(location[i][1],4)]
        result.append(
            abs((aim_location[0]-location_[0])+(aim_location[1]-location_[1]))
        )






    #print("result is",result[:100])
    #print(len(result))

    result_ = np.array(result).argsort()[:min_number][::1]# 一共三个数值，对应最小的三个值
    result = []
    #print(result_)
    for i in result_:
            # 根据最近的3个位置的index，获取最近位置
            result.append(all_location_dict_hash[i])
    return result


# 对于预测到的每个xylocation结果，采用直线距离与标签的location相减，得到之和，取和最小的前三个，对应到相应的geohash作为结果
def make_location_set_method_1():


    data = pd.read_csv("train.csv")
    data_2 = pd.read_csv("test.csv")
    start_loc = list(data["geohashed_start_loc"])
    end_loc = list(data["geohashed_end_loc"])
    start_loc2 = list(data_2["geohashed_start_loc"])

    result = []
    result.extend(start_loc)
    result.extend(end_loc)
    result.extend(start_loc2)
    print(len(result))

    result = list(set(result))

    print(len(result))

    print("location set",result[:100])
    return result





def distance_to_label():
    location_set = make_location_set_method_1()



    predictions = np.load("prediction.npy")
    #print(predictions[:100])
    #print(predictions.shape)
    loc1 = []
    loc2 = []
    loc3 = []


    a  = len(predictions)

    #a = 5


    for i in tqdm.trange(a):
        prediction = predictions[i]
        # 原来是是id相同的进行乘法，现在是对于所有求取距离，所以计算很慢，100次/8min 总共需要 100000/8000min
        # 于是不采用python自带算法，采用numpy的乘法
        #prediction = list(prediction)
        #print(prediction)

        temp  = get_closest_distance(prediction,location_set,3)
        loc1.append(temp[0])
        loc2.append(temp[1])
        loc3.append(temp[2])

    print("loc1 is",loc1)
    print("loc2 is", loc2)
    print("loc3 is", loc3)



distance_to_label()

#profile.run(distance_to_label())