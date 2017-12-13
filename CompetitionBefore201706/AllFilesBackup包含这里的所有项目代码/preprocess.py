import tqdm
import numpy as np
import pandas as pd
from collections import Counter
DATA_PATH = "gy_contest_link_traveltime_training_data.txt"


def read_data():
    data = pd.read_csv(DATA_PATH, delimiter=";")
    # print(data)
    return data

# read_data()


ROAD_DATA_PATH = "gy_contest_link_info.txt"


def read_road_data():
    data = pd.read_csv(ROAD_DATA_PATH,delimiter=";")
    #print(data)
    return data

#read_road_data()

ROAD_LINK_DATA_PATH = "gy_contest_link_top.txt"
def read_road_link_data():
    data = pd.read_csv(ROAD_LINK_DATA_PATH,delimiter=";")
    #print(data)
    return data

#read_road_link_data()


def time_preprocess():
    new_time = []
    data = read_data()
    time = list(data["time_interval"])
    a = len(time)
    # a= 100
    for i in tqdm.trange(a):# 用trange显示进度条
        _time = time[i]
        temp = _time.split(":00,")[0].split("[2016-")[1]#只要第一条，因为第二条就是第一条+2min
        date,_time = temp.split(" ")
        date = int(date.split("-")[0])*100+int(date.split("-")[1])
        _time = int(_time.split(":")[0])*100+int(_time.split(":")[1])
        total_time = date*10000+_time
        # print(total_time)
        new_time.append(total_time)
    count = Counter(new_time)

    #print(count)
    temp = sorted(count.items(), key=lambda item: item[0])  # 按照时间进行计数
    print(temp)
    value = [i[1] for i in temp]
    print(value)
    print(len(temp))

    data.insert(data.shape[1],"frame",pd.Series(new_time))
    #data.sort(["frame"])
    data.sort_values(by="frame")
    data.to_csv("road_frame_data.csv")

#time_preprocess()

#结果：一共68002帧数据，每帧是2分钟，每帧有100~130个数据不等，对于132条路，存在缺失值，这些缺失值可能需要根据上下游
#流量关系确定
#之后的环节，就是绘制道路的树状节点了，对于每一帧，每个街道有一个流量数据，但是怎样衡量和其他道路的上下游关系？
#可以尝试神经网络，用路网关系构建神经网络关系，每一个神经元就是一个道路，数值就是旅行时间，而到另一个道路，需要
#乘以权重和偏置，但是时间上的关系怎么算？可以对于每一个道路，设置一个n x 1的矩阵，代表某十几帧的旅行时间，作为输入
#然后下一时间，各节点的输出作为输出，但是这个网络怎样反向传播？
#也许不用太担心，我们的目的就是训练随时间变化的各道路之间的权重
#但是似乎每一个神经元都需要去求取和下一时间的loss，相当于把一个时间序列变成了多个具有拓扑关系的时间序列，可能很麻烦


#那应该采用怎样的方法？

#能够在一个时间帧内，保持好树状的连通关系，然后能够得到下一帧的预测？



# 时间帧数据处理

# 打算先用其他方法处理，上传数据如果效果好之后再用tensorflow node

def frame_data_process():
    data  = pd.read_csv("road_frame_data.csv")
    try:
        data = data.drop(labels="index",axis=1)
    except Exception as e:
        print("Error",e)

    data = data.drop(labels=["date","time_interval"],axis=1)

    # 只剩下id travel_time 和frame

    dict = {}
    link_id = list(data["link_ID"])
    link_id = list(set(link_id))

    # 为每个id建立一个数据列
    for i in link_id:
        dict[i] = []

    frame = list(data["frame"])
    frame = list(set(frame))
    frame = sorted(frame)
    print(frame[:100])

    a = len(frame)
    #a = 30
    name_length = len(link_id)

    # 对于每一个frame，寻找同样frame下id相同的travel time数据，将其依次append，如果没有找到，则append空字符串
    for i in tqdm.trange(a):
        frame_data = data[data["frame"]==frame[i]]
        for j in range(name_length):
            id_data = frame_data[frame_data["link_ID"]==link_id[j]]
            time = list(id_data["travel_time"])
            if len(time) == 0:
                time = ""
            else:
                time = time[0]
            dict[link_id[j]].append(time)


    data_w = pd.DataFrame()
    # 增加frame
    data_w.insert(0,"frame",frame)

    for i in link_id:

        data_w.insert(data_w.shape[1],i,dict[i])

    data_w.to_csv("result.csv",index=False)


#frame_data_process()


# 之后，对于每一个路段进行lstm，或者卷积lstm


# 首先补充缺失数据

def fill_nan_data():
    data = pd.read_csv("result.csv")
    # 直接采用前一个数据来代替空缺值
    data = data.fillna(method="pad")
    # 如果填补后还为空，那么用后面一个值去代替
    data = data.fillna(method="bfill")

    data.to_csv("filled_result.csv",index=False)




fill_nan_data()











