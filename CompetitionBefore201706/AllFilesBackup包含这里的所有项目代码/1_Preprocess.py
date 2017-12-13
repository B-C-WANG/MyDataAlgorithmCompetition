import Geohash as gh
import pandas as pd
#对地址进行decode，分别存储
# finish
def geohash_decode(file,test=False):#约半小时之内跑完
    data = pd.read_csv(file)
    if  test:
        x = []
        y = []
        geohash = data["geohashed_start_loc"]
        b = len(geohash)
        for i in range(len(geohash)):

            print(i,b)
            a = gh.decode(geohash[i])
            x.append(a[0])
            y.append(a[1])
        data=data.drop(labels="geohashed_start_loc",axis=1)
        data.insert(data.shape[1],"start_loc_x",pd.Series(x))
        data.insert(data.shape[1],"start_loc_y",pd.Series(y))
        data.to_csv("test_1.csv",index=False)
    else:
        x1 = []
        y1 = []
        x2 = []
        y2 = []
        geohash1 = data["geohashed_start_loc"]
        geohash2 = data["geohashed_end_loc"]
        b = len(geohash1)
        for i in range(len(geohash1)):
            print(i,b)
            a = gh.decode(geohash1[i])
            x1.append(a[0])
            y1.append(a[1])
        b = len(geohash2)
        for i in range(len(geohash2)):
            print(i,b)
            a = gh.decode(geohash2[i])
            x2.append(a[0])
            y2.append(a[1])
        data=data.drop(labels="geohashed_start_loc", axis=1)
        data=data.drop(labels="geohashed_end_loc", axis=1)
        data.insert(data.shape[1],"start_loc_x" ,pd.Series(x1))
        data.insert(data.shape[1],"start_loc_y" ,pd.Series(y1))
        data.insert(data.shape[1],"end_loc_x", pd.Series(x2))
        data.insert(data.shape[1],"end_loc_y" ,pd.Series(y2))
        data.to_csv("train_1.csv", index=False)




#geohash_decode("test.csv",test=True)
#geohash_decode("train.csv",test=False)


#时间转化，分别存储日以及时间，同时增加一个以5min为精确度的时间，用于刻画用户，
##@# 这个5min划分的时间是标签同时也用作数值！这样把时间分成288类，便于分类，同时相互之间有距离，不完全属于标签！
#属于可以当做数值操作的标签！也就是不用onthot
def time_transform(file,test=False):#半小时之内跑完
    if test:
        data = pd.read_csv(file)
        time_data = data["starttime"]
        date = []
        time = []
        time_min5 = []
        index = 0
        for i in time_data:
            print(index)
            index+=1

            a,b = i.split(" ")
            a = a[-2:]
            h,m,s = b.split(":")
            c = int((int(h)*60+int(m))/5)
            b = int(h)*3600 + int(m)*60 + int(s[:2])#test中这里有小数点
            #print(a,b)
            date.append(a)
            time.append(b)
            time_min5.append(c)
        data = data.drop(labels="starttime",axis=1)
        data.insert(data.shape[1],"date" ,pd.Series(date))
        data.insert(data.shape[1], "time", pd.Series(time))
        data.insert(data.shape[1], "time_min5", pd.Series(time_min5))
        data.to_csv("test_2.csv", index=False)
    else:
        data = pd.read_csv(file)
        time_data = data["starttime"]
        date = []
        time = []
        time_min5 = []
        index=0
        for i in time_data:
            print(index)
            index+=1
            a, b = i.split(" ")
            a = a[-2:]
            h, m, s = b.split(":")
            c = int((int(h) * 60 + int(m)) / 5)
            b = int(h) * 3600 + int(m) * 60 + int(s[:2])  # test中这里有小数点
            #print(a, b)
            date.append(a)
            time.append(b)
            time_min5.append(c)
        data = data.drop(labels="starttime", axis=1)
        data.insert(data.shape[1], "date", pd.Series(date))
        data.insert(data.shape[1], "time", pd.Series(time))
        data.insert(data.shape[1], "time_min5", pd.Series(time_min5))
        data.to_csv("train_2.csv", index=False)









#time_transform("test_1.csv",test=True)
#time_transform("train_1.csv")











