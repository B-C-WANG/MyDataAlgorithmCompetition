import  numpy as np
import pandas as pd
from collections import Counter
import tqdm
import profile
import pandas
# abandon
def count_(data_file_name):
    data = open(data_file_name,"r").read()
    a = data.split("\n")
    b = []
    dic={}
    for i in a:
        b.append(i.split(","))
    for i in range(len(b[0])):
        dic["{}".format(b[0][i])] = []
    for i in range(1,len(b)):
        temp=b[i]
        for j in range(len(b[i])):
            dic["{}".format(b[0][j])].append(b[i][j])
    for i in dic.values():

        print(len(set(i)))

    print(dic.keys())
#
# data = pandas.read_csv("train.csv")
#
# print(data.describe(include="all"))
#
#
# data = pandas.read_csv("test.csv")
#
# print(data.describe(include="all"))

#以下的处理要保证test和train一视同仁
#首先对bikeID处理，bikeID只与location有关！
#输入location和bikeid，得到用最常用三个location及其count共6个值，如果不足三个，设为最后一个，count为0


## ！！注意，转换过后的bikeid要写在新的表中，并且以bikeid作为索引！！！，不要在原表上加
def bikeID_to_bike_loc_1_2_3(filename,out_name):#1.5-2小时内跑完
    data = pd.read_csv(filename)
    bike_id = list(set(data["bikeid"]))
    a = len(list(bike_id))
    loc_1_x = []
    loc_1_y = []
    count_1 = []
    loc_2_x = []
    loc_2_y = []
    count_2 = []
    loc_3_x = []
    loc_3_y = []
    count_3 = []
    index = 0
    for i in bike_id:
    #for i in range(10):
        if index%1000 ==0:
            print(index, a)
        index+=1
        temp = data[(data["bikeid"]==i)]
        #temp = data[(data["bikeid"] == bike_id[i])]
        x = list(temp["start_loc_x"])
        y = list(temp["start_loc_y"])
        loc = [(x[i],y[i]) for i in range(len(x))]
        count = Counter(loc)
        total = sum(count.values())
        #print(count)
        temp = sorted(count.items(), key=lambda item:-item[1])#按照逆序进行一次排列，然后得到key等
        key = [i[0] for i in temp]
        value = [i[1]/total for i in temp]
        #print(key,"\n",value)
        if len(count) >=3:
            loc_1_x.append(key[0][0])
            loc_2_x.append(key[1][0])
            loc_3_x.append(key[2][0])
            loc_1_y.append(key[0][1])
            loc_2_y.append(key[1][1])
            loc_3_y.append(key[2][1])
            count_1.append(value[0])
            count_2.append(value[1])
            count_3.append(value[2])
        elif len(count) == 2:
            loc_1_x.append(key[0][0])
            loc_2_x.append(key[1][0])
            loc_3_x.append(key[1][0])
            loc_1_y.append(key[0][1])
            loc_2_y.append(key[1][1])
            loc_3_y.append(key[1][1])
            count_1.append(value[0])
            count_2.append(value[1])
            count_3.append(0)
        elif len(count) == 1:
            loc_1_x.append(key[0][0])
            loc_2_x.append(key[0][0])
            loc_3_x.append(key[0][0])
            loc_1_y.append(key[0][1])
            loc_2_y.append(key[0][1])
            loc_3_y.append(key[0][1])
            count_1.append(value[0])
            count_2.append(0)
            count_3.append(0)
        else:
            print("some bike have no order",len(count))

    #原来的data不需要drop，新建一个表，作为索引
    data_w = pd.DataFrame()
    data_w.insert(data_w.shape[1], "bikeid", pd.Series(bike_id))
    data_w.insert(data_w.shape[1], "bike_loc_1_x", pd.Series(loc_1_x))
    data_w.insert(data_w.shape[1], "bike_loc_1_y", pd.Series(loc_1_y))
    data_w.insert(data_w.shape[1], "bike_loc_1_count", pd.Series(count_1))
    data_w.insert(data_w.shape[1], "bike_loc_2_x", pd.Series(loc_2_x))
    data_w.insert(data_w.shape[1], "bike_loc_2_y", pd.Series(loc_2_y))
    data_w.insert(data_w.shape[1], "bike_loc_2_count", pd.Series(count_2))
    data_w.insert(data_w.shape[1], "bike_loc_3_x", pd.Series(loc_3_x))
    data_w.insert(data_w.shape[1], "bike_loc_3_y", pd.Series(loc_3_y))
    data_w.insert(data_w.shape[1], "bike_loc_3_count", pd.Series(count_3))
    data_w.to_csv("{}._bike_to_loc.csv".format(out_name), index=False)






#bikeID_to_bike_loc_1_2_3("test_2.csv","test_2")
#bikeID_to_bike_loc_1_2_3("train_2.csv","train_2")



    #将用户ID用特征table进行替换
def apply_usrID_table(read_filename,write_filename,test=False):#大约耗时20小时
    bike1 = []
    bike1_c = []
    bike2 = []
    bike2_c = []
    bike3 = []
    bike3_c = []

    biketype1_p = []

    loc_1_x = []
    loc_1_y = []
    count_1 = []
    loc_2_x = []
    loc_2_y = []
    count_2 = []
    loc_3_x = []
    loc_3_y = []
    count_3 = []

    time1 = []
    time1_c = []
    time2 = []
    time2_c = []
    time3 = []
    time3_c = []
    if test:
        usr_re = pd.read_csv("usr_table_test.csv")
    else:
        usr_re = pd.read_csv("usr_table_train.csv")
    data = pd.read_csv(read_filename)
    usr_id = list(data["userid"])
    usr_num = len(usr_id)
    #usr_num =100
    for i in tqdm.trange(usr_num):
        temp_ = usr_re[(usr_re["userid"]==usr_id[i])]

        temp = np.array(temp_)[0]

        loc_1_x.append(temp[1])
        loc_1_y.append(temp[2])
        count_1.append(temp[3])
        loc_2_x.append(temp[4])
        loc_2_y.append(temp[5])
        count_2.append(temp[6])
        loc_3_x.append(temp[7])
        loc_3_y.append(temp[8])
        count_3.append(temp[9])

        bike1.append(temp[10])
        bike1_c.append(temp[11])
        bike2.append(temp[12])
        bike2_c.append(temp[13])
        bike3.append(temp[14])
        bike3_c.append(temp[15])

        biketype1_p.append(temp[16])

        time1.append(temp[17])
        time1_c.append(temp[18])
        time2.append(temp[19])
        time2_c.append(temp[20])
        time3.append(temp[21])
        time3_c.append(temp[22])
    data_w = data
    #data_w.drop(labels="userid",axis=1)#宁愿保留也暂时不要drop

    data_w.insert(data_w.shape[1], "usr_start_loc_1_x", pd.Series(loc_1_x))
    data_w.insert(data_w.shape[1], "usr_start_loc_1_y", pd.Series(loc_1_y))
    data_w.insert(data_w.shape[1], "usr_start_loc_1_count", pd.Series(count_1))
    data_w.insert(data_w.shape[1], "usr_start_loc_2_x", pd.Series(loc_2_x))
    data_w.insert(data_w.shape[1], "usr_start_loc_2_y", pd.Series(loc_2_y))
    data_w.insert(data_w.shape[1], "usr_start_loc_2_count", pd.Series(count_2))
    data_w.insert(data_w.shape[1], "usr_start_loc_3_x", pd.Series(loc_3_x))
    data_w.insert(data_w.shape[1], "usr_start_loc_3_y", pd.Series(loc_3_y))
    data_w.insert(data_w.shape[1], "usr_start_loc_3_count", pd.Series(count_3))

    data_w.insert(data_w.shape[1], "bike1", pd.Series(bike1))
    data_w.insert(data_w.shape[1], "bike1_c", pd.Series(bike1_c))
    data_w.insert(data_w.shape[1], "bike2", pd.Series(bike2))
    data_w.insert(data_w.shape[1], "bike2_c", pd.Series(bike2_c))
    data_w.insert(data_w.shape[1], "bike3", pd.Series(bike3))
    data_w.insert(data_w.shape[1], "bike3_c", pd.Series(bike3_c))

    data_w.insert(data_w.shape[1], "biketype1_p", pd.Series(biketype1_p))

    data_w.insert(data_w.shape[1], "time1", pd.Series(time1))
    data_w.insert(data_w.shape[1], "time1_c", pd.Series(time1_c))
    data_w.insert(data_w.shape[1], "time2", pd.Series(time2))
    data_w.insert(data_w.shape[1], "time2_c", pd.Series(time2_c))
    data_w.insert(data_w.shape[1], "time3", pd.Series(time3))
    data_w.insert(data_w.shape[1], "time3_c", pd.Series(time3_c))

    data_w.to_csv(write_filename, index=False)

#profile.run('apply_usrID_table("test_2.csv","usr_test_2.csv",test=True)')
#profile.run('apply_usrID_table("train_2.csv","usr_train_2.csv",test=False)')






#按照bike_to_loc的索引，对原来数据的bikeid进行替换，

def replace_bike_id(filename,out_name,test=False):
    bike_loc_1_x = []
    bike_loc_1_y = []
    bike_loc_1_count = []
    bike_loc_2_x = []
    bike_loc_2_y =[]
    bike_loc_2_count = []
    bike_loc_3_x =[]
    bike_loc_3_y = []
    bike_loc_3_count = []

    user_bike1_loc_1_x = []
    user_bike1_loc_1_y = []
    user_bike1_loc_1_count = []
    user_bike1_loc_2_x = []
    user_bike1_loc_2_y = []
    user_bike1_loc_2_count = []
    user_bike1_loc_3_x = []
    user_bike1_loc_3_y = []
    user_bike1_loc_3_count = []

    user_bike2_loc_1_x = []
    user_bike2_loc_1_y = []
    user_bike2_loc_1_count = []
    user_bike2_loc_2_x = []
    user_bike2_loc_2_y = []
    user_bike2_loc_2_count = []
    user_bike2_loc_3_x = []
    user_bike2_loc_3_y = []
    user_bike2_loc_3_count = []

    user_bike3_loc_1_x = []
    user_bike3_loc_1_y = []
    user_bike3_loc_1_count = []
    user_bike3_loc_2_x = []
    user_bike3_loc_2_y = []
    user_bike3_loc_2_count = []
    user_bike3_loc_3_x = []
    user_bike3_loc_3_y = []
    user_bike3_loc_3_count = []
    if test:
        bike_rep = pd.read_csv("test_2._bike_to_loc.csv")
    else:
        bike_rep = pd.read_csv("train_2._bike_to_loc.csv")
    data = pd.read_csv(filename)





    bike_id = list(data["bikeid"])
    a = len(bike_id)

    for i in tqdm.trange(a):

            temp_ = bike_rep[(bike_rep["bikeid"] == bike_id[i])]
            temp = np.array(temp_)
            temp = temp[0]

            bike_loc_1_x.append(temp[1])
            bike_loc_1_y.append(temp[2])
            bike_loc_1_count.append(temp[3])
            bike_loc_2_x.append(temp[4])
            bike_loc_2_y.append(temp[5])
            bike_loc_2_count.append(temp[6])
            bike_loc_3_x.append(temp[7])
            bike_loc_3_y.append(temp[8])
            bike_loc_3_count.append(temp[9])

    data.insert(data.shape[1], "bike_loc_1_x", pd.Series(bike_loc_1_x))
    data.insert(data.shape[1], "bike_loc_1_y", pd.Series(bike_loc_1_y))
    data.insert(data.shape[1], "bike_loc_1_count", pd.Series(bike_loc_1_count))
    data.insert(data.shape[1], "bike_loc_2_x", pd.Series(bike_loc_2_x))
    data.insert(data.shape[1], "bike_loc_2_y", pd.Series(bike_loc_2_y))
    data.insert(data.shape[1], "bike_loc_2_count", pd.Series(bike_loc_2_count))
    data.insert(data.shape[1], "bike_loc_3_x", pd.Series(bike_loc_3_x))
    data.insert(data.shape[1], "bike_loc_3_y", pd.Series(bike_loc_3_y))
    data.insert(data.shape[1], "bike_loc_3_count", pd.Series(bike_loc_3_count))

    del bike_loc_1_x
    del bike_loc_1_y
    del bike_loc_1_count
    del bike_loc_2_x
    del bike_loc_2_y
    del bike_loc_2_count
    del bike_loc_3_x
    del bike_loc_3_y
    del bike_loc_3_count
    del bike_id


    bike_id = list(data["bike1"])
    for i in tqdm.trange(a):

            temp_ = bike_rep[(bike_rep["bikeid"] == bike_id[i])]
            temp = np.array(temp_)
            temp = temp[0]

            user_bike1_loc_1_x.append(temp[1])
            user_bike1_loc_1_y.append(temp[2])
            user_bike1_loc_1_count.append(temp[3])
            user_bike1_loc_2_x.append(temp[4])
            user_bike1_loc_2_y.append(temp[5])
            user_bike1_loc_2_count.append(temp[6])
            user_bike1_loc_3_x.append(temp[7])
            user_bike1_loc_3_y.append(temp[8])
            user_bike1_loc_3_count.append(temp[9])

    data.insert(data.shape[1], "user_bike1_loc_1_x", pd.Series(user_bike1_loc_1_x))
    data.insert(data.shape[1], "user_bike1_loc_1_y", pd.Series(user_bike1_loc_1_y))
    data.insert(data.shape[1], "user_bike1_loc_1_count", pd.Series(user_bike1_loc_1_count))
    data.insert(data.shape[1], "user_bike1_loc_2_x", pd.Series(user_bike1_loc_2_x))
    data.insert(data.shape[1], "user_bike1_loc_2_y", pd.Series(user_bike1_loc_2_y))
    data.insert(data.shape[1], "user_bike1_loc_2_count", pd.Series(user_bike1_loc_2_count))
    data.insert(data.shape[1], "user_bike1_loc_3_x", pd.Series(user_bike1_loc_3_x))
    data.insert(data.shape[1], "user_bike1_loc_3_y", pd.Series(user_bike1_loc_3_y))
    data.insert(data.shape[1], "user_bike1_loc_3_count", pd.Series(user_bike1_loc_3_count))

    del user_bike1_loc_1_x
    del user_bike1_loc_1_y
    del user_bike1_loc_1_count
    del user_bike1_loc_2_x
    del user_bike1_loc_2_y
    del user_bike1_loc_2_count
    del user_bike1_loc_3_x
    del user_bike1_loc_3_y
    del user_bike1_loc_3_count
    del bike_id



    bike_id = list(data["bike2"])
    for i in tqdm.trange(a):

            temp_ = bike_rep[(bike_rep["bikeid"] == bike_id[i])]
            temp = np.array(temp_)
            temp = temp[0]


            user_bike2_loc_1_x.append(temp[1])
            user_bike2_loc_1_y.append(temp[2])
            user_bike2_loc_1_count.append(temp[3])
            user_bike2_loc_2_x.append(temp[4])
            user_bike2_loc_2_y.append(temp[5])
            user_bike2_loc_2_count.append(temp[6])
            user_bike2_loc_3_x.append(temp[7])
            user_bike2_loc_3_y.append(temp[8])
            user_bike2_loc_3_count.append(temp[9])

    data.insert(data.shape[1], "user_bike2_loc_1_x", pd.Series(user_bike2_loc_1_x))
    data.insert(data.shape[1], "user_bike2_loc_1_y", pd.Series(user_bike2_loc_1_y))
    data.insert(data.shape[1], "user_bike2_loc_1_count", pd.Series(user_bike2_loc_1_count))
    data.insert(data.shape[1], "user_bike2_loc_2_x", pd.Series(user_bike2_loc_2_x))
    data.insert(data.shape[1], "user_bike2_loc_2_y", pd.Series(user_bike2_loc_2_y))
    data.insert(data.shape[1], "user_bike2_loc_2_count", pd.Series(user_bike2_loc_2_count))
    data.insert(data.shape[1], "user_bike2_loc_3_x", pd.Series(user_bike2_loc_3_x))
    data.insert(data.shape[1], "user_bike2_loc_3_y", pd.Series(user_bike2_loc_3_y))
    data.insert(data.shape[1], "user_bike2_loc_3_count", pd.Series(user_bike2_loc_3_count))

    del user_bike2_loc_1_x
    del user_bike2_loc_1_y
    del user_bike2_loc_1_count
    del user_bike2_loc_2_x
    del user_bike2_loc_2_y
    del user_bike2_loc_2_count
    del user_bike2_loc_3_x
    del user_bike2_loc_3_y
    del user_bike2_loc_3_count
    del bike_id



    bike_id = list(data["bike3"])
    for i in tqdm.trange(a):

            temp_ = bike_rep[(bike_rep["bikeid"] == bike_id[i])]
            temp = np.array(temp_)
            temp = temp[0]

            user_bike3_loc_1_x.append(temp[1])
            user_bike3_loc_1_y.append(temp[2])
            user_bike3_loc_1_count.append(temp[3])
            user_bike3_loc_2_x.append(temp[4])
            user_bike3_loc_2_y.append(temp[5])
            user_bike3_loc_2_count.append(temp[6])
            user_bike3_loc_3_x.append(temp[7])
            user_bike3_loc_3_y.append(temp[8])
            user_bike3_loc_3_count.append(temp[9])


    data.insert(data.shape[1], "user_bike3_loc_1_x", pd.Series(user_bike3_loc_1_x))
    data.insert(data.shape[1], "user_bike3_loc_1_y", pd.Series(user_bike3_loc_1_y))
    data.insert(data.shape[1], "user_bike3_loc_1_count", pd.Series(user_bike3_loc_1_count))
    data.insert(data.shape[1], "user_bike3_loc_2_x", pd.Series(user_bike3_loc_2_x))
    data.insert(data.shape[1], "user_bike3_loc_2_y", pd.Series(user_bike3_loc_2_y))
    data.insert(data.shape[1], "user_bike3_loc_2_count", pd.Series(user_bike3_loc_2_count))
    data.insert(data.shape[1], "user_bike3_loc_3_x", pd.Series(user_bike3_loc_3_x))
    data.insert(data.shape[1], "user_bike3_loc_3_y", pd.Series(user_bike3_loc_3_y))
    data.insert(data.shape[1], "user_bike3_loc_3_count", pd.Series(user_bike3_loc_3_count))

    del user_bike3_loc_1_x
    del user_bike3_loc_1_y
    del user_bike3_loc_1_count
    del user_bike3_loc_2_x
    del user_bike3_loc_2_y
    del user_bike3_loc_2_count
    del user_bike3_loc_3_x
    del user_bike3_loc_3_y
    del user_bike3_loc_3_count
    del bike_id


    # data = data.drop(labels="bikeid",axis=1)#宁愿保留也暂时不要drop
    data.to_csv(out_name, index=False)





#replace_bike_id("usr_test_2.csv","bike_usr_test_2.csv",test=True)
#replace_bike_id("usr_train_2.csv","bike_usr_train_2.csv")


#TODO：最后替换时，有一列是用户bikeid，然后是原本的bikeid






#之后处理用户，用户与bikeID给出的location有关，与time而不是date有关，与bike type=1的概率有关，
# 建立用户模式表，第一列为用户id，第2,3,4为常用3个bikeid（没有替换之前），第5为常用biketype=1的频率，第6至15为常用location和count
#第16至25为常用时间和count（时间是精确到5分钟的时间）
#修正：所有的前三频次全部改成概率！名称还是叫count不变，但是内容是概率！
def usrID_table_create(filename, out_name):#大概要跑20小时左右
            data = pd.read_csv(filename)
            userid = list(set(data["userid"]))
            a = len(list(userid))
            #a = 10
            bike1 = []
            bike1_c = []
            bike2 = []
            bike2_c = []
            bike3 = []
            bike3_c = []

            biketype1_p = []

            loc_1_x = []
            loc_1_y = []
            count_1 = []
            loc_2_x = []
            loc_2_y = []
            count_2 = []
            loc_3_x = []
            loc_3_y = []
            count_3 = []

            time1 = []
            time1_c = []
            time2 = []
            time2_c = []
            time3 =[]
            time3_c = []


            index = 0


            #location
            for i in range(a):
                usrdata = data[(data["userid"] == userid[i])]

                loc_x = list(usrdata["start_loc_x"])
                loc_y = list(usrdata["start_loc_y"])
                loc = [(loc_x[i], loc_y[i]) for i in range(len(loc_x))]

                count = Counter(loc)
                total = sum(count.values())
                #print(count,total)
                temp = sorted(count.items(), key=lambda item: -item[1])  # 按照逆序进行一次排列，然后得到key等
                key = [i[0] for i in temp]
                value = [i[1]/total for i in temp]
                # print(key,"\n",value)
                if len(count) >= 3:
                    loc_1_x.append(key[0][0])
                    loc_2_x.append(key[1][0])
                    loc_3_x.append(key[2][0])
                    loc_1_y.append(key[0][1])
                    loc_2_y.append(key[1][1])
                    loc_3_y.append(key[2][1])
                    count_1.append(value[0])
                    count_2.append(value[1])
                    count_3.append(value[2])
                elif len(count) == 2:
                    loc_1_x.append(key[0][0])
                    loc_2_x.append(key[1][0])
                    loc_3_x.append(key[1][0])
                    loc_1_y.append(key[0][1])
                    loc_2_y.append(key[1][1])
                    loc_3_y.append(key[1][1])
                    count_1.append(value[0])
                    count_2.append(value[1])
                    count_3.append(0)
                elif len(count) == 1:
                    loc_1_x.append(key[0][0])
                    loc_2_x.append(key[0][0])
                    loc_3_x.append(key[0][0])
                    loc_1_y.append(key[0][1])
                    loc_2_y.append(key[0][1])
                    loc_3_y.append(key[0][1])
                    count_1.append(value[0])
                    count_2.append(0)
                    count_3.append(0)
                elif len(count) == 0:
                    loc_1_x.append(0)
                    loc_2_x.append(0)
                    loc_3_x.append(0)
                    loc_1_y.append(0)
                    loc_2_y.append(0)
                    loc_3_y.append(0)
                    count_1.append(0)
                    count_2.append(0)
                    count_3.append(0)

                else:
                    print("some bike have no order", len(count))


                index += 1
                if index % 1000 == 0:
                    print(index, 4 * a)


            # bikeid
            for i in range(a):

                usrdata = data[(data["userid"] == userid[i])]

                bikes = list(usrdata["bikeid"])
                count = Counter(bikes)
                #print(count)
                total = sum(count.values())
                temp = sorted(count.items(), key=lambda item: -item[1])
                key = [i[0] for i in temp]
                value = [i[1] / total for i in temp]
                if len(count) >= 3:
                    bike1.append(key[0])
                    bike2.append(key[1])
                    bike3.append(key[2])
                    bike1_c.append(value[0])
                    bike2_c.append(value[1])
                    bike3_c.append(value[2])
                elif len(count) == 2:
                    bike1.append(key[0])
                    bike2.append(key[1])
                    bike3.append(key[1])

                    bike1_c.append(value[0])
                    bike2_c.append(value[1])
                    bike3_c.append(0)
                elif len(count) == 1:
                    bike1.append(key[0])
                    bike2.append(key[0])
                    bike3.append(key[0])

                    bike1_c.append(value[0])
                    bike2_c.append(0)
                    bike3_c.append(0)
                elif len(count) == 0:
                    bike1.append(0)
                    bike2.append(0)
                    bike3.append(0)

                    bike1_c.append(0)
                    bike2_c.append(0)
                    bike3_c.append(0)

                index += 1
                if index % 1000 == 0:
                    print(index, 4 * a)


                    # biketype,前提条件数据都存在
            for i in range(a):
                usrdata = data[(data["userid"] == userid[i])]

                biketype = list(usrdata["biketype"])
                count = Counter(biketype)
                #print(count)
                total = sum(count.values())
                p = count[1]/total
                #print(p)
                biketype1_p.append(p)


                index += 1
                if index % 1000 == 0:
                    print(index, 4 * a)

            # time
            for i in range(a):
                usrdata = data[(data["userid"] == userid[i])]

                time_min5 = list(usrdata["time_min5"])
                count = Counter(time_min5)
                #print(count)
                total = sum(count.values())
                temp = sorted(count.items(), key=lambda item: -item[1])
                key = [i[0] for i in temp]
                _len = len(key)

                value = [i[1] / total for i in temp]
                #print(key,value)
                if _len >= 3:
                    time1.append(key[0])
                    time2.append(key[1])
                    time3.append(key[2])
                    time1_c.append(value[0])
                    time2_c.append(value[1])
                    time3_c.append(value[2])
                elif _len == 2:
                    time1.append(key[0])
                    time2.append(key[1])
                    time3.append(key[1])

                    time1_c.append(value[0])
                    time2_c.append(value[1])
                    time3_c.append(0)
                elif _len == 1:
                    time1.append(key[0])
                    time2.append(key[0])
                    time3.append(key[0])

                    time1_c.append(value[0])
                    time2_c.append(0)
                    time3_c.append(0)
                elif _len == 0:
                    time1.append(0)
                    time2.append(0)
                    time3.append(0)

                    time1_c.append(0)
                    time2_c.append(0)
                    time3_c.append(0)

                index += 1
                if index % 1000 == 0:
                    print(index, 4*a)


            # 原来的data不需要drop，新建一个表，作为索引
            data_w = pd.DataFrame()
            data_w.insert(data_w.shape[1], "userid", pd.Series(userid))
            data_w.insert(data_w.shape[1], "start_loc_1_x", pd.Series(loc_1_x))
            data_w.insert(data_w.shape[1], "start_loc_1_y", pd.Series(loc_1_y))
            data_w.insert(data_w.shape[1], "start_loc_1_count", pd.Series(count_1))
            data_w.insert(data_w.shape[1], "start_loc_2_x", pd.Series(loc_2_x))
            data_w.insert(data_w.shape[1], "start_loc_2_y", pd.Series(loc_2_y))
            data_w.insert(data_w.shape[1], "start_loc_2_count", pd.Series(count_2))
            data_w.insert(data_w.shape[1], "start_loc_3_x", pd.Series(loc_3_x))
            data_w.insert(data_w.shape[1], "start_loc_3_y", pd.Series(loc_3_y))
            data_w.insert(data_w.shape[1], "start_loc_3_count", pd.Series(count_3))

            data_w.insert(data_w.shape[1], "bike1", pd.Series(bike1))
            data_w.insert(data_w.shape[1], "bike1_c", pd.Series(bike1_c))
            data_w.insert(data_w.shape[1], "bike2", pd.Series(bike2))
            data_w.insert(data_w.shape[1], "bike2_c", pd.Series(bike2_c))
            data_w.insert(data_w.shape[1], "bike3", pd.Series(bike3))
            data_w.insert(data_w.shape[1], "bike3_c", pd.Series(bike3_c))

            data_w.insert(data_w.shape[1], "biketype1_p", pd.Series(biketype1_p))

            data_w.insert(data_w.shape[1], "time1", pd.Series(time1))
            data_w.insert(data_w.shape[1], "time1_c", pd.Series(time1_c))
            data_w.insert(data_w.shape[1], "time2", pd.Series(time2))
            data_w.insert(data_w.shape[1], "time2_c", pd.Series(time2_c))
            data_w.insert(data_w.shape[1], "time3", pd.Series(time3))
            data_w.insert(data_w.shape[1], "time3_c", pd.Series(time3_c))

            data_w.to_csv(path_or_buf=out_name,index=False)

#usrID_table_create("test_2.csv",out_name="usr_table_test.csv")
#usrID_table_create("train_2.csv",out_name="usr_table_train.csv")



















#TODO:正则化处理


#我们只要把包含信息的东西放在特定位置上就行了，之后神经网络的神经元会帮助我们处理这些！每个神经元独立处理一个工作
#所以一开始的全连接最好不要drop，然后，数量大一点儿

