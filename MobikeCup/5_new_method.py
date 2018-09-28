
import  numpy as np
import pandas as pd
from collections import Counter
import tqdm
import profile
import geohash as gh
import json




# 采用直接统计每个用户前3个常去的地点，然后写入，如果这个可行，那么朴素贝叶斯也是可行的


# github
def get_most_frequency_three():
    data = pd.read_csv("train.csv")
    userid = list(set(data["userid"]))
    userid.sort()
    a = len(list(userid))
    loc1 = []
    loc2 = []
    loc3 = []



    #a = 100


    for i in tqdm.trange(a):

            usrdata = data[(data["userid"] == userid[i])]

            loc = list(usrdata["geohashed_end_loc"])

            count = Counter(loc)



            temp = sorted(count.items(), key=lambda item: -item[1])

            key = [i[0] for i in temp]

            count = len(key)

            if count >= 3:
                loc1.append(key[0])
                loc2.append(key[1])
                loc3.append(key[2])

            elif count == 2:
                loc1.append(key[0])
                loc2.append(key[1])
                loc3.append(gh.encode(float(gh.decode(key[1])[0]) + 0.1, float(gh.decode(key[1])[1]) + 0.1,
                                      precision=7))

            elif count == 1:

                loc1.append(key[0])
                loc2.append(gh.encode(float(gh.decode(key[0])[0]) + 0.1, float(gh.decode(key[0])[1]) + 0.1,
                                      precision=7))
                loc3.append(gh.encode(float(gh.decode(key[0])[0]) + 0.2, float(gh.decode(key[0])[1]) + 0.2,
                                      precision=7))
            else:

                loc1.append(gh.encode(
                    float(gh.decode("wx4gfbe")[0]) + 0.1, float(gh.decode("wx4gfbe"))[1] + 0.1
                ,precision=7))
                loc2.append(gh.encode(float(gh.decode("wx4gfbe")[0]) + 0.2, float(gh.decode("wx4gfbe")[1]) + 0.2,
                                      precision=7))
                loc3.append(gh.encode(float(gh.decode("wx4gfbe")[0]) + 0.3, float(gh.decode("wx4gfbe")[1]) + 0.3,
                                      precision=7))

    data_w = pd.DataFrame()
    data_w.insert(data_w.shape[1], "userid", pd.Series(userid))
    data_w.insert(data_w.shape[1], "end_loc_1", pd.Series(loc1))
    data_w.insert(data_w.shape[1], "end_loc_2", pd.Series(loc2))
    data_w.insert(data_w.shape[1], "end_loc_3", pd.Series(loc3))

    data_w.to_csv("usr_end_loc_table.csv", index=False)

#get_most_frequency_three()

# github
def get_results():

    data = pd.read_csv("test.csv")
    order_id = list(data["orderid"])
    user_id = list(data["userid"])


    user_table = pd.read_csv('usr_end_loc_table.csv')

    label1 = []
    label2 = []
    label3 = []

    a = len(order_id)

    #a = 100
    for i in tqdm.trange(a):

        temp = user_table[(user_table["userid"]==user_id[i])]

        try:
            label1.append(temp["end_loc_1"].values[0])
            label2.append(temp["end_loc_2"].values[0])
            label3.append(temp["end_loc_3"].values[0])
        except:
            label1.append("wx4gfbe")
            label2.append("wx4gfab")
            label3.append("wx4gcab")

    data_w = pd.DataFrame()
    data_w.insert(0,"a",order_id)
    data_w.insert(data_w.shape[1], "b", pd.Series(label1))
    data_w.insert(data_w.shape[1], "c", pd.Series(label2))
    data_w.insert(data_w.shape[1], "d", pd.Series(label3))

    data_w.to_csv("final_result.csv",index=False)

#get_results()


from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing
from sklearn.externals import joblib
# github
def NB_predict():
    data = pd.read_csv("train.csv")
    # 时间暂时没用，可以直接丢掉，如果为了更高的精度，可以留下
    x_train = data.drop(labels=["geohashed_start_loc",
                                "geohashed_end_loc","starttime","orderid","bikeid"],axis=1)
    test = pd.read_csv("test.csv")
    test = test.drop(["starttime","orderid","bikeid"],axis=1)


    # 将字符转化为float

    # 先收集一下所有涉及的字符
    # labelEncoder可以直接建立字典，用fit喂入字典，然后用inverse导入原来的数据
    geo_data1 = list(data["geohashed_end_loc"])
    geo_data1.extend(list(test["geohashed_start_loc"]))
    geo_data1.extend(list(data["geohashed_start_loc"]))

    print(geo_data1.__len__())
    geo_data = list(set(geo_data1))# 编码只支持list
    print(len(geo_data))

    label_encoder = preprocessing.LabelEncoder()
    # 建立和float的字典
    label_encoder.fit(geo_data)


    test_2 = test.drop("geohashed_start_loc",axis=1)
    print("shape:",test_2.shape)

    # 将drop掉的结果经过label_encoder编码过后重新加进去

    x_train.insert(x_train.shape[1],"geohashed_start_loc",pd.Series(
        label_encoder.transform(
            data["geohashed_start_loc"]
        )
    ))

    test_2.insert(test_2.shape[1], "geohashed_start_loc", pd.Series(
        label_encoder.transform(
            test["geohashed_start_loc"]
        )
    ))

    print(x_train.head())
    print(test_2.head())


    y_train = label_encoder.transform(
        list(data["geohashed_end_loc"]))



    try:
        clf = joblib.load("train_model.m")
    except:
        clf = GaussianNB()#  在NB的源代码中，我加上了tqdm模块用于展示训练时间
        clf.fit(X=x_train,y=y_train)
    # 存储训练的模型
    joblib.dump(clf,"train_model.m")

    del x_train
    del y_train

    # error: 有内存不足的问题，可以尝试分批次predict

    prediction = []

    prediction = clf.predict(test_2)
    print(len(prediction))
    print(prediction[:100])



    prediction = preprocessing.LabelEncoder().inverse_transform(prediction)


    print(prediction)

    data = pd.DataFrame()
    data.insert(0,"predict1",pd.Series(prediction))
    data.to_csv("result2.csv",index=False)


#NB_predict()

# data = pd.read_csv("train.csv")
# data = data.drop(labels=["starttime","bikeid","orderid"],axis=1)
# data = pd.get_dummies(data)
# print(data)

# 由于NB预测时内存溢出，利用pd.dummies也遇到了内存不足的问题
# 因而采用新的方法，首先将train数据中所有除了userid和start loc的其他内容删掉
# test中也只要start location
# 之后对于每一个userid，筛选出start loc 和 end loc 的映射，对于每一个start loc 和end loc
# 进行geohash_exactly之后计算出最近距离的点，如果是start loc就得到其endloc ，如果是endloc就得到其start loc
# github
def prepare():
    data = pd.read_csv("train.csv")
    data = data.drop(labels=[
        "orderid","bikeid","biketype","starttime"
    ],axis=1)
    data2 = pd.read_csv("test.csv")
    data2 = data2.drop(labels=[
        "orderid","bikeid","biketype","starttime"
    ],axis=1)

    data.to_csv("new_train.csv",index=False)
    data2.to_csv("new_test.csv",index=False)
# github
def geohash_decode(hashed_code):
    temp = gh.decode_exactly(hashed_code)
    return [temp[0],temp[1]]
# github
def get_closest_distance(measure_loc, start_loc_l, end_loc_l,min_number):

    aim_location = geohash_decode(measure_loc)
    number = len(start_loc_l)
    result = []
    for i in tqdm.trange(number):
        location = geohash_decode(start_loc_l[i])
        result.append((aim_location[0]-location[0])*(aim_location[0]-location[0])+
                      (aim_location[1] - location[1]) * (aim_location[1] - location[1])
                      )
    for i in range(number):
        location = geohash_decode(end_loc_l[i])
        result.append((aim_location[0] - location[0]) * (aim_location[0] - location[0]) +
                      (aim_location[1] - location[1]) * (aim_location[1] - location[1])
                      )
    result_ = np.array(result).argsort()[:min_number][::1]# 一共三个数值，对应最小的三个值
    result = []
    for i in result_:
        if i < number:
            # 距离start区域最近的，得到相应的end
            result.append(end_loc_l[i])
        else:
            result.append(start_loc_l[i-number])
    return result




# github
def start_do_():
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    orderid = test["orderid"]

    test_userid = list(test["userid"])

    test_start_loc = list(test["geohashed_start_loc"])

    result1 = []
    result2 = []
    result3 = []
    a = len(test_userid)
    #a= 100

    for i in tqdm.trange(a):
        try:
            train_loc_data = train[(train["userid"]==test_userid[i])]
            measure_loc = test_start_loc[i]


            start_loc_data = list(train_loc_data["geohashed_start_loc"])
            end_loc_data = list(train_loc_data["geohashed_end_loc"])

            if len(start_loc_data) >=3:
                temp = get_closest_distance(measure_loc,start_loc_data,end_loc_data,3)
                result1.append(temp[0])
                result2.append(temp[1])
                result3.append(temp[2])
            elif len(start_loc_data) ==2:
                temp = get_closest_distance(measure_loc, start_loc_data, end_loc_data,2)
                result1.append(temp[0])
                result2.append(temp[1])
                result3.append("wxxxxx0")#用作标记，日后可以替换
            elif len(start_loc_data) ==1:
                temp = get_closest_distance(measure_loc, start_loc_data, end_loc_data,1)
                result1.append(temp[0])
                result2.append("wxxxxx1")
                result3.append("wxxxxx2")
            else:
                result1.append("wxxxxx0")
                result2.append("wxxxxx1")
                result3.append("wxxxxx2")

        except Exception as e:
             print(e)
             result1.append("wxxxxx3")
             result2.append("wxxxxx1")
             result2.append("wxxxxx0")
    data_w = pd.DataFrame()
    data_w.insert(0,"result3",pd.Series(result3))
    data_w.insert(0, "result2", pd.Series(result2))
    data_w.insert(0, "result1", pd.Series(result1))
    data_w.insert(0,"userid",orderid)

    data_w.to_csv("submit_result_1.csv",index=False)


#start_do_()

# 得到了初步的结果之后，再考虑去掉重复和缺失值，同时也可以采用上次正确率0.14的数据代替一部分

#由于存在很多的缺失用户信息，所以对于那些id没有在train中出现的用户，需要根据他们的start loc 信息得到end loc信息

# 首先筛选出result1 = "wxxxxx0" 的用户id，按照id找到对应的startloc，然后应用location字典
# 字典的创建：读取train数据，对于每一个startlocation，建立一个list，存储endlocation，然后同样的方法创建enddaostart的字典，只需要把数据加上原来的字典就行，不需要重新建立第二个字典

# 之后根据start和endloc 字典得到数据，筛出前3个，如果不足三个就标记为wsssss1，要和之前因为其他原因缺失的值分开
# 应用字典：筛选出result1 = "wxxxxx0" 的用户id，根据id进行替换

# 更新：其实一个start或者endloc能够对应的信息很多，可能的话需要拆分字典，比如用startloc 加上biketype作为key
# 或者甚至是直接加上时间
# 字典作为一种函数方法还是很有用的！



import pickle
# 建立字典，速度相当快！
# 更新：建立startloc+biketype+startime离散值的字典，同时观察平均一个startloc对应多少个endloc，控制在3左右
def make_location_dict_method_1(filename):
    dict = {}

    data = pd.read_csv("train.csv")
    start_loc = list(data["geohashed_start_loc"])
    end_loc = list(data["geohashed_end_loc"])

    #time =

    a = len(start_loc)
    #a = 100000
    for i in tqdm.trange(a):
        try:
            dict[start_loc[i]].append(end_loc[i])
        except:
            dict[start_loc[i]] = []

    for i in tqdm.trange(a):
        try:
            dict[end_loc[i]].append(start_loc[i])
        except:
            dict[end_loc[i]] = []




    with open(filename, 'wb') as f:
        pickle.dump(dict, f)

def make_location_dict_method_2(filename):
    dict = {}

    data = pd.read_csv("train.csv")
    start_loc = list(data["geohashed_start_loc"])
    time  = list(data["starttime"])
    # 新建一个24离散值的时间
    time_l = []
    bike_type = list(data["biketype"])
    for i in time:
        time_l.append(int(int(i[-8:-6]))/2)
    # 把location的最后一位删掉，这样location可以模糊一点儿
    new_start_loc = []
    for i in start_loc:
        new_start_loc.append(i[:-1])



    end_loc = list(data["geohashed_end_loc"])

    new_end_loc = []
    for i in end_loc:
        new_end_loc.append(i[:-1])



    a = len(start_loc)
    #a = 100000
    for i in range(a):
        try:
            #print(dict)
            dict[new_start_loc[i]+str(time_l[i])+str(bike_type[i])].append(end_loc[i])
        except:
            dict[new_start_loc[i]+str(time_l[i])+str(bike_type[i])] = []
            dict[new_start_loc[i] + str(time_l[i]) + str(bike_type[i])].append(end_loc[i])

    # for i in tqdm.trange(a):
    #      temp = int(time_l[i]) + 4
    #      try:
    #
    #          if ( temp >= 12):
    #              temp = temp -12
    #
    #
    #          # 对于这个endloc，时间加上一个4（由于除以了2，相当于原来加上8，指的是8小时之后逆转）
    #          dict[new_end_loc[i] + str(temp) + str(bike_type[i])].append(start_loc[i])
    #      except:
    #          dict[new_end_loc[i]+str(temp)+str(bike_type[i])]=[]
    #



    with open(filename, 'wb') as f:
        pickle.dump(dict, f)





def dict_summary(filename):

    with open(filename, 'rb') as f:
        dict = pickle.load(f)
        #print(dict)
    number = 0
    no_element_count = 0
    for i in dict.values():
        temp = len(i)
        if temp == 0:
            no_element_count += 1
        else:
            number += len(i)

    print("平均一个字典有{}个值".format(number / len(dict)))


#TODO:采用不同方法建立字典，直到字典平均value数目接近3， 然后空字典数目很少


#make_location_dict_method_1("dict.pickle")

make_location_dict_method_2("dict.pickle")

dict_summary("dict.pickle")

#  运行也很快，十几秒完成
def apply_dict():
    with open("dict.pickle", 'rb') as f:
        dict = pickle.load(f)
    submit_result = pd.read_csv("submit_result_1.csv")
    test_data = pd.read_csv("test.csv")
    #  在提交文件中添加orderid列
    orderid = test_data["orderid"]
    submit_result.insert(0,"orderid",orderid)

    # 得到缺失值的orderid
    submit_absent_id = list(submit_result[submit_result["result1"]=="wxxxxx0"]["orderid"])

    # start_loc_of_absent_id = []
    # time_of_absent_id = []
    # bike_type_of_absent_id = []
    # a = len(submit_absent_id)
    # #a =100
    # for i in tqdm.trange(a):
    #
    #     temp = submit_absent_id[i]
    start_loc_of_absent_id=list(test_data[test_data["orderid"].isin(submit_absent_id)]["geohashed_start_loc"])
    time_of_absent_id=list(test_data[test_data["orderid"].isin(submit_absent_id)]["starttime"])
    bike_type_of_absent_id=list(test_data[test_data["orderid"].isin(submit_absent_id)]["biketype"])

    print(submit_absent_id)
    submit_absent_id = list(submit_absent_id)
    print(len(submit_absent_id))
    print(start_loc_of_absent_id)
    print(len(start_loc_of_absent_id))

    time_l = []

    for i in time_of_absent_id:
        # test中的time不一样
        time_l.append(int(int(i[-10:-8])) / 2)
    # 把location的最后一位删掉，这样location可以模糊一点儿
    new_start_loc = []
    for i in start_loc_of_absent_id:
        new_start_loc.append(i[:-1])




    loc1 = []
    loc2 = []
    loc3 = []
    a = len(submit_absent_id)
    #a = 100
    for i in tqdm.trange(a):
        try:
            temp = dict[new_start_loc[i]+str(time_l[i])+str(bike_type_of_absent_id[i])]
        except:
            temp = ["wsssss0","wsssss1","wsssss2"]

        try:
            loc1.append(temp[0])
        except:
            loc1.append("wssss0")
        try:
            loc2.append(temp[1])
        except:
            loc2.append("wssss1")
        try:
            loc3.append(temp[2])
        except:
            loc3.append("wssss1")

    data_w = pd.DataFrame()
    data_w.insert(0,"orderid",pd.Series(submit_absent_id))
    data_w.insert(data_w.shape[1],"loc1",pd.Series(loc1))
    data_w.insert(data_w.shape[1], "loc2", pd.Series(loc2))
    data_w.insert(data_w.shape[1], "loc3", pd.Series(loc3))

    data_w.to_csv("absent_overwrite.csv",index=False)
apply_dict()



# 将替换后的table应用到提交文件上
def replace_submission_with_overwrite_table():
    submit_data = pd.read_csv("submit_result_1.csv")

    test_data = pd.read_csv("test.csv")
    #  在提交文件中添加orderid列
    orderid = test_data["orderid"]
    submit_data.insert(0, "orderid", orderid)

    former_loc1 = submit_data["result1"]
    former_loc2 = submit_data["result2"]
    former_loc3 = submit_data["result3"]


    overwrite_data = pd.read_csv("absent_overwrite.csv")
    absent_id = list(overwrite_data["orderid"])

    loc1 = list(overwrite_data["loc1"])
    loc2 = list(overwrite_data["loc2"])
    loc3 = list(overwrite_data["loc3"])

    a = len(former_loc1)
    #a = 100000
    index = 0
    # 建立一个新的loc，如果检测到为。。。就加上loc1的数据，然后index+=1
    new_loc1 = []
    new_loc2 = []
    new_loc3 = []

    for i in tqdm.trange(a):
        if former_loc1[i] == "wxxxxx0":
            new_loc1.append(loc1[index])
            new_loc2.append(loc2[index])
            new_loc3.append(loc3[index])
            index+=1
        else:
            new_loc1.append(former_loc1[i])
            new_loc2.append(former_loc2[i])
            new_loc3.append(former_loc3[i])





    data_w = pd.DataFrame()
    data_w.insert(0,"orderid",orderid)
    data_w.insert(data_w.shape[1], "result1", pd.Series(new_loc1))
    data_w.insert(data_w.shape[1], "result2", pd.Series(new_loc2))
    data_w.insert(data_w.shape[1], "result3", pd.Series(new_loc3))
    data_w.to_csv("overwrite_result.csv",index=False)





replace_submission_with_overwrite_table()




# 新方法：字典索引法：用用户id，startloc，starttime的区间划分作为key，endlocation作为value进行
# 但是对于用户id不在里面的，需要采用另一种方法


#  后期：采用神经网络的方法，将标签转化一下


# 神经网络确认无效，最后一把：
# 先备份现在的数据，然后把所有第三列的数据替换成之前第一次用最高频率预测出的最大值对应的第一列，第一列有缺失值的，用原来的值替换

def replce_with_original():
    original = pd.read_csv("1.csv")# 之前0.15那次以概率前3的为label的结果

    overwrite = pd.read_csv("absent_overwrite.csv")
    overwrite_loc3 = overwrite["loc3"]# 之后替换第三列中absent的值，在之前，用wx4gfbe代替了
    # 现在按照顺序用非缺失的值去代替

    loc3 = list(original["result1"])# 获取概率最大的

    new_loc3 = []
    index = 0
    for i in tqdm.trange(len(loc3)):
        if i == "wx4gfbe":
            new_loc3.append(overwrite_loc3[index])
            index+=1
        else:
            new_loc3.append(loc3[i])

    data_w = pd.read_csv("2.csv")
    data_w = data_w.drop(labels="result3",axis=1)

    data_w.insert(data_w.shape[1],"result3",pd.Series(new_loc3))

    data_w.to_csv("the_4_result.csv",index=False)

#replce_with_original()





