# 采用神经网络，将表示location的label编码，当做数值进行
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import pandas as pd
import tqdm
from sklearn.externals import joblib
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Input


def time_transform_1(list):

    time_l = list
    new_time_l = []
    num = len(time_l)
    for i in tqdm.trange(num):
        data, time = time_l[i].split(" ")
        y, mouth, d = data.split("-")
        h, m, s = time.split(":")
        time_s = int(h + m + str(int(float(s))))
        new_time_l.append(time_s)

    return new_time_l

def date_transform_(list):
    time_l = list
    new_time_l = []
    num = len(time_l)
    for i in tqdm.trange(num):
        data, time = time_l[i].split(" ")
        y, mouth, d = data.split("-")

        time_s = int(d)
        new_time_l.append(time_s)

    return new_time_l

def make_loc_label_transformer():
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")

    train_loc =list(train["geohashed_start_loc"])
    test_loc = list(test["geohashed_start_loc"])
    train_loc_2 = list(train["geohashed_end_loc"])

    train_loc.extend(test_loc)
    train_loc.extend(train_loc_2)
    print(len(test_loc),len(train_loc))

    train_loc = list(set(train_loc))

    print(len(train_loc))
    clf = LabelEncoder()
    clf.fit(train_loc)
    joblib.dump(clf,"label_encoder_of_loc.m")

#make_loc_label_transformer()




def prepare_for_nn():
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")



    # 删掉orderid
    train = train.drop(labels="orderid",axis=1)
    test = test.drop(labels="orderid", axis=1)
    # 时间转换
    time_train = list(train["starttime"])
    time_test  = list(test["starttime"])

                # 标了_n的代表是训练测试所用的数据集
    new_time_train_2 = time_transform_1(time_train)
    new_time_test_2 = time_transform_1(time_test)
    new_date_train_3 = date_transform_(time_train)
    new_date_test_3 = date_transform_(time_test)

    biketype_train_4 = list(train["biketype"])
    biketype_test_4 = list(test["biketype"])

    start_loc_train_5 = list(train['geohashed_start_loc'])
    end_loc_train_y = list(train['geohashed_end_loc'])
    start_loc_test_5 = list(test['geohashed_start_loc'])

    label_encoder = joblib.load("label_encoder_of_loc.m")

    start_loc_train_5 = label_encoder.transform(start_loc_train_5)
    end_loc_train_y = label_encoder.transform(end_loc_train_y)
    start_loc_test_5 = label_encoder.transform(start_loc_test_5)

    #print(start_loc_train_5)

    x_train1 = preprocessing.scale(new_date_train_3)
    x_test1 = preprocessing.scale(new_date_test_3)
    x_train2 = preprocessing.scale(new_time_train_2)
    x_test2 = preprocessing.scale(new_time_test_2)
    x_train3 = preprocessing.scale(biketype_train_4)
    x_test3 = preprocessing.scale(biketype_test_4)
    x_train4 = preprocessing.scale(start_loc_train_5)
    x_test4 = preprocessing.scale(start_loc_test_5)



    y_train_scalar = StandardScaler().fit(end_loc_train_y)
    y_train = y_train_scalar.transform(end_loc_train_y)

    # 存储正则模型，之后可以通过inverse transform来返回原来的数据
    joblib.dump(y_train_scalar,"y_train_standardScaler.m")

    x_train = np.array([[x_train1],[x_train2],[x_train3],[x_train4]])
    x_test = np.array([[x_test1],[x_test2],[x_test3],[x_test4]])
    y_train = np.array(y_train)

    x_train = np.transpose(x_train)
    x_test = np.transpose(x_test)
    x_train = x_train.reshape(x_train.shape[0], 4)
    x_test = x_test.reshape(x_test.shape[0], 4)

    print(x_train.shape)
    print(x_test.shape)
    print(y_train.shape)

    np.savez("train_test_data.npz",x_train=x_train,y_train=y_train,question=x_test)


#prepare_for_nn()


def train_model():

    temp = np.load("train_test_data.npz")
    x = temp["x_train"]
    y = temp["y_train"]

    print(x.shape)
    print(y.shape)

    SPLIT_RATIO = 0.3
    train_num = int((1-SPLIT_RATIO)*x.shape[0])





    x_train = x[:train_num,:]
    y_train = y[:train_num]
    x_test = x[train_num:,:]
    y_test = y[train_num:]


    print(x_train.shape)
    print(x_test.shape)
    print(y_train.shape)
    print(y_test.shape)

    model = Sequential()

    model.add(Dense(16, activation="relu", bias_initializer="uniform",
                    kernel_initializer="uniform", input_shape=(4,)))

    model.add(Dense(32))
    model.add(Dropout(0.3))
    model.add(Activation("relu"))

    model.add(Dense(64))
    model.add(Dropout(0.3))
    model.add(Activation("relu"))

    model.add(Dense(32))
    model.add(Dropout(0.3))
    model.add(Activation("relu"))

    model.add(Dense(16))
    model.add(Dropout(0.3))
    model.add(Activation("relu"))

    model.add(Dense(1))
    model.add(Dropout(0.3))
    model.add(Activation("tanh"))

    model.compile(loss='mean_squared_error', optimizer="rmsprop", metrics=["accuracy"])

    try:
            model.load_weights('my_model_weights.h5', by_name=True)
            print("load weights successful")
    except Exception as a:

        print('\033[1;31;40m str(Exception):\t', str(a), "'\033[0m'")

    model.fit(x_train, y_train, batch_size=500, epochs=2, validation_data=(x_test, y_test))

    score, acc = model.evaluate(x_test, y_test,
                                )
    print('Test score:', score)
    print('Test accuracy:', acc)
    model.save_weights('my_model_weights.h5')



    scaler = joblib.load("y_train_standardScaler.m")


    prediction = model.predict(x_test)

    y_test = scaler.inverse_transform(y_test)
    prediction =scaler.inverse_transform(prediction)


    prediction = np.around(prediction,decimals=0)
    y_test = np.around(y_test,decimals=0)

    print(prediction)
    print(y_test)

    accuracy_ = np.sum(prediction==y_test)/len(y_test)

    print(accuracy_)





train_model()

