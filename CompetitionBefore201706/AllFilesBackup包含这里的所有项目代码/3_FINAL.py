import  pandas  as pd
import numpy as np
import tqdm
import Geohash as gh
from sklearn import preprocessing
import pickle

TRAIN_SET = "bike_usr_train_2.csv"
TEST_SET = "bike_usr_test_2.csv"

def get_hash(filename,col_name):
    x = []
    y = []
    data = pd.read_csv(filename)

    geohash = data[col_name]
    b  = len(geohash)

    for i in tqdm.trange(b):
        a = gh.decode_exactly(geohash[i])
        x.append(a[0])
        y.append(a[1])
    print(x,y)
    return x,y

def data_preprocess():
    train_set = pd.read_csv(TRAIN_SET)
    test_set = pd.read_csv(TEST_SET)
    print("train_set",train_set)
    print("train_set_shape",train_set.shape)
    print("test_set",test_set)
    print("test_set_shape",test_set.shape)


    # 读取train中的geohash数据，得到更加精确的数据


    train_start_x,train_start_y = get_hash("train.csv","geohashed_start_loc")
    train_end_x, train_end_y = get_hash("train.csv", "geohashed_end_loc")

    test_start_x,test_start_y = get_hash("test.csv","geohashed_start_loc")


    # train 的数据替换
    train_set = train_set.drop(labels=["start_loc_x","start_loc_y"],axis=1)
    train_set.insert(train_set.shape[1],"start_loc_x",pd.Series(train_start_x))
    train_set.insert(train_set.shape[1], "start_loc_y", pd.Series(train_start_y))

    train_set = train_set.drop(labels=["end_loc_x", "end_loc_y"], axis=1)


    # test的数据替换
    test_set = test_set.drop(labels=["start_loc_x", "start_loc_y"], axis=1)
    test_set.insert(test_set.shape[1], "start_loc_x", pd.Series(test_start_x))
    test_set.insert(test_set.shape[1], "start_loc_y", pd.Series(test_start_y))


    # 其他不变，相应的，end x 和 end y也要变

    # x = []
    # y = []
    # geohash = data["geohashed_start_loc"]
    # b = len(geohash)
    # for i in range(len(geohash)):
    #
    #     print(i, b)
    #     a = gh.decode(geohash[i])
    #     x.append(a[0])
    #     y.append(a[1])


    train_set = train_set.drop(labels=["orderid","userid",
                                       "bikeid"],axis=1)

    test_set = test_set.drop(labels=["orderid",
                                     "userid","bikeid"],axis=1)



    y_train = np.array([train_end_x,train_end_y])






    print("train_set", train_set)
    print("train_set_shape", train_set.shape)
    print("test_set", test_set)
    print("test_set_shape", test_set.shape)
    print("y_train",y_train)
    print("y_train_shape",y_train.shape)

    normalized_train = []
    normalized_test = []


    for i in range(64):
        normalized_train.append(preprocessing.scale(
            np.array(train_set.icol(i))))

        normalized_test.append(preprocessing.scale(
            np.array(test_set.icol(i))))

    normalized_train = np.transpose(normalized_train)
    normalized_test = np.transpose(normalized_test)
    normalized_y_train = []
    #  作为标签使用的y不需要进行normalization

    data = []

    for i in y_train:
        scaler = preprocessing.StandardScaler().fit(i)
        data.append(scaler.mean_)
        data.append(scaler.std_)

        normalized_y_train.append(scaler.transform(i))
    data = np.array(data)
    print(data)
    np.save("mean_and_std_.npy",data)

    normalized_y_train = np.transpose(normalized_y_train)


    print("train_shape",normalized_train.shape)
    print("test_shape",normalized_test.shape)
    print("y_train_shape",normalized_y_train.shape)

    # 此前的test都是等待验证的数据集，从这里开始，test将被存储代用，
    # 之后将从train中分割一部分出来作为有Y的test用于验证

    save_test_set(normalized_test)

    save_train_set(normalized_train,normalized_y_train)




def save_train_set(x_train,y_train):
    np.savez("train_data_set.npz",x_train,y_train)

def save_test_set(x_test):
    np.save("test_set.npy",x_test)




def load_train_set():
    temp = np.load("train_data_set.npz")
    return temp["arr_0"],temp["arr_1"]

def load_test_set():
    return np.load("test_set.npy")


#data_preprocess()


from keras.models import Sequential
from keras.layers import Dense,Input,Dropout,Activation
from keras.models import  load_model


def split_data_set(ratio,mini_v=False):
    x,y = load_train_set()


    train_num = int(x.shape[0] * (1-ratio))

    x_train = x[:train_num,:]
    x_test = x[train_num:,:]

    y_train = y[:train_num,:]
    y_test = y[train_num:,:]

    if mini_v:
        x_train = x[:10000,:]
        x_test = x[10000:12500,:]

        y_train = y[:10000, :]
        y_test = y[10000:12500, :]


    print("x_train_shape",x_train.shape)
    print("y_train_shape",y_train.shape)
    print("y_test_shape",y_test.shape)
    print("x_test_shape",x_test.shape)

    return x_train,x_test,y_train,y_test

#x_train,x_test,y_train,y_test = split_data_set(0.3)


###____________________以上由于文件已经删除，所以不能使用，只能够直接用已经保存的npy和npz文件进行训练

def train_model(x_train,y_train,x_test,y_test,train=True):


    model = Sequential()



    model.add(Dense(128,activation="relu",bias_initializer="uniform",
                    kernel_initializer="uniform",input_shape=(64,)))

    model.add(Dense(256))
    model.add(Dropout(0.5))
    model.add(Activation("relu"))

    model.add(Dense(128))
    model.add(Dropout(0.25))
    model.add(Activation("relu"))

    model.add(Dense(64))
    model.add(Dropout(0.25))
    model.add(Activation("relu"))

    model.add(Dense(2))
    model.add(Dropout(0.25))
    model.add(Activation("tanh"))

    model.compile(loss='mean_squared_error', optimizer="rmsprop", metrics=["accuracy"])

    try:
        model.load_weights('my_model_weights.h5', by_name=True)
        print("load weights successful")
    except Exception as a:
        # 如果载入错误，则一定train为true
        print('\033[1;31;40m str(Exception):\t', str(a),"'\033[0m'")
        train = True

    # 直接训练
    if train :
        model.fit(x_train,y_train,batch_size=100,epochs=5,
                  validation_data=(x_test,y_test),
                  shuffle=True)


        score, acc = model.evaluate(x_test, y_test,
                                    )

    # 预测结果，将结果反正则化，然后与标准答案进行对比，打印误差

    mean1, std1, mean2, std2 = list(np.load("mean_and_std_.npy"))
    print(mean1,std1,mean2,std2)



    # 基于x_test进行结果验证
    prediction = model.predict(x_test)



    prediction = np.transpose(prediction)
    prediction[0,:] = (prediction[0,:] * std1) + mean1
    prediction[1, :] = (prediction[1, :] * std2) + mean2
    prediction = np.transpose(prediction)

    print(prediction.shape)

    # 将其存储后，用另一个脚本进行处理

    if train:
        # 真实数据还原
        y_ = np.transpose(y_test)
        y_[0, :] = (y_[0, :] * std1) + mean1
        y_[1, :] = (y_[1, :] * std2) + mean2
        y_ = np.transpose(y_)

        model.save_weights('my_model_weights.h5')
        print('Test score:', score)
        print('Test accuracy:', acc)


        print("show some examples")

        for i in range(100):
            print("-------------")
            print("expected: ", y_[i])
            print("result: ", prediction[i])
            print("error: ", y_[i] - prediction[i])


    answer = model.predict(load_test_set())

    answer = np.transpose(answer)
    answer[0, :] = (answer[0, :] * std1) + mean1
    answer[1, :] = (answer[1, :] * std2) + mean2
    answer = np.transpose(answer)

    print("answer_shape",answer.shape)



    np.save("prediction.npy", answer)

# 模型训练
#train_model(x_train,y_train,x_test,y_test,train=False)

# 模型预测，并将结果写入文件中
#y_test = load_test_set()
#train_model(None, None, y_test,None,train=False)



# 训练：尽可能减少和目标location的方差，采用二维距离公式，
# 预测：得到预测的二维数据之后，通过之前训练集的geohash编码出的集合，对于每个集合求取最小距离的3个
# 然后geohash编码得到标签









