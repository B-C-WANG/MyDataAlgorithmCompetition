import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.layers import ConvLSTM2D, Input,Dropout,Activation,BatchNormalization,Conv2D,Dense,Conv3D
from keras.models import  Sequential
import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.externals import  joblib
import keras


def build_model2():
    '''
    保证输入为None 60 132 输出为None 30 162
    :return:
    '''



    model = Sequential()

    model.add(Conv2D(
        filters=10,
        kernel_size=(60,132),
        input_shape=(60, 132, 1), padding='same', activation="tanh"
    ))

    model.add(BatchNormalization())
    model.add(Dropout(0.75))


    model.add(Conv2D(filters=15,
                     kernel_size=(12,12),
                     strides=(2, 1)
                     , padding="same"
                     , activation="tanh"
                     ))
    model.add(BatchNormalization())
    model.add(Dropout(0.75))

    model.add(Conv2D(filters=10,
                     kernel_size=(12,12),
                     strides=(1, 1)
                     , padding="same"
                     , activation="tanh"
                     ))
    model.add(BatchNormalization())
    model.add(Dropout(0.75))

    model.add(Conv2D(filters=10,
                     kernel_size=(12, 12),
                     strides=(1, 1)
                     , padding="same"
                     , activation="tanh"
                     ))
    model.add(BatchNormalization())
    model.add(Dropout(0.75))

    model.add(Conv2D(filters=1,
                     kernel_size=(12,12),
                     padding="same",
                     activation="tanh",
                     data_format="channels_last"

                     ))

    return model




def train2(train=True,load_weights=True,save_weights=True,predict=False,valid=False,epoch=500):


    x_train = np.array(pd.read_csv("x_train.csv").drop(labels="frame",axis=1))
    y_train = np.array(pd.read_csv("y_train.csv").drop(labels="frame",axis=1))
    #test = np.array(pd.read_csv("test.csv").drop(labels="frame",axis=1))

    # 之后：将x_train和y_train合并进行正则化，test相同的方式正则化（或者所有的y都按照x的方式正则化，先试试哪个更好）

    temp = np.concatenate((x_train, y_train), axis=0)  # concat很容易，想加哪个就针对shape中的那个位置加上
    print(temp.shape)
      # 正则化fit需要dim<=2
    scaler = StandardScaler().fit(temp)
    joblib.dump(scaler, "scaler.npy")

    x_train = scaler.transform(x_train)
    y_train = scaler.transform(y_train)




    x_train = x_train.reshape(92, 60,132,1)
    y_train = y_train.reshape(92,30,132,1)



    #test = test.reshape(30,60,132,1,1)


    # 划分训练集和测试集便于了解过拟合情况
    SPLIT_NUM = 62

    x_train_ = x_train[0:SPLIT_NUM,:,:,:]
    y_train_ = y_train[0:SPLIT_NUM, :, :, :]

    x_test_ = x_train[SPLIT_NUM:,:,:,:]
    y_test_ = y_train[SPLIT_NUM:, :, :, :]


    print(x_train_.shape)
    print(x_test_.shape)




    model = build_model2()

    model.summary()

    #model.compile(loss="mean_absolute_percentage_error", optimizer="adadelta")

    model.compile(loss="mean_absolute_error", optimizer="adam")



    if load_weights:
        try:
            model.load_weights('model_weights_CNN.h5', by_name=True)
            print("load weights successful")
        except Exception as a:
            print('\033[1;31;40m str(Exception):\t', str(a), "'\033[0m'")
            train = True

    if train:
        # 增加tensorboard可视化
        tb_back = keras.callbacks.TensorBoard(log_dir='E:\~~大数据竞赛专用文件夹\智慧城市道路预测/Graph', histogram_freq=0,
                                    write_graph=True, write_images=True)



        # batch_size小一点儿，不然容易OOM
        model.fit(x_train_, y_train_, batch_size=1000, epochs=epoch,validation_data=(x_test_,y_test_)
                  ,callbacks=[tb_back])



        # 之后命令行使用tensorboard --logdir E:\~~大数据竞赛专用文件夹\智慧城市道路预测/Graph 命令查看结果

        if save_weights:
            model.save_weights('model_weights_CNN.h5')
            print("save weights_success!")

    if predict:
        test = np.array(pd.read_csv("test.csv").drop(labels="frame",axis=1))
        test = scaler.transform(test)
        test = test.reshape(30,60,132,1)
        result = model.predict(test)
        result = result.reshape(30*30,132)
        result = scaler.inverse_transform(result)
        result = result.reshape(30,30,132)

        np.save("result.npy",result)

    if valid:
        # 得出训练的总loss

        y = model.predict(x_test_)
        y = y.reshape((92-SPLIT_NUM) * 30, 132)
        y = scaler.inverse_transform(y)


        y_ = np.array(pd.read_csv("y_train.csv").drop(labels="frame", axis=1))
        y_ = y_.reshape(92, 30, 132, 1)[SPLIT_NUM:,:,:,:]
        y_ = y_.reshape((92-SPLIT_NUM) * 30, 132)

        print("y_",y_[0,:])
        print("y",y[0,:])
        loss = np.mean(np.abs(y-y_)/y_)

        print("x_test的loss",loss)


        y = model.predict(x_train)
        y = y.reshape(92  * 30, 132)
        y = scaler.inverse_transform(y)

        y_ = np.array(pd.read_csv("y_train.csv").drop(labels="frame", axis=1))

        y_ = y_.reshape(92 * 30, 132)

        print("y_", y_[0, :])
        print("y", y[0, :])
        loss = np.mean(np.abs(y - y_) / y_)

        print("所有train的loss", loss)

        return loss


def predict_and_submit():
    prediction = np.load("result.npy")
    print(prediction[0,0,:])
    print(prediction.shape)

    # 首先建立第二列,date和第三列time
    date = []
    time = []
    for i in range(30):
        for j in range(30):
            for z in range(132):
                if i <= 8 :
                    date.append("2016-06-0{}".format(i+1))
                    if j<=3:
                        time.append("[2016-06-0{} 08:0{}:00,2016-06-0{} 08:0{}:00)".format(
                            i+1,j*2,i+1,j*2+2
                        ))
                    elif j ==4:
                        time.append("[2016-06-0{} 08:08:00,2016-06-0{} 08:10:00)".format(
                            i+1,  i+1
                        ))

                    elif j == 29:
                        time.append("[2016-06-0{} 08:58:00,2016-06-0{} 09:00:00)".format(
                            i + 1, i + 1
                        ))


                    else:
                        time.append("[2016-06-0{} 08:{}:00,2016-06-0{} 08:{}:00)".format(
                            i+1, j * 2, i+1, j * 2+2
                        ))


                        # 下面的代码和上面的基本一致，只是日期{}前面不加0
                else:
                    date.append("2016-06-{}".format(i+1))
                    if j <= 3:
                        time.append("[2016-06-{} 08:0{}:00,2016-06-{} 08:0{}:00)".format(
                            i+1, j * 2, i+1, j * 2 + 2
                        ))
                    elif j == 4:
                        time.append("[2016-06-{} 08:08:00,2016-06-{} 08:10:00)".format(
                            i+1,  i+1
                        ))

                    elif j == 29:
                        time.append("[2016-06-{} 08:58:00,2016-06-{} 09:00:00)".format(
                            i + 1, i + 1
                        ))


                    else:
                        time.append("[2016-06-{} 08:{}:00,2016-06-{} 08:{}:00)".format(
                            i+1, j * 2, i+1, j * 2 + 2
                        ))

    # 建立第一列 link

    link_name = pd.read_csv("test.csv")
    link_name = link_name.keys()
    link_name = list(link_name)[1:]
    print(len(link_name),"\n",link_name)

    link =[]
    for i in range(900):
        for z in range(len(link_name)):
            link.append(link_name[z])

    data =[]

    #  从上至下，依次是不同link，同一time date 之后是不同time 最后是不同date
    # reshape的第一维度是date，第二维度是time，第三维度是link
    for i in range(30):
        for j in range(30):
            for k in range(132):
                # 每次添加第1天第1time 所有link数据，
                # 然后是第1天，第2time 所有link数据，依次下去

                # 预测的负值处理
                #if prediction[i,j,k] < 0.0:
                #    data.append(1.0)
                data.append(prediction[i,j,k])

    data_w = pd.DataFrame()
    data_w.insert(0,"link",pd.Series(link))
    data_w.insert(data_w.shape[1], "date", pd.Series(date))
    data_w.insert(data_w.shape[1], "time", pd.Series(time))
    data_w.insert(data_w.shape[1], "data", pd.Series(data))

    data_w.to_csv("Nautiloidea_2017-07-28.csv",index=False,sep="#")



# 训练
#train2()
# 验证
def run():
    a= []
    for i in range(10):
        a.append(train2(train=True,valid=True,epoch=10))
    print(a)

run()

# 预测
#train2(train=False,predict=True)

# 训练并验证
#train2(train=True,valid=True)

# 最后删一下表头，重命名csv为txt即可
#predict_and_submit()

