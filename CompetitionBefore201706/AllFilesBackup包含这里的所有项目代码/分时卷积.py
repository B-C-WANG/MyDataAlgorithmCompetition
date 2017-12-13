import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.layers import ConvLSTM2D, Input,Dropout,Activation,BatchNormalization,Conv2D,Dense,Conv3D
from keras.models import  Sequential
import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.externals import  joblib
import keras
import keras.backend as K

# Attention： 废弃内容全部在LSTM卷积方法.py中


# TODO：可以尝试将这些132x1大小的图片组成视频播放
# TODO：可尝试将132 reshape 为12x11来卷积， 因为要学习的是相关关系，相关的关系并不是一条线上可以了解的

#TODO:可以尝试卷积，先要正则化

# 更新：题目给的是3个月的时间加上6月份6点到8点的数据，预测8点到10点的数据，需要更改一下！

# 在填补了数据的filled_result.data上面更改

def transfer():
    data = pd.read_csv("filled_result.csv")


    # 将6月的数据存储
    test = data[data["frame"]>6000000]
    test.to_csv("test.csv",index=False)

    # 筛选出data中所有6点到7.59点的结果，注意type为int64，先用取余得到后4位，得到x_train

    x_train = data[(data["frame"]%10000>559) & (data["frame"]%10000<759) & (data["frame"]<6000000)]

    # 筛选出8.00到8.58的数据，作为x test

    y_train = data[(data["frame"]%10000>759)&(data["frame"]%10000<859) & (data["frame"]<6000000)]

    x_train.to_csv("x_train.csv",index=False)

    y_train.to_csv("y_train.csv",index=False)


#transfer()

# 结果 ： x_train 92 x 60 x 132 y_train 92 x 30 x 132 根据 30 x 60 x 132 求得 30 x 30 x 132



### 第二种方法：还是采用conv2d lstm的方法进行
# 但是这一次，是输入60帧，132x1x1的图像，输出，为30帧132x1x1的图像
# x_train，x_test和test都不用变，只有建模变化









def build_model2():
    '''
    输入为None 20 132， 输出为None 20 132
    :return:
    '''

    model = Sequential()



    model.add(ConvLSTM2D(
        # 可能这个filters应该再多一些
        filters=3,
        # 每一次只有一个kernel，一次卷积出100个结果，卷积的结果之间会有LSTM联系
        # 注意用summary检查参数数量，参数太多不是很好，这里如果filter是100，参数会到达1000 0000的数量级，所以这里filter改成10
        input_shape=(30, 132, 1, 1),#(n_frame, width, height, channel)
        kernel_size=(132, 1),
        padding="same",
        return_sequences=True,
        activation="selu"
    ))

    model.add(BatchNormalization())
    #model.add(Dropout(0.75))

    model.add(ConvLSTM2D(
        filters=1,
        kernel_size=(132, 3),
        padding="same",
        return_sequences=True,
        activation="selu"
    ))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Conv3D(
        filters=1,
        kernel_size=(132, 1, 1), strides=(1, 1, 1),

        activation="selu",
        padding="same",
        data_format="channels_last"

    ))


    return model


def gene_data(
        row = 132,
        col = 1,
        channel = 1,
        n_frames = 30,
        slide_window = 30
):
    # 用第一小时的30帧去预测第二小时的30帧，然后第二小时30帧预测最后一小时30帧，
    # 训练集train可以进行两次feed，测试集的已有数据也可以进行1次feed训练，更改视频长度和间隔、数量等
    # 只需要更改n_frames和slide_window就行了
    x_train = np.array(pd.read_csv("x_train.csv").drop(labels="frame", axis=1))
    y_train = np.array(pd.read_csv("y_train.csv").drop(labels="frame", axis=1))
    test = np.array(pd.read_csv("test.csv").drop(labels="frame",axis=1))

    train = np.concatenate((x_train, y_train), axis=0)
    scaler = StandardScaler().fit(train)
    joblib.dump(scaler, "scaler.npy")


    train = scaler.transform(train)
    test = scaler.transform(test)

    train = train.reshape(92, 90, 132, 1, 1)
    test = test.reshape(30, 60, 132, 1, 1)



    x_train_train = []
    y_train_train = []
    # 下面指的是用test的前一部分作为训练集
    x_test_train = []
    y_test_train = []



    n_samples = 2
    for i in range(n_samples):
        x_train_train.append(train[:,
                            n_frames*i:n_frames*(i+1),:,:,:])
        y_train_train.append(train[:,
                            n_frames * i+slide_window:n_frames * (i + 1)+slide_window, :, :, :])

    n_samples = 1
    for i in range(n_samples):
        x_test_train.append(test[:,
                             n_frames * i:n_frames * (i + 1), :, :, :])
        y_test_train.append(test[:,
                             n_frames * i + slide_window:n_frames * (i + 1) + slide_window, :, :, :])
    x_train_train = np.array(x_train_train)
    y_train_train = np.array(y_train_train)
    x_test_train = np.array(x_test_train)
    y_test_train = np.array(y_test_train)

    # 之后预测，只用test的后30帧进行
    ready_test = np.array(test[:,30:,:,:,:])


    print(x_train_train.shape)
    print(y_train_train.shape)
    print(x_test_train.shape)
    print(y_test_train.shape)
    print(ready_test.shape)

    np.savez("train_and_test_data.npz",x_train_train,y_train_train,x_test_train,y_test_train,ready_test)


def train2(train=True,load_weights=True,save_weights=True,predict=False,epoch=500):

    temp = np.load("train_and_test_data.npz")
    print(temp.files)

    x_train_train = temp["arr_0"]
    y_train_train = temp["arr_1"]
    x_test_train = temp["arr_2"]
    y_test_train = temp["arr_3"]
    ready_test  = temp["arr_4"]



    model = build_model2()

    # from keras import backend as K
    # def my_loss(y_true, y_pred):
    #     # 自定义loss，直接设置为预测准确值
    #     return K.mean(K.abs(y_pred - y_true)/y_true,axis=-1)

    model.summary()
    # compile to my own loss


    #model.compile(loss="mean_absolute_percentage_error", optimizer="adadelta")

    #model.compile(loss="mean_absolute_error", optimizer="adadelta")



    #model.compile(loss="mean_absolute_error", optimizer="adam")
    from keras import backend as K

    # 这是在loss求取前确定，避免每次求loss都调用




    def myloss(y_true,y_pred):
        # 输入是tensor对象，所以要得到eval

        # y_true = K.eval(y_true)
        # y_pred = K.eval(y_pred)
        # print(y_true)
        #
        # _y_pred = np.array(y_pred)
        # _y_pred = _y_pred.reshape((SPLIT_NUM) * 30, 132)
        # _y_pred = scaler.inverse_transform(_y_pred)
        #
        # _y_true = np.array(y_true)
        # _y_true = _y_true.reshape((SPLIT_NUM) * 30, 132)
        # _y_true = scaler.inverse_transform(_y_true)

        loss = K.mean(K.abs(K.abs(y_pred - y_true) / y_true))

        # 待会儿是否要加上验证集loss？

        #loss_2 = np.mean(np.abs(y - y_) / y_)


        return loss

    model.compile(loss=myloss, optimizer="adam")


    if load_weights:
        try:
            model.load_weights('multiply_CLSTM_weights.h5', by_name=True)
            print("load weights successful")
        except Exception as a:
            print('\033[1;31;40m str(Exception):\t', str(a), "'\033[0m'")
            train = True

    if train:
        # 增加tensorboard可视化
        tb_back = keras.callbacks.TensorBoard(log_dir='E:\\tensorboard_dir\\data_analysis_city_road_data\\Graph', histogram_freq=0,
                                    write_graph=True, write_images=True)


        # 每次会训练3组数据，input和output都是None 30 132 1 1，

        model.fit(x_train_train[0], y_train_train[0], batch_size=1000, epochs=epoch,
                  validation_data=(x_train_train[1],y_train_train[1])
                    ,callbacks=[tb_back])

        y = model.predict(x_train_train[1])


        scaler = joblib.load("scaler.npy")
        y = y.reshape(92, 30, 132)
        y = scaler.inverse_transform(y)
        #y = y.reshape(92, 30, 132, 1, 1)

        y_ = y_train_train[1]
        y_ = y_.reshape(92, 30, 132)
        y_ = scaler.inverse_transform(y_)
        #y_ = y_.reshape(92, 30, 132, 1, 1)

        abs_loss = np.mean(np.abs(y - y_) )
        loss = np.mean(np.abs(y - y_) / y_)
        print("y_", y_[0, :,:])
        print("y", y[0, :,:])
        print('\033[1;31;40m \t',"test的误差", loss,'\033[0m')
        print('\033[1;31;40m \t', "test的绝对误差", abs_loss, '\033[0m')



        model.fit(x_train_train[1], y_train_train[1], batch_size=1000, epochs=epoch,
                  validation_data=(x_train_train[1], y_train_train[1])
                  , callbacks=[tb_back])

        # 这里由于是最后一个时刻，所以不能用验证集



        model.fit(x_test_train[0], y_test_train[0], batch_size=1000, epochs=epoch,
                  validation_data=(x_test_train[0], y_test_train[0])
                  , callbacks=[tb_back])
        # 这里等同于train的第一次，所以用train的第二个来验证
        y = model.predict(x_train_train[1])
        y = y.reshape(92, 30, 132)
        y = scaler.inverse_transform(y)
        #y = y.reshape(92, 30, 132, 1, 1)

        #  这里的y_上面就已经定义好了
        abs_loss = np.mean(np.abs(y - y_))
        loss = np.mean(np.abs(y - y_) / y_)
        print("y_", y_[0, :,:])
        print("y", y[0, :,:])
        print('\033[1;31;40m \t', "test的误差", loss, '\033[0m')
        print('\033[1;31;40m \t', "test的绝对误差", abs_loss, '\033[0m')



        # 之后命令行使用tensorboard --logdir 'E:\\tensorboard_dir\\data_analysis_city_road_data\\Graph' 命令查看结果

        if save_weights:
            model.save_weights('multiply_CLSTM_weights.h5')
            print("save weights_success!")

    if predict:

        result = model.predict(ready_test)
        result = result.reshape(30*30,132)
        result = scaler.inverse_transform(result)
        result = result.reshape(30,30,132)

        np.save("result.npy",result)



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




# 首先按照条件生成x_train和x_tset等数据集，转换test，数据集会和存储在文件夹中

#gene_data()




# train save and valid
#train2(train=True,valid=True,save_weights=True)

train2(train=True,save_weights=True,epoch=100,load_weights=True)

# 最后删一下表头，重命名csv为txt即可

# 写入文件
#train2(train=False,valid=True,save_weights=False,epoch=500,load_weights=True,predict=True)
#predict_and_submit()








