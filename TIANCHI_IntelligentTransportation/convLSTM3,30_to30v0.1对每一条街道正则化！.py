import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.layers import ConvLSTM2D, Input,Dropout,BatchNormalization,Conv2D,Dense,Conv3D,Activation
from keras.models import  Sequential
import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.externals import  joblib
import keras

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
    保证输入为None 60 132 输出为None 30 162
    :return:
    '''

    FIRST_ACTIVATION = "relu"
    ACTIVATION = "relu"
    LAST_ACTIVATION = "relu"

    RECURRENT_ACTIVATION = "hard_sigmoid"

    BATCH_NORMALIZATION = True
#________________________________________________________________________

    model = Sequential()

    model.add(ConvLSTM2D(

        filters=1,
        input_shape=(30, 132, 1, 1),
        kernel_size=(132, 1),
        padding="same",
        return_sequences=True,


        recurrent_activation=RECURRENT_ACTIVATION
    ))

    if BATCH_NORMALIZATION:
        model.add(BatchNormalization())
    model.add(Activation(activation=FIRST_ACTIVATION))
    #model.add(Dropout(0.75))

#________________________________________________________________________


    model.add(ConvLSTM2D(
        filters=3,
        kernel_size=(132, 1),
        padding="same",
        return_sequences=1,

        recurrent_activation=RECURRENT_ACTIVATION
    ))

    if BATCH_NORMALIZATION:
        model.add(BatchNormalization())
    model.add(Activation(activation=ACTIVATION))
    model.add(Dropout(0.5))
    # ________________________________________________________________________


    model.add(ConvLSTM2D(
        filters=5,
        kernel_size=(132, 3),
        padding="same",
        return_sequences=1,

        recurrent_activation=RECURRENT_ACTIVATION
    ))

    if BATCH_NORMALIZATION:
        model.add(BatchNormalization())
    model.add(Activation(activation=ACTIVATION))
    model.add(Dropout(0.5))

    #________________________________________________________________________

    model.add(ConvLSTM2D(
        filters=7,
        kernel_size=(132, 5),
        padding="same",
        return_sequences=1,

        recurrent_activation=RECURRENT_ACTIVATION
    ))

    if BATCH_NORMALIZATION:
        model.add(BatchNormalization())
    model.add(Activation(activation=ACTIVATION))
    model.add(Dropout(0.75))
    # ________________________________________________________________________


    model.add(ConvLSTM2D(
        filters=5,
        kernel_size=(132, 7),
        padding="same",
        return_sequences=1,

        recurrent_activation=RECURRENT_ACTIVATION
    ))

    if BATCH_NORMALIZATION:
        model.add(BatchNormalization())
    model.add(Activation(activation=ACTIVATION))
    model.add(Dropout(0.5))

    # ________________________________________________________________________


    model.add(ConvLSTM2D(
        filters=3,
        kernel_size=(132, 5),
        padding="same",
        return_sequences=1,

        recurrent_activation=RECURRENT_ACTIVATION
    ))

    if BATCH_NORMALIZATION:
        model.add(BatchNormalization())
    model.add(Activation(activation=ACTIVATION))
    model.add(Dropout(0.5))

    #________________________________________________________________________

    model.add(Conv3D(
        filters=1,
        kernel_size=(132, 1, 3), strides=(1, 1, 1),

        padding="same",
        data_format="channels_last"

    ))
    model.add(BatchNormalization())
    model.add(Activation(activation=LAST_ACTIVATION))

    return model



def line_normalization_fit(array):
    assert array.shape[1] == 132

    for i in range(array.shape[1]):
        scaler = StandardScaler()
        # -1就相当于None
        scaler.fit(array[:,i].reshape(-1,1))
        joblib.dump(scaler,"scaler/scaler_{}.npy".format(i))


def line_normalization_transform(array):
    assert array.shape[1] == 132

    scaler = joblib.load("scaler/scaler_{}.npy".format(0))
    result = scaler.transform(array[:, 0].reshape(-1,1))
    for i in range(1, array.shape[1]):
        scaler = joblib.load("scaler/scaler_{}.npy".format(i))
        temp = scaler.transform(array[:, i].reshape(-1,1))

        result = np.concatenate((result, temp), axis=1)
    print("after inverse scaler:", result.shape)
    return result




def line_normalization_reverse(array):
    assert  array.shape[1] ==132

    scaler = joblib.load("scaler/scaler_{}.npy".format(0))
    result = scaler.inverse_transform(array[:,0].reshape(-1,1))
    result = result.reshape(-1,1)
    for i in range(1,array.shape[1]):
        scaler = joblib.load("scaler/scaler_{}.npy".format(i))
        temp = scaler.inverse_transform(array[:,i].reshape(-1,1))
        temp = temp.reshape(-1, 1)
        result = np.concatenate((result,temp),axis=1)
    print("after inverse scaler:",result.shape)
    return result






def train2(train=True,load_weights=True,save_weights=True,predict=False,valid=False,epoch=500,per_epoch=100):


    x_train = np.array(pd.read_csv("x_train.csv").drop(labels="frame",axis=1))
    y_train = np.array(pd.read_csv("y_train.csv").drop(labels="frame",axis=1))
    #test = np.array(pd.read_csv("test.csv").drop(labels="frame",axis=1))

    # 之后：将x_train和y_train合并进行正则化，test相同的方式正则化（或者所有的y都按照x的方式正则化，先试试哪个更好）

    temp = np.concatenate((x_train, y_train), axis=0)  # concat很容易，想加哪个就针对shape中的那个位置加上
    print(temp.shape)
      # 正则化fit需要dim<=2

    print(temp[0,:])
    print(temp[1, :])
    #print(temp[61,:])

    line_normalization_fit(temp)
    print(x_train.shape)
    x_train = line_normalization_transform(x_train)
    y_train = line_normalization_transform(y_train)

    # print(x_train[0,:])
    # print(y_train[1,:])
    #
    print(line_normalization_reverse(x_train)[0,:])
    print(line_normalization_reverse(x_train)[1, :])
    # print(line_normalization_reverse(y_train)[1, :])
    #


    # 由于时间是连在一起的，所以要将60放在第一维度
    x_train = x_train.reshape(92, 60,132,1,1)
    y_train = y_train.reshape(92,30,132,1,1)

#  只用改变这里，就能够改变输入
    x_train = x_train[:,30:,:,:,:]
    print(x_train.shape)


    #print(x_train[0,0,:,0,0])# 这个应该是csv的第一行
    #print(x_train[0,:,0,0,0])#这个应该是csv属于同一天的一列数据
    # 上面两个就是一天的数据块，shape为60 x 132

    #test = test.reshape(30,60,132,1,1)


    # 划分训练集和测试集便于了解过拟合情况

    SPLIT_NUM = 62

    x_train_ = x_train[0:SPLIT_NUM,:,:,:,:]
    y_train_ = y_train[0:SPLIT_NUM, :, :, :, :]

    x_test_ = x_train[SPLIT_NUM:,:,:,:,:]
    y_test_ = y_train[SPLIT_NUM:, :, :, :, :]


    print(x_train_.shape)
    print(x_test_.shape)




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
    y_ = np.array(pd.read_csv("y_train.csv").drop(labels="frame", axis=1))
    y_ = y_.reshape(92, 30, 132, 1, 1)[SPLIT_NUM:, :, :, :, :]
    y_ = y_.reshape((92 - SPLIT_NUM) * 30, 132)

    def myloss(y_true,y_pred):
        # 输入是tensor对象，所以要得到eval

        # y_true = K.eval(y_true)
        # y_pred = K.eval(y_pred)
        # print(y_true)
        #
        # _y_pred = np.array(y_pred)
        # _y_pred = _y_pred.reshape((SPLIT_NUM) * 30, 132)
        # _y_pred = line_normalization_reverse()(_y_pred)
        #
        # _y_true = np.array(y_true)
        # _y_true = _y_true.reshape((SPLIT_NUM) * 30, 132)
        # _y_true = line_normalization_reverse()(_y_true)

        loss = K.mean(K.abs(K.square(y_pred - y_true) / y_true))

        y = model.predict(x_test_)
        y = y.reshape((92 - SPLIT_NUM) * 30, 132)
        y = line_normalization_reverse(y)


        #print("y_", y_[0, :])
        #print("y", y[0, :])
        loss_2 = np.mean(np.abs(np.square(y - y_) / y_))

        # 采用train和test共同的loss和作为loss，避免过拟合，一味减小test误差和一味减小train误差都是同样错误的
        #
        return (loss*0.4+loss_2*0.6)*100

    def myloss2(y_true,y_pred):
        return (K.mean(K.abs(K.square(y_pred - y_true) / y_true)))*100

    model.compile(loss=myloss, optimizer="adam")


    if load_weights:
        try:
            model.load_weights('model_weights_30_to_30.h5', by_name=True)
            print("load weights successful")
        except Exception as a:
            print('\033[1;31;40m str(Exception):\t', str(a), "'\033[0m'")
            train = True

    if train:
        # 增加tensorboard可视化
        tb_back = keras.callbacks.TensorBoard(log_dir='E:\\tensorboard_dir\\data_analysis_city_road_data\\Graph', histogram_freq=0,
                                    write_graph=True, write_images=True)


        # 每10个epoch显示一次误差
        for i in range(int(epoch/per_epoch)):

            # 只选取一部分x_train
            x_train_ = x_train_[:45,:,:,:,:]
            y_train_ = y_train_[:45, :, :, :, :]


            model.fit(x_train_, y_train_, batch_size=1, epochs=per_epoch,validation_data=(x_test_,y_test_)
                    ,callbacks=[tb_back])

            y = model.predict(x_test_)
            y = y.reshape((92 - SPLIT_NUM) * 30, 132)
            y = line_normalization_reverse(y)

            y_ = np.array(pd.read_csv("y_train.csv").drop(labels="frame", axis=1))
            y_ = y_.reshape(92, 30, 132, 1, 1)[SPLIT_NUM:, :, :, :, :]
            y_ = y_.reshape((92 - SPLIT_NUM) * 30, 132)

            abs_loss = np.mean(np.abs(y - y_) )
            loss = np.mean(np.abs(y - y_) / y_)

            print('\033[1;31;40m \t',"x_test的loss", loss,'\033[0m')
            print('\033[1;31;40m \t', "x_test的绝对loss", abs_loss, '\033[0m')


        # 之后命令行使用tensorboard --logdir 'E:\\tensorboard_dir\\data_analysis_city_road_data\\Graph' 命令查看结果

            if save_weights:
                model.save_weights('model_weights_30_to_30.h5')
                print("save weights_success!")













    if predict:
        test = np.array(pd.read_csv("test.csv").drop(labels="frame",axis=1))
        test = line_normalization_transform(test)
        test = test.reshape(30,60,132,1,1)
        result = model.predict(test)
        result = result.reshape(30*30,132)
        result = line_normalization_reverse(result)
        result = result.reshape(30,30,132)

        np.save("result.npy",result)

    if valid:
        # 得出训练的总loss

        y = model.predict(x_test_)
        y = y.reshape((92-SPLIT_NUM) * 30, 132)
        y = line_normalization_reverse(y)


        y_ = np.array(pd.read_csv("y_train.csv").drop(labels="frame", axis=1))
        y_ = y_.reshape(92, 30, 132, 1, 1)[SPLIT_NUM:,:,:,:,:]
        y_ = y_.reshape((92-SPLIT_NUM) * 30, 132)

        print("y_",y_[0,:])
        print("y",y[0,:])
        print("y_", y_[:, 0])
        print("y", y[:, 0])
        z = np.abs(y-y_)
        print(z[0,:])
        loss = np.mean((np.abs(y - y_) / y_), axis=0)

        print("x_test的loss",loss)



        y = model.predict(x_train)
        y = y.reshape(92 * 30, 132)
        y = line_normalization_reverse(y)

        y_ = np.array(pd.read_csv("y_train.csv").drop(labels="frame", axis=1))

        y_ = y_.reshape(92 * 30, 132)

        print("y_", y_[0, :])
        print("y", y[0, :])
        print("y_", y_[:, 0])
        print("y", y[:, 0])
        z = np.abs(y - y_)
        print(z[0, :])
        #loss = np.mean(np.abs(y - y_) / y_)
        loss = np.mean((np.abs(y - y_) / y_),axis=0)

        print("所有train的loss", loss)


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




# train save and valid
#train2(train=True,valid=True,save_weights=True)




train2(train=True,valid=True,save_weights=True,epoch=2,per_epoch=1,load_weights=True)




# 最后删一下表头，重命名csv为txt即可

# 写入文件
#train2(train=False,valid=True,save_weights=False,epoch=500,load_weights=True,predict=True)
#predict_and_submit()








