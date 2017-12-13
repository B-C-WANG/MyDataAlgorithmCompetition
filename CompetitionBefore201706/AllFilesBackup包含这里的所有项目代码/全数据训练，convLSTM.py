import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.layers import ConvLSTM2D, Input,Dropout,BatchNormalization,Conv2D,Dense,Conv3D
from keras.models import  Sequential
import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.externals import  joblib
import keras



def transfer():
    data = pd.read_csv("filled_result.csv")

    test = data[data["frame"]>6000000]
    test.to_csv("test.csv",index=False)



    x_train = data[(data["frame"]<6000000)]


    x_train.to_csv("x_train_all.csv",index=False)

#transfer()


def build_model2():
    FIRST_ACTIVATION = "selu"
    ACTIVATION = "selu"
    LAST_ACTIVATION = "selu"

    RECURRENT_ACTIVATION = "hard_sigmoid"

    BATCH_NORMALIZATION = False
    # ________________________________________________________________________

    model = Sequential()

    model.add(ConvLSTM2D(

        filters=1,
        input_shape=(30, 132, 1,1),
        kernel_size=(132, 1),
        padding="same",
        return_sequences=True,

        activation=FIRST_ACTIVATION,
        recurrent_activation=RECURRENT_ACTIVATION
    ))

    if BATCH_NORMALIZATION:
        model.add(BatchNormalization())
        # model.add(Dropout(0.75))

        # ________________________________________________________________________

    model.add(ConvLSTM2D(
        filters=3,
        kernel_size=(132, 1),
        padding="same",
        return_sequences=1,
        activation=ACTIVATION,
        recurrent_activation=RECURRENT_ACTIVATION
    ))

    if BATCH_NORMALIZATION:
        model.add(BatchNormalization())
    model.add(Dropout(0.5))
    # ________________________________________________________________________


    model.add(ConvLSTM2D(
        filters=5,
        kernel_size=(132, 3),
        padding="same",
        return_sequences=1,
        activation=ACTIVATION,
        recurrent_activation=RECURRENT_ACTIVATION
    ))

    if BATCH_NORMALIZATION:
        model.add(BatchNormalization())
    model.add(Dropout(0.5))

    # ________________________________________________________________________

    model.add(ConvLSTM2D(
        filters=7,
        kernel_size=(132, 5),
        padding="same",
        return_sequences=1,
        activation=ACTIVATION,
        recurrent_activation=RECURRENT_ACTIVATION
    ))

    if BATCH_NORMALIZATION:
        model.add(BatchNormalization())
    model.add(Dropout(0.75))
    # ________________________________________________________________________


    model.add(ConvLSTM2D(
        filters=5,
        kernel_size=(132, 7),
        padding="same",
        return_sequences=1,
        activation=ACTIVATION,
        recurrent_activation=RECURRENT_ACTIVATION
    ))

    if BATCH_NORMALIZATION:
        model.add(BatchNormalization())
    model.add(Dropout(0.5))

    # ________________________________________________________________________


    model.add(ConvLSTM2D(
        filters=3,
        kernel_size=(132, 5),
        padding="same",
        return_sequences=1,
        activation=ACTIVATION,
        recurrent_activation=RECURRENT_ACTIVATION
    ))

    if BATCH_NORMALIZATION:
        model.add(BatchNormalization())
    model.add(Dropout(0.5))

    # ________________________________________________________________________

    model.add(Conv3D(
        filters=1,
        kernel_size=(132, 1, 3), strides=(1, 1, 1),
        activation=LAST_ACTIVATION,
        padding="same",
        data_format="channels_last"

    ))

    return model



def train2(train=True,load_weights=True,save_weights=True,predict=False,valid=False,epoch=500,per_epoch=100):


    # 首先载入完整数据
    data_train = np.array(pd.read_csv("x_train_all.csv").drop(labels="frame",axis=1))
    data_question = np.array(pd.read_csv("test.csv").drop(labels="frame",axis=1))
    # 问题的x只要后一个小时


    # 此时我突然发现，数据缺失了38个，为66202，没办法保证每次都是同一个时间的数据，只能够直接按照帧拆分
    data_train = data_train[:66180,:]


    print(data_train.shape)
    print(data_question.shape)

    temp = np.concatenate((data_train, data_question), axis=0)
    print(temp.shape)

    scaler = StandardScaler().fit(temp)
    joblib.dump(scaler, "scaler.npy")


    data_train = scaler.transform(data_train)
    data_question = scaler.transform(data_question)

    data_train = data_train.reshape(2206, 30, 132, 1,1)
    data_question = data_question.reshape(30, 60, 132, 1,1)[:, 30:, :, :,:]

    # test数据是train的后一帧
    x_train = data_train[0:2205,:,:,:,:]
    y_train = data_train[1:2206,:,:,:,:]


    print(x_train.shape)
    print(y_train.shape)




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
    import random
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

        randon_int = random.randint(0,66179)
        print("选择随机数{}".format(randon_int))

        # 随机选择一个进行预测
        y = model.predict(x_train[randon_int,:,:])
        y = y.reshape(30, 132)
        y = scaler.inverse_transform(y)


        #print("y_", y_[0, :])
        #print("y", y[0, :])
        y_ = y_train[randon_int,:,:,:,:]
        loss_2 = np.mean(np.abs(y - y_) / y_)

        # 采用train和test共同的loss和作为loss，避免过拟合，一味减小test误差和一味减小train误差都是同样错误的
        #
        return (loss*0.4+loss_2*0.6)*100

    def myloss2(y_true,y_pred):
        return (K.mean(K.abs(K.abs(y_pred - y_true) / y_true)))*100

    model.compile(loss=myloss2, optimizer="adam")


    if load_weights:
        try:
            model.load_weights('model_weights_30_to_30_CNN.h5', by_name=True)
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

            # 随机选择n个区间段的数据作为验证，xy是一一对应，不用+1
            N=30
            randon_int = random.randint(0, 2206-N-1)
            print("选择随机数{}".format(randon_int))
            valid_x = x_train[randon_int:randon_int + N, :,:,:,:]
            valid_y = y_train[randon_int :randon_int+N, :, :,:,:]



            model.fit(x_train, y_train, batch_size=200, epochs=per_epoch,validation_data=(valid_x,valid_y)
                    ,callbacks=[tb_back])

            N = 30
            randon_int = random.randint(0, 2206 - N - 1)
            print("选择随机数{}".format(randon_int))
            valid_x = x_train[randon_int:randon_int + N, :,:,:,:]
            valid_y = y_train[randon_int:randon_int + N, :, :,:,:]


            y = model.predict(valid_x)
            y = y.reshape(N*30,132)
            y = scaler.inverse_transform(y)
            y = y.reshape(N,30,132,1,1)

            abs_loss = np.mean(np.abs(np.abs(y - valid_y) ))
            loss = np.mean(np.abs(y - valid_y) / valid_y)

            print('\033[1;31;40m \t',"x_test的loss", loss,'\033[0m')
            print('\033[1;31;40m \t', "x_test的绝对loss", abs_loss, '\033[0m')


        # 之后命令行使用tensorboard --logdir 'E:\\tensorboard_dir\\data_analysis_city_road_data\\Graph' 命令查看结果

            if save_weights:
                model.save_weights('model_weights_30_to_30_CNN.h5')
                print("save weights_success!")

    if predict:
        test = np.array(pd.read_csv("test.csv").drop(labels="frame",axis=1))
        test = scaler.transform(test)
        test = test.reshape(30,30,132,1,1)
        result = model.predict(test)
        result = result.reshape(30*30,132)
        result = scaler.inverse_transform(result)
        result = result.reshape(30,30,132)

        np.save("result.npy",result)

    if valid:
        # 得出训练的总loss

        y = model.predict(x_train)
        y = y.reshape(2205*30,132)
        y = scaler.inverse_transform(y)
        y = y.reshape(2205,30,132,1,1)


        loss = np.mean(np.abs(np.abs(y - y_train) / y_train), axis=0)

        print("total loss",loss[:100])




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




train2(train=True,valid=True,save_weights=True,epoch=10,per_epoch=10,load_weights=True)




# 最后删一下表头，重命名csv为txt即可

# 写入文件
#train2(train=False,valid=True,save_weights=False,epoch=500,load_weights=True,predict=True)
#predict_and_submit()








