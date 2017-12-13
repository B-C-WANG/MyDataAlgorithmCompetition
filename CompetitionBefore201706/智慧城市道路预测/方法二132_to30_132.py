import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.layers import ConvLSTM2D, Input,Dropout,BatchNormalization,Conv2D,Dense,Conv3D,Activation,Add
from keras.layers import Merge  as merge
from keras.models import  Sequential,Model
import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.externals import  joblib
import keras
import keras.backend as K






def build_model2():
    '''
    保证输入为None 60 132 输出为None 30 162
    :return:
    '''


    ACTIVATION = "relu"

#________________________________________________________________________




    input = Input(shape=(1,132,))
    result = []
    now_dense = Dense(132,activation=ACTIVATION)(input)
    result.append(now_dense)
    for i in range(1,30):
        now_dense = Dense(132,activation=ACTIVATION)(now_dense)

        result.append(now_dense)

    #add = keras.layers.Add()(result)

    result = keras.layers.concatenate(inputs=result,axis=1)
    print(result)

    model = Model(input=input,outputs=result)


    model.summary()

    # input: 132, output 30, 132


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


    temp = np.concatenate((x_train, y_train), axis=0)



    line_normalization_fit(temp)

    x_train = line_normalization_transform(x_train)
    y_train = line_normalization_transform(y_train)


    print(line_normalization_reverse(x_train)[0,:])
    print(line_normalization_reverse(x_train)[1, :])



    x_train = x_train.reshape(92, 60,132)
    y_train = y_train.reshape(92,30,132)


    x_train = x_train[:,59:,:]
    # x 为 92， 1， 132, y 为 92 30 132




    SPLIT_NUM = 62

    x_train_ = x_train[0:SPLIT_NUM,:]
    y_train_ = y_train[0:SPLIT_NUM, :, :]

    x_test_ = x_train[SPLIT_NUM:,:]
    y_test_ = y_train[SPLIT_NUM:, :, :]


    print("x_train_shape",x_train_.shape)# e:  62, 132
    print("y_train_shape",y_train_.shape)# e:  62, 30 ,132
    print("x_test_shape", x_test_.shape)# e: 30, 132
    print("y_test_shape", y_test_.shape)# e: 30, 30, 132




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
    y_ = y_.reshape(92, 30, 132)[SPLIT_NUM:, :, :]
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
        print(y.shape)
        y = y.reshape((92 - SPLIT_NUM) * 30, 132)
        y = line_normalization_reverse(y)


        #print("y_", y_[0, :])
        #print("y", y[0, :])
        loss_2 = np.mean(np.abs((np.square(y - y_) / y_))

        # 采用train和test共同的loss和作为loss，避免过拟合，一味减小test误差和一味减小train误差都是同样错误的
        #
        return (loss*0.4+loss_2*0.6)*100

    def myloss2(y_true,y_pred):
        return (K.mean(K.abs(K.square(y_pred - y_true) / y_true)))*100

    def myloss3(y_true,y_pred):
        return (K.mean(K.abs(K.square(K.square(y_pred - y_true)) / y_true)))*100

    model.compile(loss=myloss, optimizer="adam")


    if load_weights:
        try:
            model.load_weights('model_weights_dense.h5', by_name=True)
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



            model.fit(x_train_, y_train_, batch_size=620, epochs=per_epoch,validation_data=(x_test_,y_test_)
                    ,callbacks=[tb_back])

            y = model.predict(x_test_)
            y = y.reshape((92 - SPLIT_NUM) * 30, 132)
            y = line_normalization_reverse(y)

            y_ = np.array(pd.read_csv("y_train.csv").drop(labels="frame", axis=1))
            y_ = y_.reshape(92, 30, 132)[SPLIT_NUM:, :, :]
            y_ = y_.reshape((92 - SPLIT_NUM) * 30, 132)

            abs_loss = np.mean(np.abs(y - y_) )
            loss = np.mean(np.abs(y - y_) / y_)

            print('\033[1;31;40m \t',"x_test的loss", loss,'\033[0m')
            print('\033[1;31;40m \t', "x_test的绝对loss", abs_loss, '\033[0m')


        # 之后命令行使用tensorboard --logdir 'E:\\tensorboard_dir\\data_analysis_city_road_data\\Graph' 命令查看结果

            if save_weights:
                model.save_weights('model_weights_dense.h5')
                print("save weights_success!")

    if predict:
        test = np.array(pd.read_csv("test.csv").drop(labels="frame",axis=1))
        test = line_normalization_transform(test)
        test = test.reshape(30,60,132)
        test =test[:,59:,:]
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
        y_ = y_.reshape(92, 30, 132)[SPLIT_NUM:,:,:]
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




train2(train=True,valid=True,save_weights=True,epoch=5000,per_epoch=1000,load_weights=True)




# 最后删一下表头，重命名csv为txt即可

# 写入文件
#train2(train=False,valid=True,save_weights=False,epoch=500,load_weights=True,predict=True)
#predict_and_submit()







