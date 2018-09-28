# coding: UTF-8

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.layers import ConvLSTM2D, Input,Dropout,BatchNormalization,Conv2D,Dense,Conv3D
from keras.models import  Sequential
import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.externals import  joblib
import keras

# Attention锛�搴熷純鍐呭�鍏ㄩ儴鍦↙STM鍗风Н鏂规硶.py涓�


# TODO锛氬彲浠ュ皾璇曞皢杩欎簺132x1澶у皬鐨勫浘鐗囩粍鎴愯�棰戞挱鏀�
# TODO锛氬彲灏濊瘯灏�32 reshape 涓�2x11鏉ュ嵎绉�紝 鍥犱负瑕佸�涔犵殑鏄�浉鍏冲叧绯伙紝鐩稿叧鐨勫叧绯诲苟涓嶆槸涓�潯绾夸笂鍙�互浜嗚В鐨�

#TODO:鍙�互灏濊瘯鍗风Н锛屽厛瑕佹�鍒欏寲

# 鏇存柊锛氶�鐩�粰鐨勬槸3涓�湀鐨勬椂闂村姞涓�鏈堜唤6鐐瑰埌8鐐圭殑鏁版嵁锛岄�娴�鐐瑰埌10鐐圭殑鏁版嵁锛岄渶瑕佹洿鏀逛竴涓嬶紒

# 鍦ㄥ～琛ヤ簡鏁版嵁鐨刦illed_result.data涓婇潰鏇存敼

def transfer():
    data = pd.read_csv("filled_result.csv")


    # 灏�鏈堢殑鏁版嵁瀛樺偍
    test = data[data["frame"]>6000000]
    test.to_csv("test.csv",index=False)

    # 绛涢�鍑篸ata涓�墍鏈�鐐瑰埌7.59鐐圭殑缁撴灉锛屾敞鎰弔ype涓篿nt64锛屽厛鐢ㄥ彇浣欏緱鍒板悗4浣嶏紝寰楀埌x_train

    x_train = data[(data["frame"]%10000>559) & (data["frame"]%10000<759) & (data["frame"]<6000000)]

    # 绛涢�鍑�.00鍒�.58鐨勬暟鎹�紝浣滀负x test

    y_train = data[(data["frame"]%10000>759)&(data["frame"]%10000<859) & (data["frame"]<6000000)]

    x_train.to_csv("x_train.csv",index=False)

    y_train.to_csv("y_train.csv",index=False)


#transfer()

# 缁撴灉 锛�x_train 92 x 60 x 132 y_train 92 x 30 x 132 鏍规嵁 30 x 60 x 132 姹傚緱 30 x 30 x 132



### 绗�簩绉嶆柟娉曪細杩樻槸閲囩敤conv2d lstm鐨勬柟娉曡繘琛�
# 浣嗘槸杩欎竴娆★紝鏄�緭鍏�0甯э紝132x1x1鐨勫浘鍍忥紝杈撳嚭锛屼负30甯�32x1x1鐨勫浘鍍�
# x_train锛寈_test鍜宼est閮戒笉鐢ㄥ彉锛屽彧鏈夊缓妯″彉鍖�









def build_model2():
    '''
    淇濊瘉杈撳叆涓篘one 60 132 杈撳嚭涓篘one 30 162
    :return:
    '''

    FIRST_ACTIVATION = "selu"
    ACTIVATION = "selu"
    LAST_ACTIVATION = "selu"

    RECURRENT_ACTIVATION = "hard_sigmoid"

    BATCH_NORMALIZATION = False
#________________________________________________________________________

    model = Sequential()
    


    model.add(ConvLSTM2D(

        filters=1,
        input_shape=(30, 132, 1, 1),
        kernel_size=(132, 1),
        padding="same",
        return_sequences=True,

        activation=FIRST_ACTIVATION,
        recurrent_activation=RECURRENT_ACTIVATION,
        kernel_initializer="random_uniform"
    ))

    if BATCH_NORMALIZATION:
        model.add(BatchNormalization())
    #model.add(Dropout(0.75))

#________________________________________________________________________


    model.add(ConvLSTM2D(
        filters=3,
        kernel_size=(132, 1),
        padding="same",
        return_sequences=1,
        activation=ACTIVATION,
        recurrent_activation=RECURRENT_ACTIVATION,
        kernel_initializer="random_uniform"
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
        recurrent_activation=RECURRENT_ACTIVATION,
        kernel_initializer="random_uniform"
    ))

    if BATCH_NORMALIZATION:
        model.add(BatchNormalization())
    model.add(Dropout(0.5))

    #________________________________________________________________________

    model.add(ConvLSTM2D(
        filters=7,
        kernel_size=(132, 5),
        padding="same",
        return_sequences=1,
        activation=ACTIVATION,
        recurrent_activation=RECURRENT_ACTIVATION,
        kernel_initializer="random_uniform"
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
        recurrent_activation=RECURRENT_ACTIVATION,
        kernel_initializer="random_uniform"
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
        recurrent_activation=RECURRENT_ACTIVATION,
        kernel_initializer="random_uniform"
    ))

    if BATCH_NORMALIZATION:
        model.add(BatchNormalization())
    model.add(Dropout(0.5))

    #________________________________________________________________________

    model.add(Conv3D(
        filters=1,
        kernel_size=(132, 1, 3), strides=(1, 1, 1),
        activation=LAST_ACTIVATION,
        padding="same",
        data_format="channels_last",
        kernel_initializer="random_uniform"

    ))

    return model




def train2(train=True,load_weights=True,save_weights=True,predict=False,valid=False,epoch=500,per_epoch=100):


    x_train = np.array(pd.read_csv("x_train.csv").drop(labels="frame",axis=1))
    y_train = np.array(pd.read_csv("y_train.csv").drop(labels="frame",axis=1))
    #test = np.array(pd.read_csv("test.csv").drop(labels="frame",axis=1))

    # 涔嬪悗锛氬皢x_train鍜寉_train鍚堝苟杩涜�姝ｅ垯鍖栵紝test鐩稿悓鐨勬柟寮忔�鍒欏寲锛堟垨鑰呮墍鏈夌殑y閮芥寜鐓�鐨勬柟寮忔�鍒欏寲锛屽厛璇曡瘯鍝�釜鏇村ソ锛�

    temp = np.concatenate((x_train, y_train), axis=0)  # concat寰堝�鏄擄紝鎯冲姞鍝�釜灏遍拡瀵箂hape涓�殑閭ｄ釜浣嶇疆鍔犱笂
    print(temp.shape)
      # 姝ｅ垯鍖杅it闇��dim<=2
    scaler = StandardScaler().fit(temp)
    joblib.dump(scaler, "scaler.npy")

    x_train = scaler.transform(x_train)
    y_train = scaler.transform(y_train)



    # 鐢变簬鏃堕棿鏄�繛鍦ㄤ竴璧风殑锛屾墍浠ヨ�灏�0鏀惧湪绗�竴缁村害
    x_train = x_train.reshape(92, 60,132,1,1)
    y_train = y_train.reshape(92,30,132,1,1)

#  鍙�敤鏀瑰彉杩欓噷锛屽氨鑳藉�鏀瑰彉杈撳叆
    x_train = x_train[:,30:,:,:,:]
    print(x_train.shape)


    #print(x_train[0,0,:,0,0])# 杩欎釜搴旇�鏄痗sv鐨勭�涓��
    #print(x_train[0,:,0,0,0])#杩欎釜搴旇�鏄痗sv灞炰簬鍚屼竴澶╃殑涓�垪鏁版嵁
    # 涓婇潰涓や釜灏辨槸涓�ぉ鐨勬暟鎹�潡锛宻hape涓�0 x 132

    #test = test.reshape(30,60,132,1,1)


    # 鍒掑垎璁�粌闆嗗拰娴嬭瘯闆嗕究浜庝簡瑙ｈ繃鎷熷悎鎯呭喌

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
    #     # 鑷�畾涔塴oss锛岀洿鎺ヨ�缃�负棰勬祴鍑嗙‘鍊�
    #     return K.mean(K.abs(y_pred - y_true)/y_true,axis=-1)

    model.summary()
    # compile to my own loss


    #model.compile(loss="mean_absolute_percentage_error", optimizer="adadelta")

    #model.compile(loss="mean_absolute_error", optimizer="adadelta")



    #model.compile(loss="mean_absolute_error", optimizer="adam")
    from keras import backend as K

    # 杩欐槸鍦╨oss姹傚彇鍓嶇‘瀹氾紝閬垮厤姣忔�姹俵oss閮借皟鐢�
    y_ = np.array(pd.read_csv("y_train.csv").drop(labels="frame", axis=1))
    y_ = y_.reshape(92, 30, 132, 1, 1)[SPLIT_NUM:, :, :, :, :]
    y_ = y_.reshape((92 - SPLIT_NUM) * 30, 132)

    def myloss(y_true,y_pred):
        # 杈撳叆鏄痶ensor瀵硅薄锛屾墍浠ヨ�寰楀埌eval

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

        y = model.predict(x_test_)
        y = y.reshape((92 - SPLIT_NUM) * 30, 132)
        y = scaler.inverse_transform(y)


        #print("y_", y_[0, :])
        #print("y", y[0, :])
        loss_2 = np.mean(np.abs(y - y_) / y_)

        # 閲囩敤train鍜宼est鍏卞悓鐨刲oss鍜屼綔涓簂oss锛岄伩鍏嶈繃鎷熷悎锛屼竴鍛冲噺灏弔est璇�樊鍜屼竴鍛冲噺灏弔rain璇�樊閮芥槸鍚屾牱閿欒�鐨�
        #
        return (loss*0.4+loss_2*0.6)*100

    def myloss2(y_true,y_pred):
        return (K.mean(K.abs(K.abs(y_pred - y_true) / y_true)))*100

    model.compile(loss=myloss2, optimizer="adam")


    if load_weights:
        try:
            model.load_weights('model_weights_30_to_30.h5', by_name=True)
            print("load weights successful")
        except Exception as a:
            print('\033[1;31;40m str(Exception):\t', str(a), "'\033[0m'")
            train = True

    if train:
        # 澧炲姞tensorboard鍙��鍖�
        tb_back = keras.callbacks.TensorBoard(log_dir='E:\\tensorboard_dir\\data_analysis_city_road_data\\Graph', histogram_freq=0,
                                    write_graph=True, write_images=True)


        # 姣�0涓猠poch鏄剧ず涓��璇�樊
        for i in range(int(epoch/per_epoch)):

            # 鍙��鍙栦竴閮ㄥ垎x_train
            x_train_ = x_train_[:45,:,:,:,:]
            y_train_ = y_train_[:45, :, :, :, :]


            model.fit(x_train_, y_train_, batch_size=1, epochs=per_epoch,validation_data=(x_test_,y_test_)
                    ,callbacks=[tb_back])

            y = model.predict(x_test_)
            y = y.reshape((92 - SPLIT_NUM) * 30, 132)
            y = scaler.inverse_transform(y)

            y_ = np.array(pd.read_csv("y_train.csv").drop(labels="frame", axis=1))
            y_ = y_.reshape(92, 30, 132, 1, 1)[SPLIT_NUM:, :, :, :, :]
            y_ = y_.reshape((92 - SPLIT_NUM) * 30, 132)

            abs_loss = np.mean(np.abs(y - y_) )
            loss = np.mean(np.abs(y - y_) / y_)

            print('\033[1;31;40m \t',"x_test鐨刲oss", loss,'\033[0m')
            print('\033[1;31;40m \t', "x_test鐨勭粷瀵筶oss", abs_loss, '\033[0m')


        # 涔嬪悗鍛戒护琛屼娇鐢╰ensorboard --logdir 'E:\\tensorboard_dir\\data_analysis_city_road_data\\Graph' 鍛戒护鏌ョ湅缁撴灉

            if save_weights:
                model.save_weights('model_weights_30_to_30.h5')
                print("save weights_success!")

    if predict:
        test = np.array(pd.read_csv("test.csv").drop(labels="frame",axis=1))
        test = scaler.transform(test)
        test = test.reshape(30,60,132,1,1)
        result = model.predict(test)
        result = result.reshape(30*30,132)
        result = scaler.inverse_transform(result)
        result = result.reshape(30,30,132)

        np.save("result.npy",result)

    if valid:
        # 寰楀嚭璁�粌鐨勬�loss

        y = model.predict(x_test_)
        y = y.reshape((92-SPLIT_NUM) * 30, 132)
        y = scaler.inverse_transform(y)


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

        print("x_test鐨刲oss",loss)



        y = model.predict(x_train)
        y = y.reshape(92 * 30, 132)
        y = scaler.inverse_transform(y)

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

        print("鎵�湁train鐨刲oss", loss)


def predict_and_submit():
    prediction = np.load("result.npy")
    print(prediction[0,0,:])
    print(prediction.shape)

    # 棣栧厛寤虹珛绗�簩鍒�date鍜岀�涓夊垪time
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


                        # 涓嬮潰鐨勪唬鐮佸拰涓婇潰鐨勫熀鏈�竴鑷达紝鍙�槸鏃ユ湡{}鍓嶉潰涓嶅姞0
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

    # 寤虹珛绗�竴鍒�link

    link_name = pd.read_csv("test.csv")
    link_name = link_name.keys()
    link_name = list(link_name)[1:]
    print(len(link_name),"\n",link_name)

    link =[]
    for i in range(900):
        for z in range(len(link_name)):
            link.append(link_name[z])

    data =[]

    #  浠庝笂鑷充笅锛屼緷娆℃槸涓嶅悓link锛屽悓涓�ime date 涔嬪悗鏄�笉鍚宼ime 鏈�悗鏄�笉鍚宒ate
    # reshape鐨勭�涓�淮搴︽槸date锛岀�浜岀淮搴︽槸time锛岀�涓夌淮搴︽槸link
    for i in range(30):
        for j in range(30):
            for k in range(132):
                # 姣忔�娣诲姞绗�澶╃�1time 鎵�湁link鏁版嵁锛�
                # 鐒跺悗鏄��1澶╋紝绗�time 鎵�湁link鏁版嵁锛屼緷娆′笅鍘�

                # 棰勬祴鐨勮礋鍊煎�鐞�
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




train2(train=True,valid=True,save_weights=True,epoch=1,per_epoch=1,load_weights=False)




# 鏈�悗鍒犱竴涓嬭〃澶达紝閲嶅懡鍚峜sv涓簍xt鍗冲彲

# 鍐欏叆鏂囦欢
#train2(train=False,valid=True,save_weights=False,epoch=500,load_weights=True,predict=True)
#predict_and_submit()








