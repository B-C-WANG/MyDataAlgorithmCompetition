import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.layers import ConvLSTM2D, Input,Dropout,Activation,BatchNormalization,Conv3D
from keras.models import  Sequential
import tqdm



# 思路：一个frame是1列，先用132个132长度的卷积核，对这一列卷积，分别得到下一列的值，至此，就可以进行训练了！
# 先直接采用keras的conv2d lstm试试，把每一帧当作一张图片！一张1 x 132的图片

# 当然也可以当成 n x 132 的图片，然后每次窗口移动1或者2


# 将数据转换成numpy格式
def pandas_to_numpy():
    data  = pd.read_csv("filled_result.csv")
    data = data.drop(labels="frame",axis=1)
    data = np.array(data)
    print(data)
    print(data.shape)
    np.save("original_data.npy",data)
#pandas_to_numpy()










# 建立conv2d lstm自编码器模型
# 预测的结果是900 * 132， 可以直接就以900 132 为一张图片进行预测，滑动窗口长度为900帧
# 也可以另外设为其他大小的滑动窗口，不用设置一帧

# 最终决定：1 x 132 作为图片，30 x 1 x 132 就是30帧的图片，

# 卷积核需要大一点儿，保证各种数据能够有所联系，当然可以采用较大kernel

def make_model(input_frame=450):
    model = Sequential()


    model.add(ConvLSTM2D(
        filters=10,
        # 每一次只有一个kernel，一次卷积出100个结果，卷积的结果之间会有LSTM联系
        # 注意用summary检查参数数量，参数太多不是很好，这里如果filter是100，参数会到达1000 0000的数量级，所以这里filter改成10
        input_shape=(input_frame,132,1,1),
        kernel_size=(132, 1),
        padding="same",
        return_sequences=True
    ))

    model.add(BatchNormalization())

    model.add(ConvLSTM2D(
        filters=10,
        kernel_size=(132, 1),
        padding="same",
        return_sequences=True
    ))

    model.add(BatchNormalization())

    model.add(ConvLSTM2D(
        filters=10,
        kernel_size=(132, 1),
        padding="same",
        return_sequences=True
    ))

    model.add(BatchNormalization())


    model.add(ConvLSTM2D(
        filters=10,
        kernel_size=(132, 1),
        padding="same",
        return_sequences=True
    ))

    model.add(BatchNormalization())

    # 用Conv3D加上filter为1的Conv，得到一个视频
    # 卷积核除了图片的长宽外，还多了一个第三维，指的是每几帧添加到一起
    model.add(Conv3D(
        filters=1,
        kernel_size=(132,1,3),
        activation="sigmoid",
        padding="same",
        data_format="channels_last"

    ))



    return model

# 准备数据

def gene_data(row=132,
              col=1,
              n_frames=720,# 一次生成900帧的图片处理，每次900帧的视频

              # 更新：由于OOM，改成450帧（900帧提示OOM，可将batch_size和epoch设低一点儿）

              # 更新：为了加快，改成225帧（主要为了不出现超过缓存的情况，450帧可以运行，但是有警告）

              # 更新：由于一天，24小时为720帧，所以这里slide window和frame都设置为720帧，相当于根据前一天得到后一天，
              #视频也就是每天的量，这样更加合理



              slide_window=720,

):
    n_samples = int((68002-n_frames)/slide_window)-1  # 按照100的窗口滑动，每次给的视频向后移动100帧，滑动671次即可，但是要给下一个视频多留出一帧

    # (68002 - 900) / 100 ~= 671
    # 671 * 100 + 900 = 68000
    # 670 = 671 -1

    # 相当于有671个900帧1x132大小的视频，每次用前一个视频输入，后一个输出
    # 最终需要得到的结果是(671 x 900 x 132 x 1 x 1)最后一个是通道，原来矩阵的8.9倍大

    data = np.load("original_data.npy")
    print(data)
    print(data.shape)

    last_video = []
    future_video = []




    start_frame = 0
    for _ in tqdm.trange(n_samples):

        last_video.append(data[start_frame:start_frame+n_frames,:])
        # future要向前100帧，因为是下一个视频
        future_video.append(data[start_frame+slide_window:start_frame+n_frames+slide_window,:])
        start_frame += slide_window

    last = np.array(last_video)
    future = np.array(future_video)
    print(last.shape)
    print(future.shape)
    # 讲道理，可以从last推断出future的，但是为了训练集的纯粹，就这样吧

    np.save("last_videos.npy",last)
    np.save("future_videos.npy",future)

#gene_data()





# 进行训练
def train(train=True,save_weights=True,load_wights=True):


        last = np.load("last_videos.npy")
        future = np.load("future_videos.npy")

        shape = last.shape

        print(shape)

        last = last.reshape((shape[0],shape[1],shape[2],1,1))
        future = future.reshape((shape[0], shape[1], shape[2], 1, 1))

        # 根据frame修改模型
        model = make_model(input_frame=shape[1])

        model.summary()
        model.compile(loss="binary_crossentropy", optimizer="adadelta")


        if load_wights:
            try:
                model.load_weights('my_model_weights_of_keras.h5', by_name=True)
                print("load weights successful")
            except Exception as a:
                print('\033[1;31;40m str(Exception):\t', str(a), "'\033[0m'")
                train = True


        if train:
            # batch_size小一点儿，不然容易OOM
            model.fit(last,future,batch_size=1,epochs=1)









        if save_weights:
            model.save_weights('my_model_weights_of_keras.h5')




#train()

# Attention： 以上全为废弃内容！！！！


# TODO：可以尝试将这些132x1大小的图片组成视频播放
# TODO：可尝试将132 reshape 为12x11来卷积， 因为要学习的是相关关系，相关的关系并不是一条线上可以了解的

# 更新：题目给的是3个月的时间加上6月份6点到8点的数据，预测8点到10点的数据，需要更改一下！

# 在填补了数据的filled_result.data上面更改

def transfer():
    data = pd.read_csv("filled_result.csv")


    # 将6月的数据存储
    data_w = data[data["frame"]>6000000]
    data_w.to_csv("test.csv",index=False)

    # 筛选出data中所有6点到7.59点的结果，注意type为int64，先用取余得到后4位，得到x_train

    x_train = data[(data["frame"]%10000>559) & (data["frame"]%10000<759) & (data["frame"]<6000000)]

    # 筛选出8.00到8.58的数据，作为x test

    x_test = data[(data["frame"]%10000>759)&(data["frame"]%10000<859) & (data["frame"]<6000000)]

    data.to_csv("train_data.csv",index=False)


transfer()