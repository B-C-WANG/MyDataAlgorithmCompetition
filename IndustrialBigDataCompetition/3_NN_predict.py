import pandas as pd
import numpy as np

X1_FILE_PATH = "x1_train_set_normalization.csv"
X2_FILE_PATH =  "x2_train_set_normalization.csv"


# 这里为了保证数据集的完整性，x1全部作为train，x2全部作为test，然后用lstm去预测
def get_train_data():
    x1 = pd.read_csv(X1_FILE_PATH)
    x2 = pd.read_csv(X2_FILE_PATH)
    x1 = np.array(x1)
    x2 = np.array(x2)
    #不要忘了array是只有一个括号，里面用逗号分开，和list不同!
    x_train = x1[:,:-1]
    y_train = x1[:,-1]
    x_test = x2[:,:-1]
    y_test = x2[:,-1]

    np.savez("x1_x2_buf.npz", x_train, y_train, x_test, y_test)


#get_train_data()




def load_data_set(file_path):
    tem = np.load(file_path)
    return tem["arr_0"],tem["arr_1"],tem["arr_2"],tem["arr_3"]



x_train, y_train, x_test, y_test = load_data_set("x1_x2_buf.npz")

print(x_train,x_train.shape, '\n',x_test,x_test.shape)
print(y_train,y_train.shape,'\n',y_test,y_test.shape)

# finished
def one_hot_code():
    new_y_train = []
    new_y_test = []
    for i in range(y_train.shape[0]):
            if y_train[i] <0.5 :
                new_y_train.append([0,1])
            else:
                new_y_train.append([1, 0])


    for i in range(y_test.shape[0]):

            if y_test[i] <0.5 :
                new_y_test.append([0, 1])
            else:
                new_y_test.append([1, 0])
    return np.array(new_y_train),np.array(new_y_test)

y_train,y_test = one_hot_code()

print(y_train,y_train.shape,'\n',y_test,y_test.shape)

from keras.models import Sequential
from keras.layers import Dense,Input,Dropout,Activation

TEST_PATH = "08_data.csv"
import matplotlib.pyplot as new_plt
# finished
def plot_prediction(prediction,true_prediction=None):# 其实可以用try语句
    if true_prediction is None :
        index = [i for i in range(len(prediction))]
        new_plt.plot(index,prediction)
        new_plt.savefig("plot_prediction_NN.png",dpi=300)
        new_plt.show()
    else:
        index = [i for i in range(len(prediction))]
        new_plt.plot(index, prediction,color="b")
        new_plt.plot(index,true_prediction,color="r")
        new_plt.savefig("plot_prediction_NN.png", dpi=300)
        new_plt.show()



def write_prediction(predictions):
    data = pd.read_csv(TEST_PATH)
    prediction = pd.Series(predictions)
    print("prediction\n",prediction)
    data.insert(data.shape[1], "predictions", predictions)
    data.to_csv(path_or_buf="test_with_prediction.csv",index = False)
    return 1
def write_result(prediction):
    startT = []
    endT = []
    now = prediction[0]


    for i in range(len(prediction)):
        if prediction[i] != now :
            if prediction[i] ==1:
                startT.append(i)
            if prediction[i]==0:
                endT.append(i-1)
            now = prediction[i]

    print(startT,"\n",endT)
    print(len(startT),len(endT))

    file = open("test1_08_results.csv","w")

    str_  = ""
    for i in range(len(startT)):
        str_ = str_ + str(startT[i])+","+str(endT[i])+"\n"
    #print(str_)
    file.write("startTime,endTime\n"+str_)
    file.close()
    return startT,endT
def get_test_x():
    return np.load("test_data_buf.npy")

def model_test(train=True,write=False,load_model=False):
    model = Sequential()

    # 2017年7月30日修改，去掉group这一组
    model.add(Dense(26,activation="tanh",bias_initializer="uniform",
                    kernel_initializer="uniform",input_shape=(26,)))

    model.add(Dense(54))
    model.add(Dropout(0.3))
    model.add(Activation("tanh"))

    model.add(Dense(108))
    model.add(Dropout(0.5))
    model.add(Activation("tanh"))

    model.add(Dense(540))
    model.add(Dropout(0.5))
    model.add(Activation("tanh"))# 高维稀疏矩阵训练


    # 先保证过拟合，保证在train数据集上accuracy较高，之后再考虑过拟合的问题

    #　这种联系不明的东西，一定要维度够高，够稀疏！

    model.add(Dense(54))
    model.add(Dropout(0.5))
    model.add(Activation("tanh"))

    model.add(Dense(27))
    model.add(Dropout(0.3))
    model.add(Activation("tanh"))

    model.add(Dense(2))
    model.add(Activation("softmax"))
    model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])




    try:
        if load_model:
            model.load_weights('my_model_weights.h5', by_name=True)
            print("load weights successful")
    except Exception as a:
        # 如果载入错误，则一定train为true
        print('\033[1;31;40m str(Exception):\t', str(a), "'\033[0m'")
        train = True


    # 去掉group这一列数据，重新训练


    x_train,_, x_test,_ = load_data_set("x1_x2_buf.npz")
    x_train = x_train[:, :26]
    x_test = x_test[:, :26]
    print(x_train.shape)


    if train :


        model.fit(x_train,y_train,batch_size=100000,epochs=100,validation_data=(x_test,y_test)
              )

        score, acc = model.evaluate(x_test, y_test,
                              )
        print('Test score:', score)
        print('Test accuracy:', acc)
        model.save_weights('my_model_weights.h5')
        print("save weights success")

    test = get_test_x()
    print(test,test.shape)

    #predict for x_test
    print("x_test")
    prediction = model.predict(x_test)
    #print("prediction:", prediction[0:100])
    #print(prediction[2000:2100])


    predictions = []


    # 实际的结果就是看到预测为1的太多，通过加上一个shift控制阈值，使其能够更多变为0
    #shift = 0.45
    shift = 0
    for i in prediction:
        if i[0]-shift < i[1] + shift:  # means [0,1]
            predictions.append(0)
        elif i[0] -shift > i[1]+shift :
            predictions.append(1)



    true_prediction = []

    for i in y_test:
        if i[0] < i[1]:  # means [0,1]
            true_prediction.append(0)
        elif i[0] > i[1]:
            true_prediction.append(0.5)# 为了便于区分

    #print("true: ",y_test)


    print("accuracy: ",np.sum(np.equal(predictions,true_prediction))/len(predictions))

    # 自动会进行test数据集对比
    plot_prediction(predictions,true_prediction)




    # 下面展示对于x_train的预测结果，观察过拟合情况
    prediction = model.predict(x_train)

    predictions = []
    shift = 0
    for i in prediction:
        if i[0] - shift < i[1] + shift:
            predictions.append(0)
        elif i[0] - shift > i[1] + shift:
            predictions.append(1)

    true_prediction = []

    for i in y_train:
        if i[0] < i[1]:
            true_prediction.append(0)
        elif i[0] > i[1]:
            true_prediction.append(0.5)
    plot_prediction(predictions, true_prediction)




    #predict for test
    if write:
        print("test")
        prediction = model.predict(test)
        #print("prediction:",prediction[0:100])
        #print(prediction[2000:2100])
        predictions = []
        for i in prediction:
            if i[0] < i[1]:#means [0,1]
                predictions.append(0)
            elif i[0] > i[1]:
                predictions.append(1)
        plot_prediction(predictions)
        write_prediction(predictions)
        write_result(predictions)




# ————————————————————————下面是2017年7月30日进行




# train：是否进行训练，write：是否写入结果，new_model：是否重新训练
model_test(train=True,write=False,load_model=True)
#model_test(train=False,write=False)







# ——————————————————————————————————————下面是2017年7月20日：采用决策树进行预测
from sklearn import tree
clf = tree.DecisionTreeClassifier()


def sklearn_train(method):

    clf = method
    clf.fit(X=x_train,y=y_train)
    prediction = clf.predict(x_test)
    #print("prediction\n",prediction[0:100])
    #print("y_test\n",y_test[0:100])
    accuracy = np.mean(prediction == y_test)
    print("accuracy\n",accuracy)
    answer = clf.predict_proba(x_test)
    #print("answer\n",answer)
    #print(classification_report(y_true=y_test, y_pred=prediction, target_names=['0', '1']))
    #print("average precision:\n", accuracy)
    return prediction,answer,clf


def DTtree_clf(clf,write=False,use_x_test=True):
    train_prediction,answer,clf = sklearn_train(clf)
    if use_x_test:
        prediction = clf.predict(x_test)
    else:
        prediction = clf.predict(get_test_x())
    print(prediction,len(prediction))
    if write:
        sT,eT = write_result(prediction)#将预测结果按照上传格式整理后上传
    return prediction



def plot_true_and_predict(prediction):

    predictions = []

    # 实际的结果就是看到预测为1的太多，通过加上一个shift控制阈值，使其能够更多变为0
    # shift = 0.45
    shift = 0.495
    #shift =1
    for i in prediction:
        if i[0] - shift < i[1] + shift:  # means [0,1]
            predictions.append(0)
        elif i[0] - shift > i[1] + shift:
            predictions.append(1)

    true_prediction = []

    for i in y_test:
        if i[0] < i[1]:  # means [0,1]
            true_prediction.append(0)
        elif i[0] > i[1]:
            true_prediction.append(0.5)  # 为了便于区分

    # print("true: ",y_test)


    print("accuracy: ", np.sum(np.equal(predictions, true_prediction)) / len(predictions))

    plot_prediction(predictions, true_prediction)

#prediction = DTtree_clf(clf)
#plot_true_and_predict(prediction)


# from sklearn import svm
# clf = svm.SVC()
# predicition = DTtree_clf(clf)
# plot_true_and_predict(prediction)


import matplotlib.pyplot as n_plt

# 讨论阈值和accuracy的关系
def discuss_shift_and_accuracy():

    x = []
    y = []
    prediction = DTtree_clf(clf)



    true_prediction = []

    for i in y_test:
        if i[0] < i[1]:  # means [0,1]
            true_prediction.append(0)
        elif i[0] > i[1]:
            true_prediction.append(0.5)

    # 实际的结果就是看到预测为1的太多，通过加上一个shift控制阈值，使其能够更多变为0
    # shift = 0.45
    for i in range(20):
        predictions = []
        shift = i
        x.append(shift)
        for i in prediction:
            if i[0] - shift < i[1] + shift:  # means [0,1]
                predictions.append(0)
            elif i[0] - shift > i[1] + shift:
                predictions.append(1)


          # 为了便于区分
        accuracy = np.sum(np.equal(predictions, true_prediction)) / len(predictions)
        print("accuracy: ", accuracy)
        y.append(accuracy)

    n_plt.plot(x,y,"ro-")
    n_plt.show()

#discuss_shift_and_accuracy()
