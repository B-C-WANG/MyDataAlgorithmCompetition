import random
import numpy as np
import pandas as pd

# 4-2 划分x_train,y_train,x_test和y_test,传参包括测试集比例，正例(label=1)和反例的比例，采用随机采样得到反例，正例全部包括,#原始数据正例:反例为0.0622

from sklearn.utils import shuffle

# 完成封装
def data_set_split(file_path,test_ratio,label_ratio,random_seed):
    if random_seed:
        random.seed(random_seed)

    data = pd.read_csv(file_path)
    set1 = data[data.icol(27) >= 0.5]#正例
    set2 = data[data.icol(27) <= 0.5]
    sample_numer = label_ratio * set1.shape[0]

    random_ = random.sample(list(range(set2.shape[0])), sample_numer)

    result = []
    for i in range(len(random_)):
        result.append(set2[random_[i]:random_[i] + 1])
    result.append(set1)
    set = pd.concat(result)

    set = shuffle(set)#打乱行数据

    try:
        label = "1.000000000000000000e+00"
        train = np.array(set.drop(labels=label,axis=1))
        label = np.array(set[label])
    except:
        label = "0.000000000000000000e+00"
        train = np.array(set.drop(labels=label, axis=1))
        label = np.array(set[label])

    total = train.shape[0]

    test_number =  int(total * test_ratio)
    train_x = train[test_number:]
    train_y = label[test_number:]
    test_x  = train[0:test_number]
    test_y  = label[0:test_number]
    print("train_x",train_x.shape,"train_y",train_y.shape,\
          "test_x",test_x.shape,"test_y",test_y.shape)
    return train_x, train_y, test_x, test_y

# 完成封装
# 4-3 将划分后的数据集缓存到二进制文件中，加快读取，便于测试
def save_data_set(x_train,y_train,x_test,y_test):
    np.savez("buf.npz",x_train,y_train,x_test,y_test)
def load_data_set(file_path):
    tem = np.load(file_path)
    return tem["arr_0"],tem["arr_1"],tem["arr_2"],tem["arr_3"]






# x_train,y_train,x_test,y_test=data_set_split(file_path="train_set_normalization.csv",
#                test_ratio=0.5,
#                label_ratio=1,
#                random_seed=1)
#
# save_data_set(x_train,y_train,x_test,y_test)

# 4-4 测试读取文件

x_train,y_train,x_test,y_test = load_data_set("buf.npz")
# 可能出现错误，打印shape数据检验
print("train_x",x_train.shape,"train_y",y_train.shape,\
          "test_x",x_test.shape,"test_y",y_test.shape)

# 5 模型预测和调参


# 模型预测相关：对测试集进行预测，上传结果

TEST_PATH = "08_data.csv"
from sklearn import preprocessing
# 读取数据，正则化，存储在numpy文件中
# 完成封装
def save_test_x():
    data = pd.read_csv(TEST_PATH)
    train = data.drop(labels="time",axis=1)
    train = np.array(train)
    new_data = []
    out = ""
    for i in range(train.shape[1]):

        new_data.append(preprocessing.scale(np.array(train[:,i]))) # 对于每一列，将指标正则化
    # 加上最后一个label指标
    test_data = np.array(new_data)
    test_data = np.transpose(test_data)
    print("test_data:",test_data,test_data.shape)
    np.save("test_data_buf", test_data)
    return 1

#save_test_x()
# 完成封装
def get_test_x():
    return np.load("test_data_buf.npy")



# 进行预测之后，得到prediction，然后写入数据到预测后的文件中进行观察
# 完成封装
def write_prediction(predictions):
    data = pd.read_csv(TEST_PATH)
    prediction = pd.Series(predictions)
    print("prediction\n",prediction)
    data.insert(data.shape[1], "predictions", predictions)
    data.to_csv(path_or_buf="test_with_prediction.csv",index = False)
    return 1
# 因为时间就是数据的index+1，所以可以直接用一列prediction来写结果
#注意是预测结冰，也就是label==！
# 完成封装
def write_result(prediction):
    startT = []
    endT = []
    now = prediction[0]


    for i in range(len(prediction)):#如果不等，替换now，end写入上一个序列，start写入这一个，相等不操作
        if prediction[i] != now :#当出现转折
            if prediction[i] ==1:#如果转折后为1，加上startT
                startT.append(i)
            if prediction[i]==0:#如果转折后为0，加上endT
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












# 这里假设指标和label一一对应，不考虑历史，不采用具有考虑历史因素的算法如RNN，如果考虑，需要对数据集排序

# 5-1 预测和总结

import matplotlib.pyplot as plt
# 完成封装
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

# ROC 曲线绘制函数
# 完成封装
def draw_ROC(fig_save_file,y_test,prediction_proba):
    answer_prob = prediction_proba[:, 1]

    fpr, tpr, thresholds = roc_curve(y_true=y_test,y_score=answer_prob)

    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
    mean_auc = auc(fpr, tpr)  # 计算平均AUC值
    plt.plot(fpr, tpr, 'k--',
             label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.savefig(fig_save_file,dpi=300)
    plt.show()


# 5-1 逻辑回归分类



from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve,auc
from sklearn.metrics import classification_report

#clf = LogisticRegression()
#prediction,answer,clf = sklearn_train(clf)
#draw_ROC(fig_save_file="ROC_of_LR_result.png",y_test=y_test,prediction_proba=answer)

# 进行预测：采用了0.75,0.5和0.25的数据集，结果正确率停留在0.85左右，对于test的预测，全是0，存在问题
#print(clf.predict(get_test_x()))
#write_prediction(clf.predict(get_test_x()))


# 5-2 决策树分类，在train和test数据集1:1的情况下，能够实现99.59%的准确率

from sklearn import tree
clf = tree.DecisionTreeClassifier()


#draw_ROC(fig_save_file="ROC_of_DecisionTree.png",y_test=y_test,prediction_proba=answer)


# 决策树的绘制:安装graphviz，配置环境变量，然后pip pydot

from sklearn.externals.six import StringIO
dot_data = StringIO()
import pydot
import os

#内部设置环境变量
os.environ["PATH"] += os.pathsep + 'G:\\Program Files\\Graphviz2.38\\bin'
# 完成封装
def plot_decision_tree_0(clf):
    tree.export_graphviz(clf, out_file=dot_data)#采用IO的方式传递二进制，然后给pydot处理
    graph = pydot.graph_from_dot_data(dot_data.getvalue())
    graph[0].write_png('DT_of_data.png')

#plot_decision_tree_0(clf)
#从plot出来的图来看，分支太多，可能需要调参剪枝


#按照3分法对数据进行聚合：首先将startT和endT连成串，然后4个4个分开，形成/4个组，每个组中4个元素，分别后减去前，得到3个距离，d1,d2,d3
#如果d1大于d2+d3,那么将前面两个聚合，添加上新的startT和endT，并将标签相应地append(1),append(0)(因为一开始是1开始，然后是0)
#如果d3>d1+d2,就是后面两个聚合，其他不变，如果中间的d2>d1+d3，就不变，标签就分别append1,0,1三个
#更新：这个方法被废弃了，可以通过作图观察+去噪点的方法实现
# 另外也应该能够找到更合适更简单的算法
# 完成封装
def result_combine_abandon(time_line_for_label_1):
    endT,startT = time_line_for_label_1#替换start和end的位置
    endT = None

    print(startT,"\n",endT,"\n",len(startT),len(endT))
    addition = len(startT)%4 #找到不整除的多余项
    print(addition)
    a_sT = startT[-1-addition:-1]# 多出来的sT和eT，最后聚合完之后加上
    a_eT = endT[-1-addition:-1]
    group_num = int((len(startT)-addition)/4)



#对预测结果进行作图，看能不能够分离出簇，或者x作为index，y作为label，进行聚类算法

import matplotlib.pyplot as new_plt
# 完成封装





#注意每一次运行的结果都不一样！
def DTtree_clf(clf):
    train_prediction,answer,clf = sklearn_train(clf)
    prediction = clf.predict(get_test_x())
    print(prediction,len(prediction))
    sT,eT = write_result(prediction)#将预测结果按照上传格式整理后上传
    return prediction

#尝试计算20次，然后取label1的并集（每个index有1则1），看能不能连成簇，这是为了减少随机因素导致的label=0，毕竟在实际情况，label=0和1交替时，label被认为1的可能性更高
def mul_DTtree_clf(clf,times):
    class_num = []
    train_prediction, answer, clf = sklearn_train(clf)
    prediction = clf.predict(get_test_x())
    class_num.append(get_class_num(prediction))#用来画出每次经过取并集之后得到的class数目
    for i in range(times-1):
        train_prediction, answer, clf = sklearn_train(clf)
        prediction_ = clf.predict(get_test_x())
        for j in range(len(prediction_)):
            if prediction_[j] ==1 or prediction[j] ==1:
                prediction[j] =1
        class_num.append(get_class_num(prediction))
    return prediction,class_num

# 这个函数与write_result相同，给出prediction返回startT，endT，此时我们只要len(startT)
def get_class_num(prediction):
                startT = []
                endT = []
                now = prediction[0]
                for i in range(len(prediction)):  # 如果不等，替换now，end写入上一个序列，start写入这一个，相等不操作
                    if prediction[i] != now:  # 当出现转折
                        if prediction[i] == 1:  # 如果转折后为1，加上startT
                            startT.append(i)
                        if prediction[i] == 0:  # 如果转折后为0，加上endT
                            endT.append(i - 1)
                        now = prediction[i]
                print(startT)
                return len(startT)



import matplotlib.pyplot as plot2
def run__():
    prediction ,class_num= mul_DTtree_clf(clf,500)
    plot_prediction(prediction)
    write_result(prediction)
    plot2.plot([i for i in range(len(class_num))],class_num)
    plot2.savefig("class_to_predict_time.png",dpi=300)
    plot2.show()


#结果：对于20次预测和取并集，得到的“条形码”有了聚成一团的情况，但是反而class数目更多了，这表明离散的没有连起来的很多，需要增加预测和取并集的
#次数，这个需要根据class_to_predict_time来判断什么时候会下降
# 结果发现，迭代200次，都只能最后是保持class在一个较高值，最终决定：迭代500次，然后写入结果文件中，提交！

#write_prediction(clf.predict(get_test_x()))







#进行结果的聚合和验证，总的来说0和1一定是连在一起的，而这个预测是将每个分离开来的，所以要进行聚合
#聚合完成之后， 进行一次error分析，看有多少比例与聚合前不同，现在2017年7月13日用的没有聚合前的数据，正确率57%
# 2017年7月14日：为什么会出现0101交替？是因为系统逐渐在0和1之间变化，非常反复，所以作图能够反映问题



#新方法：像NN_predict一样，采用全部的一个文件作为train，全部的一个文件作为test，不进行数据打乱和平等划分





def load_data_set(file_path):
    tem = np.load(file_path)
    return tem["arr_0"],tem["arr_1"],tem["arr_2"],tem["arr_3"]



x_train, y_train, x_test, y_test = load_data_set("x1_x2_buf.npz")
print(x_train,x_train.shape, '\n',x_test,x_test.shape)
print(y_train,y_train.shape,'\n',y_test,y_test.shape)

#DTtree_clf(clf)

#结果：如果采用决策树，只有85%上下的准确率，训练数据集大了，反而影响准确度，但是可能会避免过拟合
#另外一点，之前采用的是两个文件混合的方式，也就是说test中的一部分会有train，而现在test中没有train的数据，
#准确率就降低了，这样的话，NN效果更好