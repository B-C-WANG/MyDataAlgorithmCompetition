import pandas as pd
import numpy as np
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier,BaggingClassifier,GradientBoostingClassifier
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
import tqdm
import matplotlib.pyplot as new_plt

def load_data_set(file_path):
    tem = np.load(file_path)
    return tem["arr_0"],tem["arr_1"],tem["arr_2"],tem["arr_3"]

x_train, y_train, x_test, y_test = load_data_set("x1_x2_buf.npz")




def get_question():
        return np.load("test_data_buf.npy")

# sklearn内置了VotingClassifier，可以输入几个clf，但是这样灵活性不强，故自己封装了类
class EnsembleLearningClassify:

    @staticmethod
    def plot_prediction(prediction, true_prediction=None):
        if true_prediction is None:
            index = [i for i in range(len(prediction))]
            new_plt.plot(index, prediction)
            new_plt.savefig("plot_prediction_NN.png", dpi=300)
            new_plt.show()
        else:
            index = [i for i in range(len(prediction))]
            new_plt.plot(index, prediction, color="b")
            new_plt.plot(index, true_prediction, color="r")
            new_plt.savefig("plot_prediction_NN.png", dpi=300)
            new_plt.show()

    @staticmethod
    def write_result(prediction):
        startT = []
        endT = []
        now = prediction[0]

        for i in range(len(prediction)):
            if prediction[i] != now:
                if prediction[i] == 1:
                    startT.append(i)
                if prediction[i] == 0:
                    endT.append(i - 1)
                now = prediction[i]

        print(startT, "\n", endT)
        print(len(startT), len(endT))

        file = open("test1_08_results.csv", "w")

        str_ = ""
        for i in range(len(startT)):
            str_ = str_ + str(startT[i]) + "," + str(endT[i]) + "\n"
        # print(str_)
        file.write("startTime,endTime\n" + str_)
        file.close()
        return startT, endT



    models = []
    results = []

    @classmethod
    def init(cls,x_train,y_train):
        cls.x_train = x_train
        cls.y_train = y_train



    @classmethod
    def build_models(cls,models_list):
        '''
        build multiply models
        :param models_list:
        :return:
        '''
        for i in range(len(models_list)):
            cls.models.append(models_list[i])

    @classmethod
    def train_save_and_test(cls,x_test,y_test):
        '''
        train all models and save models, then save results

        :param x_test:
        :return:
        '''
        cls.results = []
        for i in tqdm.trange(len(cls.models)):

                model = cls.models[i]
                model.fit(cls.x_train,cls.y_train)
                joblib.dump(model,"model/EnsembleLearning_model_{}.m".format(i))
                if x_test is not None and y_test is not None:
                    prediction = model.predict(x_test)
                    cls.results.append(prediction)
                    print("accuracy of the model {}: ".format(i), np.sum(np.equal(prediction, y_test)) / len(prediction))

    @classmethod
    def load_and_test(cls,x_test,model_number):
        '''
        load models and make prediction

        :param x_test:
        :return:
        '''
        cls.models = []
        cls.results = []
        for i in range(model_number):

                model = joblib.load("model/EnsembleLearning_model_{}.m".format(i))
                cls.models.append(model)
                cls.results.append(model.predict(x_test))

    @classmethod
    def load(cls,model_number):
        cls.models = []
        for i in range(model_number):

                model = joblib.load("model/EnsembleLearning_model_{}.m".format(i))
                cls.models.append(model)


    @classmethod
    def voting_and_show(cls,gate,y_test):
        '''
        :param gate:
        :param y_test:
        :return:
        '''
        result = np.array(cls.results)

        result = np.sum(result,axis=0)

        prediction = result>(gate*result.shape[0])
        print("accuracy of the model ", np.sum(np.equal(prediction, y_test)) / prediction.shape[0])
        cls.plot_prediction(prediction, y_test)

    @classmethod
    def write_answer(cls,question):
            prediction = clf.predict(question)
            print("prediction:", prediction[0:100])
            print(prediction[2000:2100])
            plot_prediction(prediction)
            write_result(prediction)

    @classmethod
    def show_one_model_results(cls,index,x_test,y_test):
        model = joblib.load("model/EnsembleLearning_model_{}.m".format(index))
        prediction = model.predict(x_test)
        cls.plot_prediction(prediction, y_test)



# 训练
def ensemble_train():
    clf1 = RandomForestClassifier()
    clf2 = AdaBoostClassifier(n_estimators=100)
    clf3 = BaggingClassifier()
    clf4 = tree.DecisionTreeClassifier()
    clf5 = GradientBoostingClassifier()
    clf6 = GaussianNB()
    clf7 = LogisticRegression()


    clf = EnsembleLearningClassify()
    clf.init(x_train=x_train,y_train=y_train)
    clf.build_models([clf1,clf2,clf3,clf4,clf5,clf6,clf7])
    clf.train_save_and_test(x_test,y_test)
# 测试
def test():
    clf  = EnsembleLearningClassify()
    clf.load_and_test(x_test,model_number=7)

    clf.voting_and_show(gate=0.2,y_test=y_test)

#ensemble_train()

test()

