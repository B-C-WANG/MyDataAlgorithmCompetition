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
import autosklearn.classification


def load_data_set(file_path):
    tem = np.load(file_path)
    return tem["arr_0"],tem["arr_1"],tem["arr_2"],tem["arr_3"]

x_train, y_train, x_test, y_test = load_data_set("x1_x2_buf.npz")




def get_question():
        return np.load("test_data_buf.npy")




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






