import os
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = [16,10]
plt.rcParams["font.size"] = 16
import seaborn as sns
from mpl_toolkits.axes_grid1 import ImageGrid
from tqdm import tqdm
from keras.models import Model, Sequential
from keras.layers import Input, Flatten, Dense, Conv2D, MaxPooling2D,Dropout
from keras.utils import layer_utils
from keras import backend as K
from keras.optimizers import RMSprop, SGD, Adam
from keras.callbacks import  EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard, CSVLogger
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score, f1_score, confusion_matrix
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.applications import xception
from keras.applications import inception_v3
from keras.applications.vgg16 import  preprocess_input, decode_predictions
from keras.preprocessing.image import ImageDataGenerator
from sklearn.linear_model import LogisticRegression



class VggTrainPSC(object):

    def __init__(self):

        models_dir = os.path.join('models')
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        self.CATEGORIES = ['Black-grass', 'Charlock', 'Cleavers', 'Common Chickweed', 'Common wheat', 'Fat Hen',
                      'Loose Silky-bent',
                      'Maize', 'Scentless Mayweed', 'Shepherds Purse', 'Small-flowered Cranesbill', 'Sugar beet']
        self.NUM_CATEGORIES = len(self.CATEGORIES)
        self.SAMPLE_PER_CATEGORY = 220
        self.SEED = 1987
        self.train_dir = "train"
        self.test_dir = "test"
        self.sample_submission = pd.read_csv('sample_submission.csv')



        self.prepare_train_test_fig_path()
        self.visualization()
        self.generate_validation_split_parameter()



    def visualization(self):
        for category in self.CATEGORIES:
            print('{} {} images'.format(category, len(os.listdir(os.path.join(self.train_dir, category)))))


    def prepare_train_test_fig_path(self):
        '''

        first get all file_path | category_id | category table
        and then choose according to the SAMPLE_PER_CATEGORY

        :return:
        '''
        train = []
        for category_id, category in enumerate(self.CATEGORIES):
            for file in os.listdir(os.path.join(self.train_dir, category)):
                train.append(['train/{}/{}'.format(category, file), category_id, category])
        train = pd.DataFrame(train, columns=['file', 'category_id', 'category'])
        train.head(2)
        print("total train shape",train.shape)
        train = pd.concat([train[train['category'] == c][:self.SAMPLE_PER_CATEGORY] for c in self.CATEGORIES])
        train = train.sample(frac=1)
        train.index = np.arange(len(train))
        train.head(2)
        print("use train shape",train.shape)
        self.train_data = train

        test = []
        for file in os.listdir(self.test_dir):
            test.append(['test/{}'.format(file), file])
        test = pd.DataFrame(test, columns=['filepath', 'file'])
        test.head(2)
        print(test.shape)
        self.test_data = test

    @staticmethod
    def read_image(filepath,size):
        img = image.load_img(filepath, target_size=size)
        img = image.img_to_array(img)
        return img
    def give_example_images(self):
        fig = plt.figure(1, figsize=(self.NUM_CATEGORIES, self.NUM_CATEGORIES))
        grid = ImageGrid(fig, 111, nrows_ncols=(self.NUM_CATEGORIES, self.NUM_CATEGORIES), axes_pad=0.05)
        i = 0
        for category_id, category in enumerate(self.CATEGORIES):
            for filepath in self.train_data[self.train_data['category'] == category]['file'].values[:self.NUM_CATEGORIES]:
                ax = grid[i]
                img = self.read_image(filepath, (224, 224))
                ax.imshow(img / 255.)
                ax.axis('off')
                if i % self.NUM_CATEGORIES == self.NUM_CATEGORIES - 1:
                    ax.text(250, 112, filepath.split('/')[1], verticalalignment='center')
                i += 1
        plt.show()

    def generate_validation_split_parameter(self):
        np.random.seed(seed=self.SEED)
        rnd = np.random.random(len(self.train_data))
        train_idx = rnd < 0.8
        valid_idx = rnd >= 0.8
        ytr = self.train_data.loc[train_idx, 'category_id'].values
        yv = self.train_data.loc[valid_idx, 'category_id'].values
        self.train_idx = train_idx
        self.valid_idx = valid_idx
        self.ytr = ytr  # true y
        self.yv = yv    # validation y
        print(len(ytr),len(yv))

    def Vgg_bottleneck_feature_extraction_then_LR(self):
        INPUT_SIZE = 224
        POOLING = 'avg'
        x_train = np.zeros((len(self.train_data), INPUT_SIZE, INPUT_SIZE, 3), dtype='float32')
        for i, file in tqdm(enumerate(self.train_data['file'])):
            img = self.read_image(file, (INPUT_SIZE, INPUT_SIZE))
            x = preprocess_input(np.expand_dims(img.copy(), axis=0))
            x_train[i] = x
        print('Train Images shape: {} size: {:,}'.format(x_train.shape, x_train.size))

        Xtr = x_train[self.train_idx]
        Xv = x_train[self.valid_idx]
        print((Xtr.shape, Xv.shape, self.ytr.shape, self.yv.shape))
        vgg_bottleneck = VGG16(weights='imagenet', include_top=False, pooling=POOLING)
        train_vgg_bf = vgg_bottleneck.predict(Xtr, batch_size=32, verbose=1)
        valid_vgg_bf = vgg_bottleneck.predict(Xv, batch_size=32, verbose=1)
        print('VGG train bottleneck features shape: {} size: {:,}'.format(train_vgg_bf.shape, train_vgg_bf.size))
        print('VGG valid bottleneck features shape: {} size: {:,}'.format(valid_vgg_bf.shape, valid_vgg_bf.size))


        '''
        the output of vgg or xception bottleneck is a vector
        and we can plot the vector of different sample
        and see the Distribution of Sample Features
           
        
        
        '''




        logreg = LogisticRegression(multi_class='multinomial', solver='lbfgs', random_state=self.SEED)
        # after directly get output from 512, without Dense train, just use LR train
        logreg.fit(train_vgg_bf, self.ytr)
        valid_probs = logreg.predict_proba(valid_vgg_bf)
        valid_preds = logreg.predict(valid_vgg_bf)

        print('Validation VGG Accuracy {}'.format(accuracy_score(self.yv, valid_preds)))
        self.valid_preds = valid_preds
        self.logreg = logreg
        self.INPUT_SIZE = INPUT_SIZE


    def Xception_feature_extraction_then_LR(self):
        INPUT_SIZE = 299
        POOLING = 'avg'
        x_train = np.zeros((len(self.train_data), INPUT_SIZE, INPUT_SIZE, 3), dtype='float32')
        for i, file in tqdm(enumerate(self.train_data['file'])):
            img = self.read_image(file, (INPUT_SIZE, INPUT_SIZE))
            x = xception.preprocess_input(np.expand_dims(img.copy(), axis=0))
            x_train[i] = x
        print('Train Images shape: {} size: {:,}'.format(x_train.shape, x_train.size))

        Xtr = x_train[self.train_idx]
        Xv = x_train[self.valid_idx]
        print((Xtr.shape, Xv.shape, self.ytr.shape, self.yv.shape))
        xception_bottleneck = xception.Xception(weights='imagenet', include_top=False, pooling=POOLING)
        train_x_bf = xception_bottleneck.predict(Xtr, batch_size=32, verbose=1)
        valid_x_bf = xception_bottleneck.predict(Xv, batch_size=32, verbose=1)
        print('Xception train bottleneck features shape: {} size: {:,}'.format(train_x_bf.shape, train_x_bf.size))
        print('Xception valid bottleneck features shape: {} size: {:,}'.format(valid_x_bf.shape, valid_x_bf.size))

        logreg = LogisticRegression(multi_class='multinomial', solver='lbfgs', random_state=self.SEED)
        logreg.fit(train_x_bf, self.ytr)
        valid_probs = logreg.predict_proba(valid_x_bf)
        valid_preds = logreg.predict(valid_x_bf)
        print('Validation VGG Accuracy {}'.format(accuracy_score(self.yv, valid_preds)))
        self.valid_preds = valid_preds
        self.logreg = logreg
        self.INPUT_SIZE = INPUT_SIZE
        self.xception_bottleneck = xception_bottleneck

    def confusion_matrix_plot(self):
        cnf_matrix = confusion_matrix(self.yv,self.valid_preds)
        abbreviation = ['BG', 'Ch', 'Cl', 'CC', 'CW', 'FH', 'LSB', 'M', 'SM', 'SP', 'SFC', 'SB']
        pd.DataFrame({'class':self.CATEGORIES, 'abbreviation': abbreviation})
        fig, ax = plt.subplots(1)
        ax = sns.heatmap(cnf_matrix, ax=ax, cmap=plt.cm.Greens, annot=True)
        ax.set_xticklabels(abbreviation)
        ax.set_yticklabels(abbreviation)
        plt.title('Confusion Matrix')
        plt.ylabel('True class')
        plt.xlabel('Predicted class')
        plt.show()

    def create_submission(self):
        x_test = np.zeros((len(self.test_data), self.INPUT_SIZE, self.INPUT_SIZE, 3), dtype='float32')
        for i, filepath in tqdm(enumerate(self.test_data['filepath'])):
            img = self.read_image(filepath, (self.INPUT_SIZE, self.INPUT_SIZE))
            x = xception.preprocess_input(np.expand_dims(img.copy(), axis=0))
            x_test[i] = x
        print('test Images shape: {} size: {:,}'.format(x_test.shape, x_test.size))
        test_x_bf = self.xception_bottleneck.predict(x_test, batch_size=32, verbose=1)
        print('Xception test bottleneck features shape: {} size: {:,}'.format(test_x_bf.shape, test_x_bf.size))
        test_preds = self.logreg.predict(test_x_bf)
        self.test_data['category_id'] = test_preds
        self.test_data['species'] = [self.CATEGORIES[c] for c in test_preds]
        self.test_data[['file', 'species']].to_csv('submission.csv', index=False)


if __name__ == '__main__':
    psc = VggTrainPSC()
    #psc.Vgg_feature_extraction_then_LR()
    psc.Xception_feature_extraction_then_LR()
    #psc.confusion_matrix_plot()
    psc.create_submission()