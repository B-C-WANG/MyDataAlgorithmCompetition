import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = [16,10]
plt.rcParams["font.size"] = 16
from tqdm import tqdm
import pickle
from keras.preprocessing import image
import cv2

class MaskDataFeed(object):
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
        self.train_dir = "green_mask\\train"
        self.test_dir = "green_mask_test\\test"
        self.sample_submission = pd.read_csv('sample_submission.csv')
        self.INPUT_SIZE = 300

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
                train.append(['green_mask\\train/{}/{}'.format(category, file), category_id, category])
        train = pd.DataFrame(train, columns=['filepath', 'category_id', 'category'])
        print(train.head(2))
        print("total train shape",train.shape)
        train = pd.concat([train[train['category'] == c][:self.SAMPLE_PER_CATEGORY] for c in self.CATEGORIES])
        train = train.sample(frac=1)
        train.index = np.arange(len(train))
        train.head(2)
        print("use train shape",train.shape)
        self.train_data = train
        test = []
        for file in os.listdir(self.test_dir):
            test.append(['green_mask_test\\test/{}'.format(file), file])
        test = pd.DataFrame(test, columns=['filepath', 'file'])
        test.head(2)
        print(test.shape)
        self.test_data = test

    @staticmethod
    def read_image(filepath,size):
        img = image.load_img(filepath, target_size=size)
        img = image.img_to_array(img)
        return img

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

    def generator_mask_figs(self):
        for i, filepath in tqdm(enumerate(self.test_data['filepath'])):
            debug=0
            img = self.read_image(filepath, (self.INPUT_SIZE, self.INPUT_SIZE))
            if debug:
                img = self.read_image("C:\\Users\Administrator\Desktop\KaggleCompetitionStorage\PlantSeedlingsClassification\\train\Black-grass\\775735fb9.png", (self.INPUT_SIZE, self.INPUT_SIZE))
            # kill Red and Blue
            img[img[:,:,0]>img[:,:,1]] = (0,0,0)
            img[img[:,:,2]>img[:,:,1]] = (0,0,0)
            # kill white: r+b > 1.6*green
            img[(img[:,:,0]+img[:,:,2])>img[:,:,1]*1.6] = (0,0,0)
            if debug:
                plt.imshow(img/255)
                plt.show()
                exit()
            cv2.imwrite("green_mask_test/"+filepath,img)

    def prepare_mask_rcnn_data(self):
        def preprocess_input(x):
            x /= 255.
            x -= 0.5
            x *= 2.
            return x
        '''
        mask rcnn data is {"train":{
        "<label>": [image_array,mask_array]}
        "test":{"label":[...]}
        }
        }

        '''
        mask_rcnn_data = {}

        x_train = np.zeros((len(self.train_data), self.INPUT_SIZE, self.INPUT_SIZE, 3), dtype='float32')
        for i, file in tqdm(enumerate(self.train_data['filepath'])):
            img = self.read_image(file, (self.INPUT_SIZE, self.INPUT_SIZE))
            x = preprocess_input(np.expand_dims(img.copy(), axis=0))
            x_train[i] = x
        print('Train Images shape: {} size: {:,}'.format(x_train.shape, x_train.size))

        Xtr = x_train[self.train_idx]
        Xv = x_train[self.valid_idx]
        print("all data shape is: ",(Xtr.shape, Xv.shape, self.ytr.shape, self.yv.shape))
        x_test = np.zeros((len(self.test_data), self.INPUT_SIZE, self.INPUT_SIZE, 3), dtype='float32')

        for i, filepath in tqdm(enumerate(self.test_data['filepath'])):
            img = self.read_image(filepath, (self.INPUT_SIZE, self.INPUT_SIZE))
            x = preprocess_input(np.expand_dims(img.copy(), axis=0))
            x_test[i] = x

        mask_rcnn_data["X_train"] = Xtr
        mask_rcnn_data["X_valid"] = Xv
        mask_rcnn_data["y_train"] = self.ytr
        mask_rcnn_data["y_valid"] = self.yv
        mask_rcnn_data["X_test"] = x_test
        with open("mask_rcnn_data.pkl","wb") as f:
            pickle.dump(mask_rcnn_data,f)

if __name__ == '__main__':
    psc = MaskDataFeed()
    psc.prepare_mask_rcnn_data()
    #psc.generator_mask_figs()
