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
import copy
import visualize

class MaskDataFeed(object):
    def __init__(self):

        models_dir = os.path.join('models')
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        self.CATEGORIES = ['Black-grass', 'Charlock', 'Cleavers', 'Common Chickweed', 'Common wheat', 'Fat Hen',
                      'Loose Silky-bent',
                      'Maize', 'Scentless Mayweed', 'Shepherds Purse', 'Small-flowered Cranesbill', 'Sugar beet']
        self.NUM_CATEGORIES = len(self.CATEGORIES)
        self.SAMPLE_PER_CATEGORY = 20
        self.SEED = 1987

        self.sample_submission = pd.read_csv('sample_submission.csv')
        self.INPUT_SIZE = 320# image size must set to 256, 320, 384, 448, 512, ... that's for RCNN



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
            for file in sorted(os.listdir(os.path.join(self.train_dir, category))):


                train.append(['{}/{}/{}'.format(self.train_dir,category, file), category_id, category])
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
        for file in sorted(os.listdir(self.test_dir)):
            test.append(['{}/{}'.format(self.test_dir,file), file])
        test = pd.DataFrame(test, columns=['filepath', 'file'])
        test.head(2)
        print(test.shape)
        self.test_data = test

    def prepare_train_fig_path(self):
        train = []
        for category_id, category in enumerate(self.CATEGORIES):
            for file in sorted(os.listdir(os.path.join(self.train_dir, category))):


                train.append(['{}/{}/{}'.format(self.train_dir,category, file), category_id, category])
        train = pd.DataFrame(train, columns=['filepath', 'category_id', 'category'])
        print(train.head(2))
        print("total train shape", train.shape)
        train = pd.concat([train[train['category'] == c][:self.SAMPLE_PER_CATEGORY] for c in self.CATEGORIES])
        train = train.sample(frac=1)
        train.index = np.arange(len(train))
        train.head(2)
        print("use train shape", train.shape)
        self.train_data = train



    @staticmethod
    def read_image(filepath,size):
        img = image.load_img(filepath, target_size=size)
        img = image.img_to_array(img)
        return img

    def generator_mask_figs_for_train(self):
        self.train_dir = "train"
        self.prepare_train_fig_path()

        # TODO: use all the file, do not set self.SAMPLE_PER_CATEGORY

        for i, filepath in tqdm(enumerate(self.train_data['filepath'])):
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
            cv2.imwrite("green_mask_train/"+filepath,img)

    def prepare_normal_image_data(self):
        self.train_dir = "train"
        self.test_dir = "test"
        self.prepare_train_test_fig_path()
        def preprocess_input(x):
            x /= 255.
            x -= 0.5
            x *= 2.
            return x

        train_data = {}
        test_data = {}

        x_train = np.zeros((len(self.train_data), self.INPUT_SIZE, self.INPUT_SIZE, 3), dtype='float32')

        for i, file in tqdm(enumerate(self.train_data['filepath'])):
            img = self.read_image(file, (self.INPUT_SIZE, self.INPUT_SIZE))
            x = np.expand_dims(img.copy(), axis=0)
            x_train[i] = x

        x_test = np.zeros((len(self.test_data), self.INPUT_SIZE, self.INPUT_SIZE, 3), dtype='float32')
        for i, filepath in tqdm(enumerate(self.test_data['filepath'])):
            img = self.read_image(filepath, (self.INPUT_SIZE, self.INPUT_SIZE))
            x = np.expand_dims(img.copy(), axis=0)
            x_test[i] = x

        test_file_name = np.array(self.test_data["filepath"])


        for i in range(x_train.shape[0]):

            train_data[str(i)] = x_train[i]


        for i in range(x_test.shape[0]):
            test_data[str(i)] = [x_test[i],test_file_name[i]]

        with open("mask_rcnn_data_train.pkl", "wb") as f:
            pickle.dump(train_data, f)
        print("train_data[i]: image ")

        with open("mask_rcnn_data_test.pkl", "wb") as f:
            pickle.dump(test_data, f)
        print("train_test[i]: image, file_path(which can get the filename by spliting '/')")



    def prepare_mask_rcnn_data(self,debug=0):

        self.train_dir = "green_mask\\train"
        self.prepare_train_fig_path()
        self.visualization()

        def preprocess_input(x):
            x /= 255.
            x -= 0.5
            x *= 2.
            return x

        mask_rcnn_data = {}

        x_train = np.zeros((len(self.train_data), self.INPUT_SIZE, self.INPUT_SIZE, 3), dtype='float32')
        for i, file in tqdm(enumerate(self.train_data['filepath'])):
            img = self.read_image(file, (self.INPUT_SIZE, self.INPUT_SIZE))
            x = np.expand_dims(img.copy(), axis=0)
            x_train[i] = x
        print('Train Images shape: {} size: {:,}'.format(x_train.shape, x_train.size))


        mask = np.zeros((len(self.train_data), self.INPUT_SIZE, self.INPUT_SIZE, 1), dtype='float32')

        mask_image = np.sum(x_train,axis=3).reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2],1)
        print(mask_image.shape)
        for i in range(mask_image.shape[0]):
            _mask =  np.array((mask_image[i][:,:,:]!=0)).astype("int")
            mask[i] = np.expand_dims(_mask.copy(), axis=0)

        print(mask[0].shape)
        assert mask[0].shape[2] == 1
        if debug:
            plt.imshow(mask[0].reshape(mask.shape[1],mask.shape[2])*255)
            plt.show()

        self.label = np.array(self.train_data['category_id'])
        print(self.label,self.label.shape)

        mask_data = {}

        for i in range(mask.shape[0]):
            mask_data[str(i)] = [mask[i],[self.label[i]]]

        with open("mask_rcnn_data_mask.pkl","wb") as f:
            pickle.dump(mask_data,f)
        print("mask_data[i]: [mask image, label_list]")

    def prepare_total_data(self):
        self.train_dir = "train"
        self.test_dir = "test"
        self.prepare_train_test_fig_path()

        def preprocess_input(x):
            x /= 255.
            x -= 0.5
            x *= 2.
            return x

        train_data = {}
        test_data = {}
        mask_data = {}

        x_train = []
        mask = []

        for i, file in tqdm(enumerate(self.train_data['filepath'])):
            img = self.read_image(file, (self.INPUT_SIZE, self.INPUT_SIZE))
            #x = np.expand_dims(img.copy(), axis=0)
            x_train.append(copy.deepcopy(img))

            img[img[:, :, 0] > img[:, :, 1]] = (0, 0, 0)
            img[img[:, :, 2] > img[:, :, 1]] = (0, 0, 0)
            # kill white: r+b > 1.6*green
            img[(img[:, :, 0] + img[:, :, 2]) > img[:, :, 1] * 1.6] = (0, 0, 0)

            mask_image = np.sum(img, axis=2).reshape( img.shape[0], img.shape[1], 1)


            _mask = np.array((mask_image[:, :, :] != 0)).astype("int")
            #_mask = np.expand_dims(_mask.copy(), axis=0)
            mask.append([copy.deepcopy(_mask), [self.train_data.loc[i,"category_id"]]])
            #print(self.train_data.loc[i,:])


            # i = 1
            # for _ in range(4):
            #     plt.subplot(4, 2, i)
            #
            #     plt.axis('off')
            #     x1 = x_train[0]
            #     print(x1)
            #     plt.imshow(x1.reshape(x1.shape[0], x1.shape[1], x1.shape[2]).astype(np.uint8))
            #     i += 1
            #     plt.subplot(4, 2, i)
            #     _mask1 = mask[0]
            #     plt.imshow(_mask1.reshape(_mask1.shape[0], _mask1.shape[1]).astype(np.uint8))
            #     i += 1
            # plt.show()
            # exit()


        # TODO: last debug here



        x_test = []

        for i, filepath in tqdm(enumerate(self.test_data['filepath'])):
            t_img = self.read_image(filepath, (self.INPUT_SIZE, self.INPUT_SIZE))
            x_test.append([copy.deepcopy(t_img), self.test_data.loc[i,"filepath"] ])

        #test_file_name = np.array(self.test_data["filepath"])
        #
        # for i in range(len(x_train)):
        #     train_data[str(i)] = x_train[i]
        #
        # for i in range(len(x_test)):
        #     test_data[str(i)] = [x_test[i], test_file_name[i]]
        #
        # for i in range(len(mask)):
        #     mask_data[str(i)] = [mask[i], [self.label[i]]]


        #visualize.display_top_masks(x_train[3], mask[3][0],self.CATEGORIES , [self.CATEGORIES[mask[3][1][0]]])

        save = 1

        if save:
            with open("mask_rcnn_data_mask.pkl", "wb") as f:
                pickle.dump(mask, f)
            print("mask_data[i]: [mask image, label_list]")

            with open("mask_rcnn_data_train.pkl", "wb") as f:
                pickle.dump(x_train, f)
            print("train_data[i]: image ")

            with open("mask_rcnn_data_test.pkl", "wb") as f:
                pickle.dump(x_test, f)


        print("train_test[i]: image, file_path(which can get the filename by spliting '/')")


if __name__ == '__main__':
    psc = MaskDataFeed()

    # 1. prepare for mask figs, first need to set the test and train dir to the image
    #psc.generator_mask_figs_for_train()

    # 2. after that, set the test and train dit to the masked image
    # then generate the mask and figs for maskRCNN, which need the input of shape:
    #psc.prepare_mask_rcnn_data()

    # 3. and generate normal image for mask RCNN
    #psc.prepare_normal_image_data()

    psc.prepare_total_data()