import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import pickle
from config import Config
import utils
import model as modellib
import visualize
from model import log

# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Path to COCO trained weights
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")


class SeedlingConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "seedling"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 8

    # Number of classes (including background)
    NUM_CLASSES = 1 + 12  # background + 3 shapes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 128
    IMAGE_MAX_DIM = 128

    # Use smaller anchors because our image and objects are small
    #RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    RPN_ANCHOR_SCALES = (8, 16,32, 64, 128)

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5


config = SeedlingConfig()
config.display()


def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax


train_val_split_ratio = 0.8

class SeedlingTrain(utils.Dataset):
    """Generates the shapes synthetic dataset. The dataset consists of simple
    shapes (triangles, squares, circles) placed randomly on a blank surface.
    The images are generated on the fly. No file access required.
    """
    def __init__(self):
        #TODO: read the data from pickle
        super(SeedlingTrain, self).__init__()

        with open("mask_rcnn_data_train.pkl", "rb") as f:
            self.train = pickle.load(f)

        with open("mask_rcnn_data_mask.pkl", "rb") as f:
            self.mask_data = pickle.load(f)

        self.train_samples = int(len(self.train)*0.8)

        for i, label in enumerate(['Black-grass', 'Charlock', 'Cleavers', 'Common Chickweed', 'Common wheat', 'Fat Hen',
                      'Loose Silky-bent',
                      'Maize', 'Scentless Mayweed', 'Shepherds Purse', 'Small-flowered Cranesbill', 'Sugar beet']):
            print(i,label)
            self.add_class("seedling",i+1,label)

        # !!!!! should attention that: the class should start from 1 rather than 0 !!!!! 0 is background!!!!!


        for i in range(self.train_samples):
            self.add_image(image_id=i, label=np.array([self.mask_data[i][1][0]+1]),source="seedling",path=None)




    def load_image(self, image_id):
        """Generate an image from the specs of the given image ID.
        Typically this function loads the image from a file, but
        in this case it generates the image on the fly from the
        specs in image_info.
        """
        return self.train[image_id]

    def load_mask(self,image_id):

        return self.mask_data[image_id][0], np.array([self.mask_data[image_id][1][0]+1])


    def image_reference(self, image_id):
        """Return the shapes data of the image."""
        info = self.image_info[image_id]
        return info["label"]


class SeedlingVal(utils.Dataset):
    """Generates the shapes synthetic dataset. The dataset consists of simple
    shapes (triangles, squares, circles) placed randomly on a blank surface.
    The images are generated on the fly. No file access required.
    """
    def __init__(self):
        #TODO: read the data from pickle
        super(SeedlingVal, self).__init__()

        with open("mask_rcnn_data_train.pkl", "rb") as f:
            self.train = pickle.load(f)

        with open("mask_rcnn_data_mask.pkl", "rb") as f:
            self.mask_data = pickle.load(f)

        self.val_samples = int(len(self.train)*0.8)


        for i, label in enumerate(['Black-grass', 'Charlock', 'Cleavers', 'Common Chickweed', 'Common wheat', 'Fat Hen',
                      'Loose Silky-bent',
                      'Maize', 'Scentless Mayweed', 'Shepherds Purse', 'Small-flowered Cranesbill', 'Sugar beet']):
            print(i,label)
            self.add_class("seedling",i+1,label)

        # !!!!! should attention that: the class should start from 1 rather than 0 !!!!! 0 is background!!!!!



        for i in range(self.val_samples,):
            self.add_image(image_id=i-self.val_samples, label=np.array([self.mask_data[i][1][0]+1]),source="seedling",path=None)




    def load_image(self, image_id):
        """Generate an image from the specs of the given image ID.
        Typically this function loads the image from a file, but
        in this case it generates the image on the fly from the
        specs in image_info.
        """
        return self.train[image_id]

    def load_mask(self,image_id):

        return self.mask_data[image_id][0], np.array([self.mask_data[image_id][1][0]+1])


    def image_reference(self, image_id):
        """Return the shapes data of the image."""
        info = self.image_info[image_id]
        return info["label"]



class SeedlingTest(utils.Dataset):
    def __init__(self):
        super(SeedlingTest, self).__init__()

        with open("mask_rcnn_data_test.pkl", "rb") as f:
            self.test = pickle.load(f)

        for i, label in enumerate(['Black-grass', 'Charlock', 'Cleavers', 'Common Chickweed', 'Common wheat', 'Fat Hen',
                      'Loose Silky-bent',
                      'Maize', 'Scentless Mayweed', 'Shepherds Purse', 'Small-flowered Cranesbill', 'Sugar beet']):
            print(i,label)
            self.add_class("seedling",i+1,label)

        # !!!!! should attention that: the class should start from 1 rather than 0 !!!!! 0 is background!!!!!

        self.test_sample = len(self.test)

        for i in range(self.test_sample):
            self.add_image(image_id=i, label=None,source="seedling",path=None)


    def load_image(self, image_id):
        """Generate an image from the specs of the given image ID.
        Typically this function loads the image from a file, but
        in this case it generates the image on the fly from the
        specs in image_info.
        """
        return self.test[image_id][0]
    def load_filename(self,image_id):
        return self.test[image_id][1]

    def load_mask(self,image_id):

        return self.mask_data[image_id][0], [self.mask_data[image_id][1][0]+1]



debug = 0
if debug:

    dataset_train = SeedlingTrain()
    dataset_train.prepare()

    image_ids = np.random.choice(dataset_train.image_ids, 3)

    for image_id in image_ids:
        image = dataset_train.load_image(image_id)
        mask, class_ids = dataset_train.load_mask(image_id)

        #print("the dataset shape: \n")
        #print(image.shape,"\n",mask.shape,"\n",class_ids)
        #print(image,"\n",mask)

        visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)



# Validation dataset


dataset_train = SeedlingTrain()
dataset_train.prepare()

dataset_val = SeedlingVal()
dataset_val.prepare()


image_id_debug = 0
if image_id_debug:

    for i in range(dataset_train.train_samples):
        print(dataset_train.load_mask(i)[1])
    exit()


# Create model in training mode
model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)


# Which weights to start with?
init_with = "coco"  # imagenet, coco, or last

if init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == "coco":
    # Load weights trained on MS COCO, but skip layers that
    # are different due to the different number of classes
    # See README for instructions to download the COCO weights
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                "mrcnn_bbox", "mrcnn_mask"])
elif init_with == "last":
    # Load the last model you trained and continue training
    model.load_weights(model.find_last()[1], by_name=True)




model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=1,
                layers='heads')

model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE / 10,
            epochs=2,
            layers="all")

class InferenceConfig(ShapesConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

inference_config = InferenceConfig()

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference",
                          config=inference_config,
                          model_dir=MODEL_DIR)

# Get path to saved weights
# Either set a specific path or find last trained weights
# model_path = os.path.join(ROOT_DIR, ".h5 file name here")
model_path = model.find_last()[1]

# Load trained weights (fill in path to trained weights here)
assert model_path != "", "Provide path to trained weights"
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)



# Test on a random image
image_id = random.choice(dataset_val.image_ids)
original_image, image_meta, gt_class_id, gt_bbox, gt_mask =\
    modellib.load_image_gt(dataset_val, inference_config,
                           image_id, use_mini_mask=False)

log("original_image", original_image)
log("image_meta", image_meta)
log("gt_class_id", gt_bbox)
log("gt_bbox", gt_bbox)
log("gt_mask", gt_mask)

visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id,
                            dataset_train.class_names, figsize=(8, 8))




results = model.detect([original_image], verbose=1)

r = results[0]
visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'],
                            dataset_val.class_names, r['scores'], ax=get_ax())

# Compute VOC-Style mAP @ IoU=0.5
# Running on 10 images. Increase for better accuracy.
image_ids = np.random.choice(dataset_val.image_ids, 10)
APs = []
for image_id in image_ids:
    # Load image and ground truth data
    image, image_meta, gt_class_id, gt_bbox, gt_mask = \
        modellib.load_image_gt(dataset_val, inference_config,
                               image_id, use_mini_mask=False)
    molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)
    # Run object detection
    results = model.detect([image], verbose=0)
    r = results[0]
    # Compute AP
    AP, precisions, recalls, overlaps = \
        utils.compute_ap(gt_bbox, gt_class_id,
                         r["rois"], r["class_ids"], r["scores"])
    APs.append(AP)

print("mAP: ", np.mean(APs))


class PredictSubmit(object):



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