import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'                    # Hide log

import tensorflow as tf                                     # Data augmentation
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import PIL                                                  # Read pictures
from PIL import Image, ImageOps                             # Using ImageOps for padding
import numpy as np                                          # Handling picture as matrix, randomize etc...
import matplotlib.pyplot as plt                             # Optional, if you would like to show results

import random
import sklearn
from sklearn import neighbors
from sklearn.metrics import confusion_matrix

# Need to supervise
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Layer, AveragePooling2D, Input
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Activation
from keras import backend as K

from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras.models import Model


## BASIC VARIABLES

median_ratio = 1.48
q3_x = 600
q3_y = 400
alpha = 0.2
lr = 0.05
epochs = 100

## BASIC FUNCTIONS
   
def read_image(pic_list):   
# Initalize az empty list which will contain the 3 picture as array. 
    apn_list = []
# Read 3 picture
    for i in pic_list:
    # Read image and get dimensions
        img = PIL.Image.open(i).convert('RGB')              # Read image
        height = np.asarray(img).shape[0]                   # Get y max coord
        width = np.asarray(img).shape[1]                    # Get x max coord
        current_ratio = float(width/height)                 # Calculate ratio
        
    # Define initial new resolution (temporary)
        new_tb = height
        new_lr = width

    # Calculate new size with keeping the aspect ratio
        if current_ratio < median_ratio:                    
            new_lr = int(median_ratio*height)
        elif current_ratio > median_ratio:
            new_tb = int(width/median_ratio)

    # Padding new image to the middle of the picture
        new_img = PIL.ImageOps.pad(img, (new_lr,new_tb), color=None, centering=(0.5, 0.5))

    # Using Quantile 3 to resize images with keeping the aspect ratio
        new_height = int(q3_x * new_tb / new_lr)
        new_width  = int(q3_y * new_lr / new_tb)
        new_img = new_img.resize((new_width, new_height), Image.ANTIALIAS)

    # Padding the new image again to keep the q3_x and q3_y
        new_img = np.asarray(PIL.ImageOps.pad(new_img, (q3_x,q3_y), color=None, centering=(0.5, 0.5)))/255
        apn_list.append(new_img)
        
    #print(np.asarray(apn_list).shape)                      # Check dimensions if necessary
    return np.asarray(apn_list)

def lr_schedule(epoch):
    """Learning Rate Schedule
    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.
    # Arguments
        epoch (int): The number of epochs
    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr

## PATH DEFINITIONS
    
# 2880 pic
szazxszep="/home/wildhorse_project/detectron_pic/FINAL_TEST/szazxszep_train.csv"
thetisz="/home/wildhorse_project/detectron_pic/FINAL_TEST/thetisz_train.csv"
panka="/home/wildhorse_project/detectron_pic/FINAL_TEST/panka_train.csv"
huba="/home/wildhorse_project/detectron_pic/FINAL_TEST/huba_train.csv"
lilla="/home/wildhorse_project/detectron_pic/FINAL_TEST/lilla_train.csv"
lantos="/home/wildhorse_project/detectron_pic/FINAL_TEST/lantos_train.csv"

# Validation pictures
szazxszep_val="/home/wildhorse_project/detectron_pic/FINAL_TEST/less/szazxszep_val_80.csv"
thetisz_val="/home/wildhorse_project/detectron_pic/FINAL_TEST/less/thetisz_val_80.csv"
panka_val="/home/wildhorse_project/detectron_pic/FINAL_TEST/less/panka_val_80.csv"
huba_val="/home/wildhorse_project/detectron_pic/FINAL_TEST/less/huba_val_80.csv"
lilla_val="/home/wildhorse_project/detectron_pic/FINAL_TEST/less/lilla_val_80.csv"
lantos_val="/home/wildhorse_project/detectron_pic/FINAL_TEST/less/lantos_val_80.csv"
gerle_val="/home/wildhorse_project/detectron_pic/FINAL_TEST/less/gerle_val_80.csv"
noci_val="/home/wildhorse_project/detectron_pic/FINAL_TEST/less/noci_val_80.csv"
vadoc_val="/home/wildhorse_project/detectron_pic/FINAL_TEST/less/vadoc_val_80.csv"

# Less image with higher distance between frames
szazxszep_6th="/home/wildhorse_project/detectron_pic/FINAL_TEST/less/szazxszep_6th.csv"
thetisz_6th="/home/wildhorse_project/detectron_pic/FINAL_TEST/less/thetisz_6th.csv"
panka_6th="/home/wildhorse_project/detectron_pic/FINAL_TEST/less/panka_6th.csv"
huba_6th="/home/wildhorse_project/detectron_pic/FINAL_TEST/less/huba_6th.csv"
lilla_6th="/home/wildhorse_project/detectron_pic/FINAL_TEST/less/lilla_6th.csv"
lantos_6th="/home/wildhorse_project/detectron_pic/FINAL_TEST/less/lantos_6th.csv"
gerle_6th="/home/wildhorse_project/detectron_pic/FINAL_TEST/less/gerle_6th.csv"
noci_6th="/home/wildhorse_project/detectron_pic/FINAL_TEST/less/noci_6th.csv"
vadoc_6th="/home/wildhorse_project/detectron_pic/FINAL_TEST/less/vadoc_6th.csv"

# Catalog photos
szazxszep_plus_val="/home/wildhorse_project/detectron_pic/FINAL_TEST/plus_val_szazxszep.csv"
thetisz_plus_val="/home/wildhorse_project/detectron_pic/FINAL_TEST/plus_val_thetisz.csv"
panka_plus_val="/home/wildhorse_project/detectron_pic/FINAL_TEST/plus_val_panka.csv"
huba_plus_val="/home/wildhorse_project/detectron_pic/FINAL_TEST/plus_val_huba.csv"
lilla_plus_val="/home/wildhorse_project/detectron_pic/FINAL_TEST/plus_val_lilla.csv"
lantos_plus_val="/home/wildhorse_project/detectron_pic/FINAL_TEST/plus_val_lantos.csv"

# Cropped catalog photos
szazxszep_plus_val_crop="/home/wildhorse_project/detectron_pic/szazxszep_plus/plus_val_szazxszep_crop.csv"
thetisz_plus_val_crop="/home/wildhorse_project/detectron_pic/thetisz_plus/plus_val_thetisz_crop.csv"
panka_plus_val_crop="/home/wildhorse_project/detectron_pic/panka_plus/plus_val_panka_crop.csv"
huba_plus_val_crop="/home/wildhorse_project/detectron_pic/huba_plus/plus_val_huba_crop.csv"
lilla_plus_val_crop="/home/wildhorse_project/detectron_pic/lilla_plus/plus_val_lilla_crop.csv"
lantos_plus_val_crop="/home/wildhorse_project/detectron_pic/lantos_plus/plus_val_lantos_crop.csv"

# Cropped catalog photos with black
szazxszep_plus_val_crop_black="/home/wildhorse_project/detectron_pic/szazxszep_plus_black/plus_val_szazxszep_crop_black.csv"
thetisz_plus_val_crop_black="/home/wildhorse_project/detectron_pic/thetisz_plus_black/plus_val_thetisz_crop_black.csv"
panka_plus_val_crop_black="/home/wildhorse_project/detectron_pic/panka_plus_black/plus_val_panka_crop_black.csv"
huba_plus_val_crop_black="/home/wildhorse_project/detectron_pic/huba_plus_black/plus_val_huba_crop_black.csv"
lilla_plus_val_crop_black="/home/wildhorse_project/detectron_pic/lilla_plus_black/plus_val_lilla_crop_black.csv"
lantos_plus_val_crop_black="/home/wildhorse_project/detectron_pic/lantos_plus_black/plus_val_lantos_crop_black.csv"

## APPEND FILE PATHES

horses = [szazxszep,thetisz,panka,huba,lilla,lantos]                                                                                                   # 2880 image
val_horses = [szazxszep_val,thetisz_val,panka_val,huba_val,lilla_val,lantos_val,gerle_val,noci_val,vadoc_val]                                                                       # Video validation
val_plus_horses = [szazxszep_plus_val,thetisz_plus_val,panka_plus_val,huba_plus_val,lilla_plus_val,lantos_plus_val]                                    # Catalog photos validation
val_plus_horses_crop = [szazxszep_plus_val_crop,thetisz_plus_val_crop,panka_plus_val_crop,huba_plus_val_crop,lilla_plus_val_crop,lantos_plus_val_crop] # Cropped photos validation
less_horses = [szazxszep_6th,thetisz_6th,panka_6th,huba_6th,lilla_6th,lantos_6th,gerle_6th,noci_6th,vadoc_6th]                                                                      # Every 6th images
val_plus_horses_crop_black = [szazxszep_plus_val_crop_black,thetisz_plus_val_crop_black,panka_plus_val_crop_black,
                            huba_plus_val_crop_black,lilla_plus_val_crop,lantos_plus_val_crop]                                                         # Cropped photos black validation

## CREATE LIST FROM READ DATA

# Generate a big numpy array from the '.csv' file
# my_data = np.empty(shape=(0,3))
# for i in horses:
#     my_data = np.concatenate((my_data,np.genfromtxt(i, dtype=str, delimiter=',')),axis = 0)

# Less_data
my_data2 = np.empty(shape=(0,3))
for i in less_horses:
    my_data2 = np.concatenate((my_data2,np.genfromtxt(i, dtype=str, delimiter=',')),axis = 0)
    
# Generate a big numpy array from the '.csv' file
val_data = np.empty(shape=(0,3))
for i in val_horses:
    val_data = np.concatenate((val_data,np.genfromtxt(i, dtype=str, delimiter=',')),axis = 0)

# Generate a big numpy array from the '.csv' file
val_plus_data = np.empty(shape=(0,3))
for i in val_plus_horses:
    val_plus_data = np.concatenate((val_plus_data,np.genfromtxt(i, dtype=str, delimiter=',')),axis = 0)
    
# Generate a big numpy array from the '.csv' file
val_plus_data_crop = np.empty(shape=(0,3))
for i in val_plus_horses_crop:
    val_plus_data_crop = np.concatenate((val_plus_data_crop,np.genfromtxt(i, dtype=str, delimiter=',')),axis = 0)
    
# Generate a big numpy array from the '.csv' file
val_plus_data_crop_black = np.empty(shape=(0,3))
for i in val_plus_horses_crop_black:
    val_plus_data_crop_black = np.concatenate((val_plus_data_crop_black,np.genfromtxt(i, dtype=str, delimiter=',')),axis = 0)
    
## DEFINE MODEL
    
resnet1 = tf.keras.applications.ResNet50V2(input_shape = (q3_y,q3_x,3),include_top=False, classes = 9)
#resnet1 = tf.keras.applications.ResNet101V2(input_shape = (q3_y,q3_x,3),include_top=False, classes = 9, weights='imagenet')
#resnet1 = tf.keras.applications.InceptionResNetV2(input_shape = (q3_y,q3_x,3),include_top=False, classes = 9, weights='imagenet')

y = tf.keras.layers.GlobalAveragePooling2D()(resnet1.output)
outputs = Dense(9,
                activation='softmax',
                activity_regularizer='l2',
                kernel_initializer='he_normal',
                )(y)

#model1 = tf.keras.models.load_model('/home/dkatona/regulizer_1e1.h5')
model1 = Model(inputs=resnet1.input, outputs=outputs)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

## READ IMAGES

# train_images = read_image(my_data[:,0])
# train_classes = np.array(my_data[:,1].astype(int))

train_images = read_image(my_data2[:,0])
train_classes = np.array(my_data2[:,1].astype(int))-1
train_classes = tf.keras.utils.to_categorical(train_classes, 9)


val_images = read_image(val_data[:,0])
val_classes = np.array(val_data[:,1].astype(int))-1
val_classes = tf.keras.utils.to_categorical(val_classes, 9)

val_plus_images = read_image(val_plus_data[:,0])
val_plus_classes = np.array(val_plus_data[:,1].astype(int))

# val_plus_images_crop = read_image(val_plus_data_crop[:,0])
# val_plus_classes_crop = np.array(val_plus_data_crop[:,1].astype(int))

# val_plus_images_crop_black = read_image(val_plus_data_crop_black[:,0])
# val_plus_classes_crop_black = np.array(val_plus_data_crop_black[:,1].astype(int))

print("ALL FILES WERE READ")

datagen = ImageDataGenerator(
        # set input mean to 0 over the dataset
        featurewise_center=False,
        # set each sample mean to 0
        samplewise_center=False,
        # divide inputs by std of dataset
        featurewise_std_normalization=False,
        # divide each input by its std
        samplewise_std_normalization=False,
        # apply ZCA whitening
        zca_whitening=False,
        # epsilon for ZCA whitening
        zca_epsilon=1e-06,
        # randomly rotate images in the range (deg 0 to 180)
        rotation_range=30,
        # randomly shift images horizontally
        width_shift_range=0.1,
        # randomly shift images vertically
        height_shift_range=0.1,
        # set range for random shear
        shear_range=0.,
        # set range for random zoom
        zoom_range=0.1,
        # set range for random channel shifts
        channel_shift_range=0.,
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        # value used for fill_mode = "constant"
        cval=0.,
        # randomly flip images
        horizontal_flip=True,
        # randomly flip images
        vertical_flip=True,
        # set rescaling factor (applied before any other transformation)
        rescale=None,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format=None,
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.0,
        # Brightness range
        brightness_range=[0.9,1.1]
)

datagen.fit(train_images)
epochs = 400

model1.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=lr_schedule(0)),
              metrics=['accuracy'])

model1.fit_generator(datagen.flow(train_images, train_classes, batch_size=4),
                        validation_data=(val_images, val_classes),
                        epochs=epochs, verbose=1)
model1.save('/home/dkatona/paci_classy.h5')