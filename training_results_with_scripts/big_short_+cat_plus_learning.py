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
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Activation, Lambda
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
lr = 0.005
epochs = 50
columns_shape = 3
dimensions = 8

## BASIC FUNCTIONS

def triplet_loss(model1, apn_list, alpha):
    y = data_augmentation(apn_list)
    x = model1(y, training = True)
    a = x[0]
    p = x[1]
    n = x[2]
    return tf.math.maximum(tf.math.square(tf.norm(a-p, ord='euclidean'))-tf.math.square(tf.norm(a-n, ord='euclidean'))+alpha,0)
 
def grad(model1, apn_list, alpha):
    with tf.GradientTape() as tape:
        loss = triplet_loss(model1, apn_list, alpha)+sum(model1.losses)
    return loss,tape.gradient(loss,model1.trainable_variables)
    
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

data_augmentation = tf.keras.Sequential([
  layers.experimental.preprocessing.RandomZoom((-0.05, 0.2),(-0.05, 0.2),fill_mode='nearest'),
  layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
  layers.experimental.preprocessing.RandomRotation(0.02, 'nearest'),
  layers.experimental.preprocessing.RandomContrast(0.7),
])

def create_dataset(raw_dataset,columns_shape):
    dataset = np.empty(shape=(0,columns_shape))
    for i in raw_dataset:
        dataset = np.concatenate((dataset,np.genfromtxt(i, dtype=str, delimiter=',')),axis = 0)
    return dataset
    
## PATH DEFINITIONS

# 200picture with black background
gerle_black="/home/wildhorse_project/detectron_pic/black/gerle_black_170.csv"
huba_black="/home/wildhorse_project/detectron_pic/black/huba_black_200.csv"
lantos_black="/home/wildhorse_project/detectron_pic/black/lantos_black_200.csv"
lilla_black="/home/wildhorse_project/detectron_pic/black/lilla_black_200.csv"
noci_black="/home/wildhorse_project/detectron_pic/black/noci_black_175.csv"
panka_black="/home/wildhorse_project/detectron_pic/black/panka_black_200.csv"
szazxszep_black="/home/wildhorse_project/detectron_pic/black/szazxszep_black_200.csv"
thetisz_black="/home/wildhorse_project/detectron_pic/black/thetisz_black_200.csv"
vadoc_black="/home/wildhorse_project/detectron_pic/black/vadoc_black_200.csv"

# 30picture with black background 30
gerle_black_val="/home/wildhorse_project/detectron_pic/black/gerle_black_30_val.csv"
huba_black_val="/home/wildhorse_project/detectron_pic/black/huba_black_30_val.csv"
lantos_black_val="/home/wildhorse_project/detectron_pic/black/lantos_black_30_val.csv"
lilla_black_val="/home/wildhorse_project/detectron_pic/black/lilla_black_30_val.csv"
noci_black_val="/home/wildhorse_project/detectron_pic/black/noci_black_30_val.csv"
panka_black_val="/home/wildhorse_project/detectron_pic/black/panka_black_30_val.csv"
szazxszep_black_val="/home/wildhorse_project/detectron_pic/black/szazxszep_black_30_val.csv"
thetisz_black_val="/home/wildhorse_project/detectron_pic/black/thetisz_black_30_val.csv"
vadoc_black_val="/home/wildhorse_project/detectron_pic/black/vadoc_black_30_val.csv"
    
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

horse_mixed=[gerle_black,huba_black,lantos_black,lilla_black,noci_black,panka_black,szazxszep_black,thetisz_black,vadoc_black,
             szazxszep_6th,thetisz_6th,panka_6th,huba_6th,lilla_6th,lantos_6th,gerle_6th,noci_6th,vadoc_6th]

horse_val_mixed=[huba_black_val,lantos_black_val,lilla_black_val,panka_black_val,szazxszep_black_val,thetisz_black_val,
                szazxszep_val,thetisz_val,panka_val,huba_val,lilla_val,lantos_val]

horse_catalog_pic =[szazxszep_plus_val_crop_black,thetisz_plus_val_crop_black,panka_plus_val_crop_black,huba_plus_val_crop_black,lilla_plus_val_crop_black,lantos_plus_val_crop_black]

#train_data = create_dataset(horse_mixed,columns_shape)
train_data = create_dataset(horse_catalog_pic,columns_shape)
validation_data = create_dataset(horse_val_mixed,columns_shape)
catalog_data = create_dataset(horse_catalog_pic,columns_shape)

## DEFINE MODEL
    
model1 = tf.keras.models.load_model("/home/dkatona/inception_test_latest_catalog_BEST.h5")

optimizer = tf.keras.optimizers.SGD(learning_rate=lr)

## READ IMAGES

train_images = read_image(catalog_data[:,0])
train_classes = np.array(catalog_data[:,1].astype(int))

validation_images = read_image(validation_data[:,0])
validation_classes = np.array(validation_data[:,1].astype(int))

#catalog_images = read_image(catalog_data[:,0])
#catalog_classes = np.array(catalog_data[:,1].astype(int))

print("ALL FILES WERE READ:")
print("  Training pictures: ", train_images.shape[0])
print("  Validation pictures: ", validation_images.shape[0])
#print("  Catalog pictures: ", catalog_images.shape[0])

# Simple initialization 

all_classes = np.unique(train_classes)
conf_matrix = np.ones(shape = (len(all_classes),len(all_classes)))
np.fill_diagonal(conf_matrix, 0)

max_accuracy = 0
max_acc_epoch = 0

for i in range(epochs):
    if i%10 == 9:
        lr *=0.5
        optimizer.learning_rate=lr
    if i%2 == 1:    
        train_prediction = model1.predict(train_images, batch_size = 1)
        #train_prediction = tf.linalg.normalize(train_prediction, axis = 1)[0] 
        
        validation_prediction = model1.predict(validation_images, batch_size = 1)
        #validation_prediction = tf.linalg.normalize(validation_prediction, axis = 1)[0]
        
        #catalog_prediction = model1.predict(catalog_images, batch_size = 1)
        
        # KNN Neighbours
        n_neighbors = 5
        clf = neighbors.KNeighborsClassifier(n_neighbors)
        clf.fit(train_prediction,train_classes)
        
        train_result_array = clf.predict(train_prediction)
        validation_result_array = clf.predict(validation_prediction)
        #catalog_result_array = clf.predict(catalog_prediction)
        
        train_accuracy = sum(train_result_array == train_classes)/train_data.shape[0]
        validation_accuracy = sum(validation_result_array == validation_classes)/validation_data.shape[0]
        #catalog_accuracy = sum(catalog_result_array == catalog_classes)/catalog_data.shape[0]
    
        print("Training accuracy: ", train_accuracy)
        print("  Confusion matrix (train pictures over training prediction)")
        print(confusion_matrix(train_result_array, train_classes))
        
        print("############################################################################################")
        
        print("Validation accuracy: ", validation_accuracy)
        print("  Confusion matrix (validation pictures over validation prediction)")
        print(confusion_matrix(validation_result_array, validation_classes))
        
        if validation_accuracy > max_accuracy:
            max_acc_epoch = i
            max_accuracy  = validation_accuracy
            model1.save('/home/dkatona/inception_test_latest_catalog_plus_learning.h5')
        
    mean_loss = 0
    indexing_counter = 0
    zero_counter = 0
        
    for j in np.random.permutation( range(len(train_data)) ):
        indexing_counter+=1
        p = np.random.choice(np.where( (train_data[:,1] == train_data[j,1]))[0],1)[0] 
        if p == j:
            p = np.random.choice(np.where( (train_data[:,1] == train_data[j,1]))[0],1)[0]
        p_class = int(train_data[j,1])-1
        
        n_column = random.choices(population=all_classes, weights=conf_matrix[:,p_class], k=1)
        n = np.random.choice(np.where(train_data[:,1] == str(n_column[0]))[0],1)[0]                           # Get random id from another class
         
        apn_list = np.stack((train_images[j],train_images[p],train_images[n]), axis = 0)
        loss,grad_m1 = grad(model1,apn_list,alpha)
        mean_loss = (mean_loss + loss)/(j+1)
        optimizer.apply_gradients(zip(grad_m1, model1.trainable_variables))
        if loss == 0:
            zero_counter += 1
            
    if zero_counter == len(train_data):
        print("Too many zeros")
        break   
    print("Epoch: " + str(i) + "     " + "    mean_loss: " + str(float(mean_loss)))
            
print("Max_value_epoch: " + str(max_acc_epoch))
