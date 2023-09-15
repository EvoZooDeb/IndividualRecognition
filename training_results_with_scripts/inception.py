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
lr = 0.5
epochs = 100

## BASIC FUNCTIONS

def triplet_loss(model1, apn_list, alpha):
    y = data_augmentation(apn_list)
    x = model1(y, training = True)
    a = tf.linalg.normalize(x[0])[0]
    p = tf.linalg.normalize(x[1])[0]
    n = tf.linalg.normalize(x[2])[0]
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
    
#resnet1 = tf.keras.applications.ResNet50V2(input_shape = (q3_y,q3_x,3),include_top=False, classes = 16, weights='imagenet')
#resnet1 = tf.keras.applications.ResNet101V2(input_shape = (q3_y,q3_x,3),include_top=False, classes = 16, weights='imagenet')
resnet1 = tf.keras.applications.InceptionResNetV2(input_shape = (q3_y,q3_x,3),include_top=False, classes = 16, weights='imagenet')
# depth = 6*n+2
depth = 20

y = tf.keras.layers.GlobalAveragePooling2D()(resnet1.output)
outputs = Dense(16,
                activation='softmax',
                activity_regularizer='l2',
                kernel_initializer='he_normal',
                kernel_regularizer=l2(1e-1))(y)

#model1 = tf.keras.models.load_model('/home/dkatona/regulizer_1e1.h5')
model1 = Model(inputs=resnet1.input, outputs=outputs)
optimizer = tf.keras.optimizers.SGD(learning_rate=lr)

## READ IMAGES

# train_images = read_image(my_data[:,0])
# train_classes = np.array(my_data[:,1].astype(int))

train_images = read_image(my_data2[:,0])
train_classes = np.array(my_data2[:,1].astype(int))

val_images = read_image(val_data[:,0])
val_classes = np.array(val_data[:,1].astype(int))

val_plus_images = read_image(val_plus_data[:,0])
val_plus_classes = np.array(val_plus_data[:,1].astype(int))

# val_plus_images_crop = read_image(val_plus_data_crop[:,0])
# val_plus_classes_crop = np.array(val_plus_data_crop[:,1].astype(int))

# val_plus_images_crop_black = read_image(val_plus_data_crop_black[:,0])
# val_plus_classes_crop_black = np.array(val_plus_data_crop_black[:,1].astype(int))

print("ALL FILES WERE READ")

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
        train_pred = model1.predict(train_images, batch_size = 1)
        train_pred = tf.linalg.normalize(train_pred, axis = 1)[0] 
        
        val_pred = model1.predict(val_images, batch_size = 1)
        val_pred = tf.linalg.normalize(val_pred, axis = 1)[0]
        
        val_plus_pred = model1.predict(val_plus_images, batch_size = 1)
        val_plus_pred = tf.linalg.normalize(val_plus_pred, axis = 1)[0] 
        
#         val_plus_pred_crop = model1.predict(val_plus_images_crop, batch_size = 1)
#         val_plus_pred_crop = tf.linalg.normalize(val_plus_pred_crop, axis = 1)[0] 
        
#         val_plus_pred_crop_black = model1.predict(val_plus_images_crop_black, batch_size = 1)
#         val_plus_pred_crop_black = tf.linalg.normalize(val_plus_pred_crop_black, axis = 1)[0] 
        
        n_neighbors = 5
        clf = neighbors.KNeighborsClassifier(n_neighbors)
        clf.fit(train_pred,train_classes)
        val_res = clf.predict(val_pred)
        val_plus_res = clf.predict(val_plus_pred)
#         val_plus_res_crop = clf.predict(val_plus_pred_crop)
#         val_plus_res_crop_black = clf.predict(val_plus_pred_crop_black)
        
        accuracy = sum(val_res == val_classes)/val_data.shape[0]
        plus_accuracy = sum(val_plus_res == val_plus_classes)/val_plus_data.shape[0]
#         plus_accuracy_crop = sum(val_plus_res_crop == val_plus_classes_crop)/val_plus_data_crop.shape[0]
#         plus_accuracy_crop_black = sum(val_plus_res_crop_black == val_plus_classes_crop_black)/val_plus_data_crop_black.shape[0]
        
        print(accuracy)
        print("EPOCH CONF_MATRIX VAL_RES")
        print(confusion_matrix(val_res, val_classes))
        conf_matrix = confusion_matrix(val_res, val_classes)+(np.ones(shape = (len(all_classes),len(all_classes)))*5)
        np.fill_diagonal(conf_matrix, 0)
#         print(val_res)
        
        if accuracy > max_accuracy:
            max_acc_epoch = i
            max_accuracy  = accuracy
            model1.save('/home/dkatona/inception.h5')
        
        print("#########################")
        
        print(plus_accuracy)
        print("Catalog pictures: EPOCH CONF_MATRIX VAL_PLUS_RES")
        print(confusion_matrix(val_plus_res, val_plus_classes))
#         print(val_plus_res)
        
#         print("#########################")
        
#         print(plus_accuracy_crop)
#         print("EPOCH CONF_MATRIX VAL_PLUS_RES_CROP")
#         print(confusion_matrix(val_plus_res_crop, val_plus_classes_crop))
#         print(val_plus_res_crop)
        
#         print("#########################")
        
#         print(plus_accuracy_crop_black)
#         print("EPOCH CONF_MATRIX VAL_PLUS_RES_CROP_black")
#         print(confusion_matrix(val_plus_res_crop_black, val_plus_classes_crop_black))
#         print(val_plus_res_crop_black)
        
    mean_loss = 0
    indexing_counter = 0
    zero_counter = 0
    for j in np.random.permutation( range(len(my_data2)) ):
        indexing_counter+=1
        p = np.random.choice(np.where( (my_data2[:,1] == my_data2[j,1]))[0],1)[0] 
        if p == j:
            p = np.random.choice(np.where( (my_data2[:,1] == my_data2[j,1]))[0],1)[0]
        p_class = int(my_data2[j,1])-1
        
        n_column = random.choices(population=all_classes, weights=conf_matrix[:,p_class], k=1)
        n = np.random.choice(np.where(my_data2[:,1] == str(n_column[0]))[0],1)[0]                           # Get random id from another class
        
        apn_list = np.stack((train_images[j],train_images[p],train_images[n]), axis = 0)
        loss,grad_m1 = grad(model1,apn_list,alpha)
        mean_loss = (mean_loss + loss)/(j+1)
        optimizer.apply_gradients(zip(grad_m1, model1.trainable_variables))
        if loss == 0:
            zero_counter += 1
            
    if zero_counter == len(my_data2):
        print("Too many zeros")
        break   
    print("Epoch: " + str(i) + "     " + "    mean_loss: " + str(float(mean_loss)))
            
print("Max_value_epoch: " + str(max_acc_epoch))
