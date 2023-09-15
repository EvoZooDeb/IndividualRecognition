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




median_ratio = 1.48
q3_x = 820
q3_y = 630
alpha = 0.2
lr = 1e-1
epochs = 100

def triplet_loss(model1, apn_list, alpha):
    y = data_augmentation(apn_list)
    x = model1(y, training = True)
    a = tf.linalg.normalize(x[0])[0]
    p = tf.linalg.normalize(x[1])[0]
    n = tf.linalg.normalize(x[2])[0]
    return tf.math.maximum(tf.math.square(tf.norm(a-p, ord='euclidean'))-tf.math.square(tf.norm(a-n, ord='euclidean'))+alpha,0)
    
def grad(model1, apn_list, alpha):
    with tf.GradientTape() as tape:
        t_loss = triplet_loss(model1, apn_list, alpha)
        loss = t_loss+(0.01*sum(model1.losses))
        #loss = t_loss
    return t_loss,loss,tape.gradient(loss,model1.trainable_variables)
    
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

# RESNET V1
# depth = 6*n+2

# RESNET V2
depth = 20
    
data_augmentation = tf.keras.Sequential([
  layers.experimental.preprocessing.RandomZoom((-0.05, 0.2),(-0.05, 0.2),fill_mode='nearest'),
  layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
  layers.experimental.preprocessing.RandomRotation(0.02, 'nearest'),
  layers.experimental.preprocessing.RandomContrast(0.7),
])

# Input data
# szazxszep_L="/home/wildhorse_project/detectron_pic/szazxszep_L.csv"                         # Examined horse  [take 2 horse]
# thetisz_L="/home/wildhorse_project/detectron_pic/thetisz_L.csv"                      # Different horse [take 1 horse]
# panka_L="/home/wildhorse_project/detectron_pic/panka_L.csv"

szazxszep="/home/wildhorse_project/detectron_pic/horses_with_3param/szazxszep_train.csv"                         # Examined horse  [take 2 horse]
thetisz="/home/wildhorse_project/detectron_pic/horses_with_3param/thetisz_train.csv"                      # Different horse [take 1 horse]
panka="/home/wildhorse_project/detectron_pic/horses_with_3param/panka_train.csv"
huba="/home/wildhorse_project/detectron_pic/horses_with_3param/huba_train.csv"
lilla="/home/wildhorse_project/detectron_pic/horses_with_3param/lilla_train.csv"
lantos="/home/wildhorse_project/detectron_pic/horses_with_3param/lantos_train.csv"

horses = [szazxszep,thetisz,panka,huba,lilla,lantos]

# Generate a big numpy array from the '.csv' file
my_data = np.empty(shape=(0,3))
for i in horses:
    my_data = np.concatenate((my_data,np.genfromtxt(i, dtype=str, delimiter=',')),axis = 0)

# Validaton dataset
# szazxszep_R="/home/wildhorse_project/detectron_pic/szazxszep_R.csv"                         # Examined horse  [take 2 horse]
# thetisz_R="/home/wildhorse_project/detectron_pic/thetisz_R.csv"                      # Different horse [take 1 horse]
# panka_R="/home/wildhorse_project/detectron_pic/panka_R.csv"                   

szazxszep_val="/home/wildhorse_project/detectron_pic/horses_with_3param/szazxszep_val.csv"                         # Examined horse  [take 2 horse]
thetisz_val="/home/wildhorse_project/detectron_pic/horses_with_3param/thetisz_val.csv"                      # Different horse [take 1 horse]
panka_val="/home/wildhorse_project/detectron_pic/horses_with_3param/panka_val.csv"
huba_val="/home/wildhorse_project/detectron_pic/horses_with_3param/huba_val.csv"
lilla_val="/home/wildhorse_project/detectron_pic/horses_with_3param/lilla_val.csv"
lantos_val="/home/wildhorse_project/detectron_pic/horses_with_3param/lantos_val.csv"

val_horses = [szazxszep_val,thetisz_val,panka_val,huba_val,lilla_val,lantos_val]

# Generate a big numpy array from the '.csv' file
val_data = np.empty(shape=(0,3))
for i in val_horses:
    val_data = np.concatenate((val_data,np.genfromtxt(i, dtype=str, delimiter=',')),axis = 0)

resnet1 = tf.keras.applications.ResNet50V2(input_shape = (q3_y,q3_x,3),include_top=False, classes = 16, weights='imagenet')
#resnet1 = tf.keras.applications.ResNet101V2(input_shape = (q3_y,q3_x,3),include_top=False, classes = 16, weights='imagenet')

# Conv2D Layer
# conv_layer = tf.keras.layers.Conv2D(
#     filters = 3072, kernel_size = 7, strides=(3,3), padding='valid',
#     data_format=None, dilation_rate=(1, 1), groups=1, activation='relu'
# )(resnet1.output)

#y = tf.keras.layers.GlobalAveragePooling2D()(conv_layer)
y = tf.keras.layers.GlobalAveragePooling2D()(resnet1.output)

# y = tf.keras.layers.AveragePooling2D(
#     pool_size=(3, 3), strides=3
# )(resnet1.output)

# y = tf.keras.layers.Flatten()(y)

# Dense Layer
outputs = Dense(
    16, activation='softmax', kernel_initializer='he_normal', kernel_regularizer='l2'
)(y)

model1 = Model(inputs=resnet1.input, outputs=outputs)
optimizer = tf.keras.optimizers.SGD(learning_rate=lr)

val_images = read_image(val_data[:,0])
val_classes = np.array(val_data[:,1].astype(int))

train_images = read_image(my_data[:,0])
train_classes = np.array(my_data[:,1].astype(int))
print(model1.summary())

for i in range(epochs):
    if i%10 == 9:
        lr *=0.5
        optimizer.learning_rate=lr
    if i%2 == 1:      
        val_pred = model1.predict(val_images, batch_size = 1)
        val_pred = tf.linalg.normalize(val_pred, axis = 1)[0]
        
        train_pred = model1.predict(train_images, batch_size = 1)
        train_pred = tf.linalg.normalize(train_pred, axis = 1)[0] 
        
        n_neighbors = 5
        clf = neighbors.KNeighborsClassifier(n_neighbors)
        clf.fit(train_pred,train_classes)
        val_res = clf.predict(val_pred)
        
        accuracy = sum(val_res == val_classes)/val_data.shape[0]
        
        print(accuracy)
        print(confusion_matrix(val_res, val_classes))
        #print(np.unique(val_res, return_counts=True))
        print(val_res)

    mean_loss = 0
    indexing_counter = 0
    for j in np.random.permutation( range(len(my_data)) ):
        indexing_counter+=1
        p = np.random.choice(np.where( (my_data[:,1] == my_data[j,1]))[0],1)[0] 
        if p == j:
            p = np.random.choice(np.where( (my_data[:,1] == my_data[j,1]))[0],1)[0]
        n = np.random.choice(np.where(my_data[:,1] != my_data[j,1])[0],1)[0]                            # Get random id from another class
        apn_list = np.stack((train_images[j],train_images[p],train_images[n]), axis = 0)
        # print(model1.layers[1].weights[0][0][0])
        t_loss, loss,grad_m1 = grad(model1,apn_list,alpha)
        mean_loss = (mean_loss + loss)/(j+1)
        optimizer.apply_gradients(zip(grad_m1, model1.trainable_variables))
        print("Epoch: " + str(i) + "     "  + "loss: " + str(float(loss)) + "    triplet_loss: "
              + str(float(t_loss)) + "     "  +   "step: " + str(indexing_counter))
            
        
