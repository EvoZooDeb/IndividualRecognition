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

from shutil import copyfile

# Global variables
median_ratio = 1.48                   # Picture aspect ratio
q3_x = 600                            # Decrease x size
q3_y = 400                            # Decrease y size
alpha = 0.2                           # Triplet loss variable
lr = 0.01                             # learning rate at start
epochs = 50                          
columns_shape = 6                     # Shape of the reading csv file
dimensions = 8                        # Output dimensions
knn_neighbors = 5                     # Common neighbor number
output_model_file_name = "/home/dkatona/inception_test_linear_v2.h5"
validation_x_epoch = 2
change_lr_x_epoch = 10


## BASIC FUNCTIONS

# Read multiple text file 
def read_csvs(file_path):
    csv_file_list= np.array([])
    for i in file_path:
        file = open(i, 'r')
        csv_file_list = np.append(csv_file_list,file.read().splitlines())
    return csv_file_list

# Create datasets
def create_dataset(raw_dataset,columns_shape):
    dataset = np.empty(shape=(0,columns_shape))
    output_string_list = []
    number_of_rows = 0
    for i in raw_dataset:
        dataset = np.concatenate((dataset,np.genfromtxt(i, dtype=str, delimiter=',',skip_header=1)),axis = 0)
        temporary_row = "     %s --> %s " % (str(dataset.shape[0]-number_of_rows).ljust(4), i)
        output_string_list.append(temporary_row)
        number_of_rows = dataset.shape[0]
    return output_string_list, dataset

# Read 3 images
def read_image(pic_list):   
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
  
# Gradient calculation
def grad(model1, apn_list, alpha):
    with tf.GradientTape() as tape:
        loss = triplet_loss(model1, apn_list, alpha)+sum(model1.losses)
    return loss,tape.gradient(loss,model1.trainable_variables)    

# Triplet loss calculation
def triplet_loss(model1, apn_list, alpha):
    y = data_augmentation(apn_list)
    x = model1(y, training = True)
    a = x[0]
    p = x[1]
    n = x[2]
    return tf.math.maximum(tf.math.square(tf.norm(a-p, ord='euclidean'))-tf.math.square(tf.norm(a-n, ord='euclidean'))+alpha,0)

# Calculate prediction accuracy
def prediction_calculation(output_message,fitting_images,fitting_classes,fit_or_not = False):
    # Background calculation
    prediction = model1.predict(fitting_images, batch_size = 1)
    if fit_or_not:
        global clf
        clf = neighbors.KNeighborsClassifier(knn_neighbors)
        clf.fit(prediction,fitting_classes)
    fitted_model_result_array = clf.predict(prediction)
    fitted_model_accuracy = sum(fitted_model_result_array == fitting_classes)/fitting_images.shape[0]
    
    # Simple printing
    print(output_message, fitted_model_accuracy)
    print("  Confusion matrix results:")
    print(confusion_matrix(fitted_model_result_array, fitting_classes))

    return fitted_model_accuracy
    
##########################################################################################################################################################
    
## MODEL DEFINITION   
#resnet1 = tf.keras.applications.ResNet50V2(input_shape = (q3_y,q3_x,3),include_top=False, classes = 16, weights='imagenet')
#resnet1 = tf.keras.applications.ResNet101V2(input_shape = (q3_y,q3_x,3),include_top=False, classes = 16, weights='imagenet')
resnet1 = tf.keras.applications.InceptionResNetV2(input_shape = (q3_y,q3_x,3),include_top=False, classes = dimensions, weights='imagenet')

y = tf.keras.layers.GlobalAveragePooling2D()(resnet1.output)
outputs = Dense(dimensions,
                activation='softmax',
#                activity_regularizer=l2(1e-3),
                kernel_initializer='he_normal',
                kernel_regularizer=l2(1e-1))(y)
outputs = Lambda(lambda t: tf.linalg.normalize(t, axis=1)[0])(outputs)
          
#model1 = Model(inputs=resnet1.input, outputs=outputs)
model1 = tf.keras.models.load_model('/home/dkatona/inception_test_linear.h5')
optimizer = tf.keras.optimizers.SGD(learning_rate=lr)    

## DATA AUGMENTATION
data_augmentation = tf.keras.Sequential([
  layers.experimental.preprocessing.RandomZoom((-0.05, 0.2),(-0.05, 0.2),fill_mode='nearest'),
  layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
  layers.experimental.preprocessing.RandomRotation(0.02, 'nearest'),
  layers.experimental.preprocessing.RandomContrast(0.7),
])

## APPEND FILE PATHES
# Six individual
ind6_crop_T="/home/wildhorse_project/detectron_pic/FILELIST/T_6ind_480 cropped.txt"
ind6_raw_catalog_V = "/home/wildhorse_project/detectron_pic/FILELIST/V_6ind_~5-10_[catalog].txt"
ind6_crop_black_catalog_V = "/home/wildhorse_project/detectron_pic/FILELIST/V_6ind_~5-10_cropped_black_bg_[catalog].txt"
ind6_crop_catalog_V = "/home/wildhorse_project/detectron_pic/FILELIST/V_6ind_~5-10_cropped_[catalog].txt"

# Nine individual
ind9_crop_black_T = "/home/wildhorse_project/detectron_pic/FILELIST/T_9ind_200_cropped_black_bg.txt"
ind9_crop_T = "/home/wildhorse_project/detectron_pic/FILELIST/T_9ind_80 cropped.txt"
ind9_crop_black_V = "/home/wildhorse_project/detectron_pic/FILELIST/V_9ind_30_cropped_black_bg.txt"
ind9_crop_V = "/home/wildhorse_project/detectron_pic/FILELIST/V_9ind_80 cropped.txt"

# horse_mixed = read_csvs([ind9_crop_black_T,
#                          ind9_crop_T])
# horse_val_mixed = read_csvs([ind9_crop_black_V,
#                          ind9_crop_V])

#HÜLYE VAGY DÁVID
# horse_mixed = read_csvs(["/home/wildhorse_project/detectron_pic/datasets/training_dataset.csv"])
#horse_val_mixed = read_csvs(["/home/wildhorse_project/detectron_pic/datasets/validation_dataset.csv"])

horse_mixed = ["/home/wildhorse_project/detectron_pic/datasets/training_dataset.csv"]
horse_val_mixed = ["/home/wildhorse_project/detectron_pic/datasets/validation_dataset.csv"]
# horse_catalog_pic = read_csvs([ind6_crop_black_catalog_V])

# SIMPLE CSV call e.g: 
#create_dataset(["/home/wildhorse_project/valami.csv"],columns_shape)
horse_mixed_details, train_data = create_dataset(horse_mixed,columns_shape)
horse_val_mixed_details, validation_data = create_dataset(horse_val_mixed,columns_shape)
# horse_catalog_pic_details, catalog_data = create_dataset(horse_catalog_pic,columns_shape)

print("Training set elements: ", *horse_mixed_details, sep = "\n")
print("Validation set elements: ",*horse_val_mixed_details, sep = "\n")
# print("Catalog set elements: ",*horse_catalog_pic_details, sep = "\n")
print("")

## READ IMAGES
train_images = read_image(train_data[:,0])
train_classes = np.array(train_data[:,2].astype(int))
all_classes = np.unique(train_classes)

validation_images = read_image(validation_data[:,0])
validation_classes = np.array(validation_data[:,2].astype(int))

# catalog_images = read_image(catalog_data[:,0])
# catalog_classes = np.array(catalog_data[:,2].astype(int))

print("ALL FILES WERE READ:")
print("  Training pictures:", train_images.shape[0])
print("  Validation pictures:", validation_images.shape[0])
# print("  Catalog pictures:", catalog_images.shape[0])
print("  Individuals classes:",all_classes)
print("")

# Simple initialization 
max_accuracy = 0
max_acc_epoch = 0
conf_matrix = np.ones(shape = (len(all_classes),len(all_classes)))
np.fill_diagonal(conf_matrix, 0)

print(" Start Training: ")
for i in range(epochs):
    if i%10 == 9:
        optimizer.learning_rate=lr*0.5
    if i%2 == 1: 
        t_acc = prediction_calculation("Training accuracy: ",train_images,train_classes,True)
        v_acc = prediction_calculation("Validation accuracy: ",validation_images,validation_classes)
#         c_acc = prediction_calculation("Catalog accuracy: ",catalog_images,catalog_classes)
        print("##############################################################################################################")

        if v_acc > max_accuracy:
            max_acc_epoch = i
            max_accuracy  = v_acc
            model1.save(output_model_file_name)
        
    mean_loss = 0
    indexing_counter = 0
    zero_counter = 0
        
    for j in np.random.permutation( range(len(train_data)) ):
        indexing_counter+=1
        p = np.random.choice(np.where( (train_data[:,2] == train_data[j,2]))[0],1)[0] 
        if p == j:
            p = np.random.choice(np.where( (train_data[:,2] == train_data[j,2]))[0],1)[0]
        p_class = int(train_data[j,2])-1
        
        n_column = random.choices(population=all_classes, weights=conf_matrix[:,p_class], k=1)
        n = np.random.choice(np.where(train_data[:,2] == str(n_column[0]))[0],1)[0]                           # Get random id from another class
         
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

print("Training finished")
print("Max_value_epoch: " + str(max_acc_epoch))
    
