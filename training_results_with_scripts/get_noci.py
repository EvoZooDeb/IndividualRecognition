import pickle
import cv2
import os.path
from os import path as p
from random import shuffle

# input file
# Gerle
#FILE = "/home/dkatona/results/results_gerle_21_04_26.pkl"

# Noci
#FILE = "/home/dkatona/results/results_noci_21_04_26_1.pkl"
#FILE = "/home/dkatona/results/results_noci_21_04_26_2.pkl"

# Vadoc
#FILE = "/home/dkatona/results/results_vadoc_21_04_26.pkl"

# Huba
#FILE="/home/dkatona/results/results_huba_873_30.pkl"
FILE="/home/dkatona/results/results_huba_874_71.pkl"
# FILE="/home/dkatona/results/results_huba_875_200.pkl"
# FILE="/home/dkatona/results/results_huba_876_151.pkl"
# FILE="/home/dkatona/results/results_huba_877_161_p1.pkl"
# FILE="/home/dkatona/results/results_huba_877_200_p2.pkl"
# FILE="/home/dkatona/results/results_huba_877_200_p3.pkl"
# FILE="/home/dkatona/results/results_huba_878_141_p1.pkl"
# FILE="/home/dkatona/results/results_huba_878_200_p2.pkl"
# FILE="/home/dkatona/results/results_huba_878_200_p3.pkl"
# FILE="/home/dkatona/results/results_huba_879_25.pkl"
# FILE="/home/dkatona/results/results_huba_880_81.pkl"
# FILE="/home/dkatona/results/results_huba_881_115_p1.pkl"
# FILE="/home/dkatona/results/results_huba_881_150_p3.pkl"
# FILE="/home/dkatona/results/results_huba_881_150_p5.pkl"
# FILE="/home/dkatona/results/results_huba_881_200_p2.pkl"
# FILE="/home/dkatona/results/results_huba_881_210.pkl"
# FILE="/home/dkatona/results/results_huba_881_50_p4.pkl"
# FILE="/home/dkatona/results/results_huba_883_175_p1.pkl"
# FILE="/home/dkatona/results/results_huba_883_200_p2.pkl"
# FILE="/home/dkatona/results/results_plus_huba.pkl"

#FILE = "/home/wildhorse_project/results.pkl"
with open(FILE, 'rb') as fi:
    results = pickle.load(fi)
    
# available classes
classes = results['classes']
print("Available classes in the network:")
print(*classes, sep = ", ")

# Choose the object types you want to extract e.g. "dog"
class_filter = ["horse"]

#if p.exists("masks") is False:
#    !mkdir masks
#

background_color = (0,0,0,255)

# util method to convert the detectron2 box format
def xyxy_to_xywh(box):
    x1 = int(box[0])
    y1 = int(box[1])
    x2 = int(box[2])
    y2 = int(box[3])
    w = x2-x1
    h = y2-y1
    return [x1,y1,w,h]

# Cut out masks
print("Extracting objects...")
index = 0
for path in results['instances']:
    tensor_array = results['instances'][path].pred_boxes.tensor.cpu().numpy()
    # Get the rectangle with the largest area
    counter = 0
    max_index = 0
    for i in tensor_array:
        area = (i[2]-i[0])*(i[3]-i[1])
        if counter == 0:
            max = area
            max_index = counter
        elif max < area:
            max = area
            max_index = counter
        counter+=1
    
    mask = results['instances'][path].pred_masks[max_index]
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
    
    x=0
    y=0
    for line in mask:
        for column in line:
            if not column:
                img[x,y] = background_color
            y+=1
        y=0
        x+=1
                
    # Cropping image to the size of the objects bounding box
    box = results['instances'][path].pred_boxes[max_index]
    box = box.tensor.cpu().numpy()[0]
    box = xyxy_to_xywh(box)
    img = img[box[1]:box[1]+box[3], box[0]:box[0]+box[2]]
    
    path_array = path.split('_')
    array_length = len(path_array)
    exact_video_name = path_array[array_length-2]
    exact_frame_index = path_array[array_length-1][:-4]
    new_img_path = '/home/wildhorse_project/detectron_pic/black/huba/huba_' + str(exact_video_name) + "_" +str(exact_frame_index) + ".png"
    cv2.imwrite(new_img_path,img)
    print("Removed background from '" + path+"'. Saved object in '" + new_img_path + "")
    index+=1

print("Done...")