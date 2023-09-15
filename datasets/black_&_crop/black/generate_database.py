# The results will be shown in "individual_path" folder

import pandas as pd
import os
import csv

# MUST MODIFY VARIABLES
individual_path = "/home/wildhorse_project/detectron_pic/black/drone"
json_path = "/home/wildhorse_project/detectron_pic/black/all_individual.json"
Type = "black"
Recording = "drone"

# Read individual json files
json_df = pd.DataFrame(pd.read_json(json_path,typ='series'), columns=['Id'])
json_df['Name'] = json_df.index.str.lower()
json_df = json_df.set_index(['Id'])

list_of_files = os.listdir(individual_path)
for directory in list_of_files:
    list_of_dirs = ['Path,Individual,ID,Type,Recording']
    individual_name = directory
    directory = individual_path + "/" + directory
    if os.path.isdir(directory):
        ID = str(json_df.loc[json_df['Name'] == individual_name].index[0])
        file_list = os.listdir(directory)
        for files in file_list:
            list_of_dirs.append(directory + "/" + files + "," + individual_name + "," + ID + "," + Type + "," + Recording)
        with open(individual_path + "/" + individual_name + ".csv", 'w', newline='') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_NONE, delimiter='\n', quotechar='')
            wr.writerow(list_of_dirs)
