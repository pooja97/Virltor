import cv2
import numpy as np 
from glob import glob
import os

'''
Creating a location variable for the data folder and 
Creating a list of image folders present in the data directory using glob
'''

image_dir_loc = r'/Users/sheshmani/Desktop/virltor/data_dir'
image_folders = glob(os.path.join(image_dir_loc,'[0-6]'))


# Creating location variable for the data preprocessing directory. 
processed_dir =  r'/Users/sheshmani/Desktop/virltor/data_preprocessing'

'''
checking if folders exists for images in the preprocess directory or not. 
If exists do nothing, if not then create a directory at the specified location with the specified name. 
'''
for i in range(len(image_folders)):
    processed_image_path = os.path.join(processed_dir,str(i)) # Creating folder path for storing the processed images in the preprocessing directory. 
    if os.path.exists(processed_image_path):
        pass
    else:
        os.mkdir(processed_image_path)

    # Creating path variable for each image folder
    image_folder_path = os.path.join(image_dir_loc,str(i))

    '''
    Creating a loop for processing each image separately. 
    Fetching the file name using os.path.basename to store the latitude and longitude information
    '''
    for images in os.listdir(image_folder_path):
        file_name = os.path.basename(images) 
        img = cv2.imread(os.path.join(image_folder_path,images))
        img = img[295:1080,1100:]
        cv2.imwrite(os.path.join(processed_image_path,file_name), img)
cv2.waitKey(0)
cv2.destroyAllWindows() 





