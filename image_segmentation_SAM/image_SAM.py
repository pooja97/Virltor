# import torch
import matplotlib.pyplot as plt
import matplotlib
import cv2
import numpy as np
from glob import glob
import os
import sys
from tqdm import tqdm

sys.path.append("..")
from segment_anything import SamPredictor, sam_model_registry

sam_checkpoint = '/Users/sheshmani/Desktop/virltor/image_segmentation_SAM/sam_vit_h_4b8939.pth'
device = 'cuda'
model_type = 'default'

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
# sam.to(device=device)
predictor = SamPredictor(sam)



def show_mask(mask, ax):
    color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))   


'''
Creating a location variable for the data folder and 
Creating a list of image folders present in the data directory using glob
'''

image_directory_location = r'/Users/sheshmani/Desktop/community_rating/data_preprocessing'
image_folders_location = glob(os.path.join(image_directory_location,'[1-7]'))

print(image_folders_location)
'''
Creating a directory variable for storing the segmentation result
'''
segmentation_directory =  r'/Users/sheshmani/Desktop/community_rating/image_segmentation_SAM'


'''
checking if folders exists for images in the segmentation directory or not. 
If exists do nothing, if not then create a directory at the specified location with the specified name. 
'''
for i in range(6,len(image_folders_location)+1):
    segmented_image_path = os.path.join(segmentation_directory,str(i)) # Creating folder path for storing the processed images in the preprocessing directory. 
    if os.path.exists(segmented_image_path):
        pass
    else:
        os.mkdir(segmented_image_path)

    # Creating path variable for each image folder
    image_folder_path = os.path.join(image_directory_location,str(i))

    '''
    Creating a loop for processing each image separately. 
    Fetching the file name using os.path.basename to store the latitude and longitude information
    '''
    for images in os.listdir(image_folder_path):
        file_name = os.path.basename(images) 
        img = cv2.imread(os.path.join(image_folder_path,images))
        image = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

        predictor.set_image(image)

        input_point = np.array([[920,700]])
        input_label = np.array([1])

        masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=False,
    )
        
        for i, (mask, score) in enumerate(zip(masks, scores)):
            plt.figure(figsize=(15,15))
            plt.imshow(image)
            show_mask(mask, plt.gca())
            show_points(input_point, input_label, plt.gca())
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
            plt.axis('off')
            plt.savefig(os.path.join(segmented_image_path,file_name))  
            matplotlib.pyplot.close()


        # img = img[295:1080,1100:]
        # cv2.imwrite(os.path.join(segmented_image_path,file_name), img)
cv2.waitKey(0)
cv2.destroyAllWindows() 

