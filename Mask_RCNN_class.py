#JSS
# To do:
#1) set the paths properly
#2) make  --done
import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
from pathlib import Path

import Mask_RCNN.model as modellib
from Mask_RCNN import utils
from Mask_RCNN import coco
COCO_MODEL_PATH = Path("/home/Mask_RCNN/mask_rcnn_coco.h5")
MODEL_DIR = Path("/home/Mask_RCNN/logs")


# Load a random image from the images folder
os.environ["CUDA_VISIBLE_DEVICES"] = '4'
class Model_RCNN:
    def __init__(self):
        print("Initilaized RCNN")
        print("GPU 4")
        self.model_create()  
    def model_create(self):
        with tf.device('/GPU:4'):
            class InferenceConfig(coco.CocoConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
                GPU_COUNT = 1
                IMAGES_PER_GPU = 1

            config = InferenceConfig()
            # config.display()
            self.model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
            self.model.load_weights(COCO_MODEL_PATH, by_name=True)
    def rcnn_final (self,image):
        self.image = image
        self.results = self.model.detect([self.image], verbose=0)
        r = self.results[0]
        self.indexs = np.where(r["class_ids"]==1)[0]   
        w = r["masks"][:,:,self.indexs]
        w_new = np.sum(w, axis = 2)
        w_new = w_new[...,np.newaxis]
        dilation_size =15 ## hyper parameter
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*dilation_size+1, 2*dilation_size+1), (dilation_size, dilation_size))
        w_new = cv2.dilate(w_new.astype(np.uint8), kernel)
        w_conv = np.ones(np.shape(w_new))
        w_new = w_conv - w_new 
        w_new = w_new.astype(np.uint8)
        trail = cv2.bitwise_and(self.image, self.image, mask = w_new)
        
        
        return trail
        
    def rcnn_only_mask (self,image):
        self.image = image
        self.results = self.model.detect([self.image], verbose=0)
        r = self.results[0]
        self.indexs = np.where(r["class_ids"]==1)[0]   
        w = r["masks"][:,:,self.indexs]
        w_new = np.sum(w, axis = 2)
        w_new = w_new[...,np.newaxis]
        dilation_size =15 ## hyper parameter
        print("dilation_size_is ",dilation_size)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*dilation_size+1, 2*dilation_size+1), (dilation_size, dilation_size))
        w_new = cv2.dilate(w_new.astype(np.uint8), kernel)
        return w_new

## send RGB image
    def rcnn_results_func(self, image):
        self.image = image

        self.results = self.model.detect([self.image], verbose=0)
        r = self.results[0]
        image_empty = np.zeros((np.shape(image)[0], np.shape(image)[1],1))
        for i in range(np.shape(self.image)[0]):
            for j in range(np.shape(self.image)[1]):
                image_empty[i,j,:]= np.max(r["masks"][i,j,:])
        dilation_size =15 ## hyper parameter
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*dilation_size+1, 2*dilation_size+1), (dilation_size, dilation_size))
        image_empty = cv2.dilate(image_empty, kernel)
        image_empty = image_empty[:,:,np.newaxis]
        self.image[:,:,:] = np.where(image_empty==1, 0, self.image[:,:,:])
        return self.image
    
# a function to select only a few out of the original indexs    
## send RGB image ## can make this function more robust
    def rcnn_mask_person(self, image):
        self.image = image
        self.results = self.model.detect([self.image], verbose=0)
        r = self.results[0]
        image_empty = np.zeros((np.shape(image)[0], np.shape(image)[1],1))
        if 1 in r['class_ids']:
            print("class id",r['class_ids'])
            self.indexs = np.where(r["class_ids"]==1)[0]            
            for i in range(np.shape(self.image)[0]):
                for j in range(np.shape(self.image)[1]):
                    print("indexs",self.indexs)
                    print("mask shape",np.shape(r["masks"]))
                    r["masks"] = r["masks"][:,:, self.indexs]
                    image_empty[i,j,:]= np.max(r["masks"][i,j,:])
            dilation_size =15 ## hyper parameter
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*dilation_size+1, 2*dilation_size+1), (dilation_size, dilation_size))
            image_empty = cv2.dilate(image_empty, kernel)
            image_empty = image_empty[:,:,np.newaxis]
            self.image[:,:,:] = np.where(image_empty==1, 0, self.image[:,:,:])
        return self.image