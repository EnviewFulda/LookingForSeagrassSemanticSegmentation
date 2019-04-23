import warnings
warnings.filterwarnings('ignore')

import cv2 
import numpy as np 
import csv 
import tensorflow as tf
import os
import matplotlib as mpl
#mpl.use('TkAgg')
import matplotlib.pyplot as plt

plt.style.use('seaborn-white')

def sanityCheckMain(sess, config, data, graph):
    
    x_valid = []
    y_valid = []
    x_preds = []
    y_preds = []
    
    print("loading data...")
    
    
    iterator = graph["preFetchIterators"][2]
    testSize = int(data.config["testSize"]/config["batchSize"])
    if testSize > 60:
        testSize = 60
        
    imgNextData = iterator.get_next()
        
    for r in range(testSize):

        imgData  = sess.run(imgNextData)
        if imgData[0].shape[0] == config["batchSize"]:
            feed_dict = {
                graph["imagePlaceholder"]: imgData[0]
            }

            pred = graph["softmaxOut"].eval(feed_dict=feed_dict)
            pred = np.argmax(pred, axis=3)
            labels = imgData[1]
     
            for b in range(config["batchSize"]):
                x_preds.append(pred[b].squeeze())
                x_valid.append(imgData[0][b].squeeze())
                y_valid.append(imgData[1][b].squeeze())
                
    
    x_valid = np.array(x_valid)
    y_valid = np.array(y_valid)
    x_preds = np.array(x_preds)
    y_preds = np.array(y_preds)
    
    print(x_valid.shape, y_valid.shape, x_preds.shape)
    sanityCheck(x_valid, y_valid, x_preds)

# Source https://www.kaggle.com/bguberfain/unet-with-depth
def sanityCheck(x_valid, y_valid, preds_valid):
    print("Sanity Check")
    # display ground-truth
    max_images = 60
    grid_width = 15
    grid_height = int(max_images / grid_width)
    fig, axs = plt.subplots(grid_height, grid_width, figsize=(grid_width, grid_height))
    for idx, i in enumerate(x_valid[:max_images]):
        img = (x_valid[idx] * 255).astype(np.uint8)
        mask = (y_valid[idx]  * 255).astype(np.uint8)
        ax = axs[int(idx / grid_width), idx % grid_width]
        #ax.imshow(img, cmap="Greys")
        ax.imshow(mask, alpha=0.6, cmap="Greens")
        #ax.imshow(pred, alpha=0.6, cmap="OrRd")
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        
  
    plt.suptitle("Green: ground truth")
    plt.show()
    
    #display predictions
    max_images = 60
    grid_width = 15
    grid_height = int(max_images / grid_width)
    fig, axs = plt.subplots(grid_height, grid_width, figsize=(grid_width, grid_height))
    for idx, i in enumerate(x_valid[:max_images]):
        img = (x_valid[idx] * 255).astype(np.uint8)
        pred = (preds_valid[idx] * 255).astype(np.uint8)
        ax = axs[int(idx / grid_width), idx % grid_width]
        #ax.imshow(img, cmap="Greys")
        #ax.imshow(mask, alpha=0.6, cmap="Greens")
        ax.imshow(pred, alpha=0.6, cmap="OrRd")
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        
        
    plt.suptitle("Red: prediction")
    plt.show()

