import tensorflow as tf 
import sys
import numpy as np 
import csv
import cv2
from PIL import Image
from IPython.display import display 
from metricsSemSeg import pixel_accuracy, mean_accuracy, mean_IU, frequency_weighted_IU
import time

# predicts an image and save the result in same directory with the NNs name
# default is the image under results/predict.jpg

def predict(sess, config, data, graph, filePath = "../results/", fileName = "predict.jpg" ):

    startTotal = time.time()
    
    imagePath = filePath + fileName
    print("Predicting ", imagePath)
    img = cv2.imread(imagePath)  
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    imgRes = cv2.resize(img, (data.config["x"], data.config["y"]), interpolation=cv2.INTER_NEAREST)
    imgRes = (imgRes - imgRes.mean()) / imgRes.std()
    
    inputData = np.expand_dims(imgRes, axis=0)
    
    if config["batchSize"] > 1:
        fillerArr = np.zeros((1,data.config["y"], data.config["x"], data.config["imageChannels"]))
        for x in range(config["batchSize"]-1):
            inputData = np.concatenate((inputData, fillerArr), axis=0)
  
    feed_dict = {
            graph["imagePlaceholder"]: inputData 
        }
                       
    startPrediction = time.time()
    predClasses = sess.run(graph["prediction"], feed_dict=feed_dict)
    endPrediction = time.time()
    predClasses = predClasses[0].reshape(data.config["x"]*data.config["y"])
    predImg = np.zeros((data.config["x"]*data.config["y"],3))


    for idx, p in enumerate(predClasses):
        predImg[idx] = data.config["ClassToRGB"][p]

   
    predImg = predImg.reshape((data.config["y"], data.config["x"], data.config["imageChannels"])).astype("uint8")
    endTotal = time.time()
    predTime = endPrediction - startPrediction
    totalTime = endTotal - startTotal
    print(" Total time: ", totalTime, " only inference time: ", predTime)
    savePath = "../results/"+data.config["name"]+str(data.config["x"])+str(data.config["y"])+config["neuralNetwork"]+".png"
    savedImage = Image.fromarray(predImg, "RGB")
    savedImage.save(savePath)
    
    return savedImage
    
    # blend image onto original image
    #predImg[((predImg == 255).any(-1))] = [64, 255, 0]
    #predImg = Image.fromarray(predImg, "RGB")
    #imgRes = Image.fromarray(imgRes, "RGB")
    #predImg.convert("RGBA")
    #imgRes.convert("RGBA")
    #blendImg = Image.blend(imgRes, predImg, 0.3)
    #display(blendImg)
  