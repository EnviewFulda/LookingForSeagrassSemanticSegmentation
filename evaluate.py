import tensorflow as tf 
import sys
import numpy as np 
import csv 
from PIL import Image
from LookingForSeagrassSemanticSegmentation.metricsSemSeg import pixel_accuracy, mean_accuracy, mean_IU, frequency_weighted_IU

# simple test function, tests all images in testdata once
def evaluate(sess, config, data, graph, scr, setProgress = False):
    totalCorrect = 0
    totalCount = data.config["testSize"]*data.config["x"]*data.config["y"]*data.config["imageChannels"]

    totalPAcc = []
    totalMAcc = []
    totalMIU = []
    totalFWIU = []

    i = 0
    
    iterator = graph["preFetchIterators"][2]
    nextImgData = iterator.get_next()

    evalSize = int(data.config["testSize"]/config["batchSize"])
    scr.logger.info("EvalSize")
    scr.logger.info(evalSize)
    for r in range(evalSize):
        if(setProgress):
            scr.update_progress((r/evalSize)*100)
        imgData  = sess.run(nextImgData)
        if imgData[0].shape[0] == config["batchSize"]:
            feed_dict = {
                graph["imagePlaceholder"]: imgData[0]
            }

            pred = sess.run(graph["prediction"], feed_dict=feed_dict)
            labels = imgData[1]
            for b in range(config["batchSize"]):
                predClasses = np.squeeze(pred[b])
                labelData = np.squeeze(labels[b])
                
                if i % 200 == 0:
                    print("Image ", i, " evaluated...")
                    #scr.logger.info("Image " +  str(i) +  " evaluated...")

                totalPAcc.append(pixel_accuracy(predClasses, labelData))
                totalMAcc.append(mean_accuracy(predClasses, labelData))
                totalMIU.append(mean_IU(predClasses, labelData))
                totalFWIU.append(frequency_weighted_IU(predClasses, labelData))
               
                i = i+1
    totalPAcc = np.mean(np.array(totalPAcc))
    totalMAcc = np.mean(np.array(totalMAcc))
    totalMIU = np.mean(np.array(totalMIU))
    totalFWIU = np.mean(np.array(totalFWIU))

    print("Pixel accuracy: ", totalPAcc ," || Mean accuracy: ", totalMAcc ," || Mean intersection union:", totalMIU ," || frequency weighted IU: ", totalFWIU)
    scr.logger.info("Pixel accuracy: " + str(totalPAcc)  +" || Mean accuracy: " +  str(totalMAcc)  +" || Mean intersection union:" + str(totalMIU) +" || frequency weighted IU: " + str(totalFWIU))
    
    eval_status = {
        "iteration": scr.iteration,
        "pixel_acc": str(totalPAcc),
        "mean_acc": str(totalMAcc),
        "meanIoU": str(totalMIU),
        "frequency_weighted_IU": str(totalFWIU)

    }
    return totalMIU, eval_status