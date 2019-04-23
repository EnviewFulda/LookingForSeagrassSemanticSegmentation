# Semantic Segmentation with Tensorflow
# by Franz Weidmann www.github.com/Aequalitas

import tensorflow as tf
import os
import numpy as np
import sys
import json


from data import Data
from graph import buildGraph
from train import doTrain
from predict import predict
from evaluate import evaluate
from sanityCheck import sanityCheckMain


def deepSS(MODE, networkName, GPUNr="0"):

    if MODE == 'train' or MODE == 'predict' or MODE == 'eval' or MODE == 'serialize' or MODE == "sanityCheck":
        print("MODE: ", MODE)
    else:
        raise Exception("Provide one argument: train, eval, predict, runLenEncode, classWeights or serialize!")


    #load config for tensorflow procedure from json
    config = json.load(open("nets/"+networkName+"Config.json"))
    # load data object initially which provides training and test data loader
    data = Data("../data/"+config["dataset"]+"/configData"+config["dataset"]+".json")
    
    if MODE == "classWeights":
        data.getClassWeights("Freq")
    elif MODE == "serialize":
        print("Serializing dataset to ",data.config["path"]+data.config["fileName"])

        if data.config["fileName"] != "":
            np.save(data.config["path"]+data.config["fileName"], data.getDataset(flipH=False))
            print("Finished serializing!")
        else:
            print("You have to set a filename for the serialized file in the config file!")
        
    else:
        # create the tensorflow graph and logging
        graph = buildGraph(data, config, "train")


        os.environ["CUDA_VISIBLE_DEVICES"]=GPUNr
        tf.logging.set_verbosity(tf.logging.INFO)
        tfconfig = tf.ConfigProto()
        tfconfig.gpu_options.allow_growth = True
        #config.gpu_options.per_process_gpu_memory_fraction = 0.3

        with tf.Session(config=tfconfig) as sess:

            sess.run(tf.global_variables_initializer())
            modelFileName = "../models/"+str(data.config["depth"])+"meters/FlippedH/model"+str(data.config["x"])+str(data.config["y"])+data.config["name"]+config["neuralNetwork"]+"Batch"+str(config["batchSize"])+".ckpt"
            
            try:
                graph["saver"].restore(sess, modelFileName)
            except:
                print("No valid checkpoint found")

            history = [[],[],[]]

            if MODE == "train":
                print("Starting training...")

                best_acc = 0
                LRcounter = 0
                LRcorrectionCounter = 0
                bestMeanIoU = 0
                for e in range(1, config["epochs"]+1):
                    curr_acc, loss = doTrain(e, sess, graph, config, data, modelFileName)
                    history[0].append(curr_acc)
                    history[1].append(loss)
                    
                    predict(sess, config, data, graph)
                    
                    if best_acc < curr_acc:
                        
                        print("val acc of ", curr_acc, " better than ", best_acc)
                        best_acc = curr_acc
                        LRcounter = 0
                        LRcorrectionCounter = 0
                        #graph["saver"].save(sess, modelFileName)
                    else:
                        print("val acc of ", curr_acc, " NOT better than ", best_acc)
                        if LRcounter >= 3:
                            lr = graph["learningRate"].eval()
                            graph["learningRate"] = tf.assign(graph["learningRate"], lr*0.1)
                            print("Learning rate of ", lr ," is now decreased to ", lr * 0.1)
                            LRcounter = 0
                            if LRcorrectionCounter >= 10:
                                break
                                
                        LRcounter = LRcounter + 1
                        LRcorrectionCounter = LRcorrectionCounter + 1
                                        
                    meanIoU = evaluate(sess, config, data, graph)
                    history[2].append(meanIoU)
                    if bestMeanIoU < meanIoU:
                        print("meanIoU of ", meanIoU, " better than ", bestMeanIoU, " Saving model...")
                        graph["saver"].save(sess, modelFileName)
                        bestMeanIoU = meanIoU

                graph["saver"].save(sess, modelFileName+"END")
                return history
                
            elif MODE == "eval":
                evaluate(sess, config, data, graph)
            elif MODE == "predict":
                predict(sess, config, data, graph)
            elif MODE == "sanityCheck":
                sanityCheckMain(sess, config, data, graph)
           
