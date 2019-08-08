# Semantic Segmentation with Tensorflow
# by Franz Weidmann www.github.com/Aequalitas

import tensorflow as tf
import os
import numpy as np
import sys
import json


from LookingForSeagrassSemanticSegmentation.data import Data
from LookingForSeagrassSemanticSegmentation.graph import buildGraph
from LookingForSeagrassSemanticSegmentation.train import doTrain
from LookingForSeagrassSemanticSegmentation.predict import predict
from LookingForSeagrassSemanticSegmentation.evaluate import evaluate
from LookingForSeagrassSemanticSegmentation.sanityCheck import sanityCheckMain

def find(lst, key, value):
    for i, dic in enumerate(lst):
        if dic[key] == value:
            return i
    return -1



def deepSS(MODE, networkName, GPUNr="0", 
scr=None, 
images_to_predict=None, 
models_path= None,
model_file_name = 'seagrass.ckpt',
images_path = None,
labeled_images_path = None,
train_json_path = None,
eval_json_path = None,
config_path = None,
net_config_path = None,
dataset_path = None,
iteration = None,
media_path = None,
SIA_result_path= None,
SIA_best_epochs_path = None,
result_predict_path= None
):

    if MODE == 'train' or MODE == 'predict' or MODE == 'eval' or MODE == 'serialize' or MODE == "sanityCheck":
        print("MODE: ", MODE)
        scr.logger.info("MODE: " +  str(MODE))
    else:
        raise Exception("Provide one argument: train, eval, predict, runLenEncode, classWeights or serialize!")

    if not os.path.exists(models_path + str(scr.iteration) + "/"):
        os.mkdir(models_path + str(scr.iteration) + "/")
    model_file_path = models_path + str(scr.iteration) + "/" + str(scr.iteration)+ "_"+ model_file_name   
    if(scr.iteration == 0):
        last_model_file_path = model_file_path
    else:
        last_model_file_path = (models_path + str(scr.iteration -1) + "/" + str(scr.iteration - 1)+ "_"+ model_file_name   )


    #load config for tensorflow procedure from json
    # config = json.load(open("nets/"+networkName+"Config.json"))
    config = json.load(open(net_config_path))
    # load data object initially which provides training and test data loader
    data = Data(
        config_path, 
        scr,
        train_json_path = train_json_path,
        labeled_images_path = labeled_images_path,
        eval_json_path = eval_json_path,
        images_path= images_path
        )
    # data = Data("../data/"+config["dataset"]+"/configData"+config["dataset"]+".json")
    if MODE == "classWeights":
        data.getClassWeights("Freq")
    elif MODE == "serialize":
        print("Serializing dataset to ",data.config["path"]+data.config["fileName"])
        scr.logger.info("Serializing dataset to " + data.config["path"]+data.config["fileName"])

        if data.config["fileName"] != "":
            np.save(data.config["path"]+data.config["fileName"], data.getDataset(flipH=False))
            print("Finished serializing!")
            scr.logger.info("Finished serializing!")
        else:
            print("You have to set a filename for the serialized file in the config file!")
            scr.logger.info("You have to set a filename for the serialized file in the config file!")
        
    else:
        # create the tensorflow graph and logging
        graph = buildGraph(data, config, "train", scr)
        os.environ["CUDA_VISIBLE_DEVICES"]=GPUNr
        tf.logging.set_verbosity(tf.logging.INFO)
        tfconfig = tf.ConfigProto()
        tfconfig.gpu_options.allow_growth = True
        #config.gpu_options.per_process_gpu_memory_fraction = 0.3

        with tf.Session(config=tfconfig) as sess:
            sess.run(tf.global_variables_initializer())
            # model_file_path = "../models/"+str(data.config["depth"])+"meters/FlippedH/model"+str(data.config["x"])+str(data.config["y"])+data.config["name"]+config["neuralNetwork"]+"Batch"+str(config["batchSize"])+".ckpt"
            # model_file_path = scr.get_path("test.ckpt", context="pipe')

            #For Test
            # model_file_path = scr.get_path("test.ckptEND", context='pipe')
            try:
                if(MODE == 'train' or MODE =='predict'):
                    scr.logger.info("TRY LOADEDING CHECKPOINT: " + last_model_file_path)
                    graph["saver"].restore(sess, last_model_file_path)
                else:
                    scr.logger.info("TRY LOADEDING CHECKPOINT: " + model_file_path)
                    graph["saver"].restore(sess, model_file_path)
            except Exception as e:
                print("No valid checkpoint found")
                scr.logger.info("No valid checkpoint found")
                scr.logger.info(str(e))
            history = [[],[],[]]

            if MODE == "train":
                print("Starting training...")
                scr.logger.info("Starting training...")

                best_acc = 0
                LRcounter = 0
                LRcorrectionCounter = 0
                bestMeanIoU = 0


                epochs_data = []



                if os.path.exists(SIA_result_path):
                    with open(SIA_result_path, 'r') as outfile:
                        results = json.load(outfile)
                else:
                    results = []
                results.append({
                    "iteration" :iteration
                })                
                
                if os.path.exists(SIA_best_epochs_path):
                    with open(SIA_best_epochs_path, 'r') as outfile:
                        best_epochs = json.load(outfile)
                else:
                    best_epochs = []
                    
                best_epochs.append({
                    "iteration" :iteration
                })


                for e in range(1, config["epochs"]+1):
                    scr.update_progress(e)

                    

                    curr_acc, loss, training_status_list = doTrain(e, sess, graph, config, data, model_file_path, scr)
                    history[0].append(curr_acc)
                    history[1].append(loss)
                    
                    # predict(sess, config, data, graph, scr= scr)
                    # test_predict_path = scr.get_path("predict/predict.jpg", context="static")
                    # test_result_path = scr.get_path("predict/" + str(e) + ".png", context="static")
                    # predict(sess, config, data, graph, scr= scr, predict_image_path = test_predict_path, result_predict_path=test_result_path )




                    if best_acc < curr_acc:
                        
                        print("val acc of ", curr_acc, " better than ", best_acc)
                        scr.logger.info("val acc of " + str(curr_acc) + " better than " +  str(best_acc))
                        best_acc = curr_acc
                        LRcounter = 0
                        LRcorrectionCounter = 0
                        graph["saver"].save(sess, model_file_path)
                    else:
                        print("val acc of ", curr_acc, " NOT better than ", best_acc)
                        scr.logger.info("val acc of " +  str(curr_acc) + " NOT better than "+  str(best_acc))
                        if LRcounter >= 3:
                            lr = graph["learningRate"].eval()
                            graph["learningRate"] = tf.assign(graph["learningRate"], lr*0.1)
                            print("Learning rate of ", lr ," is now decreased to ", lr * 0.1)
                            scr.logger.info("Learning rate of " + str(lr) + " is now decreased to " +  str(lr * 0.1))
                            LRcounter = 0
                            if LRcorrectionCounter >= 10:
                                scr.logger.info("LRCorrectionCounter")
                                scr.logger.info(LRcorrectionCounter)
                                break
                                
                        LRcounter = LRcounter + 1
                        LRcorrectionCounter = LRcorrectionCounter + 1
                                        
                    meanIoU, eval_status = evaluate(sess, config, data, graph, scr)
                    


                    epoch_data ={
                        "epoch": str(e),
                        "eval": eval_status,
                        "training": training_status_list
                    }
                    epochs_data.append(epoch_data)

                    result_dict = {
                        "iteration": iteration,
                        "data": epochs_data
                    }
                    index = find(results, "iteration", iteration)
                    results[index] = result_dict

                    
                    # stats_list_path = scr.get_path("stat.json")

                    with open(SIA_result_path, 'w') as outfile:
                        json.dump(results, outfile, indent=1, ensure_ascii=True)


                    history[2].append(meanIoU)
                    if bestMeanIoU < meanIoU:
                        print("meanIoU of ", meanIoU, " better than ", bestMeanIoU, " Saving model...")
                        scr.logger.info("meanIoU of " +  str(meanIoU) +  " better than " +  str(bestMeanIoU) + " Saving model...")
                        graph["saver"].save(sess, model_file_path)
                        bestMeanIoU = meanIoU

                        index = find(best_epochs, "iteration", iteration)
                        iteration_dict = {
                            "iteration": iteration
                        }
                        best_epochs[index] = {**iteration_dict, **epoch_data}
                        with open(SIA_best_epochs_path, 'w') as outfile:
                            json.dump(best_epochs, outfile, indent=1, ensure_ascii=True)

                graph["saver"].save(sess, model_file_path+"END")
                return history
                
            elif MODE == "eval":                                    # evaluate
                return evaluate(sess, config, data, graph, scr, setProgress= True)
            elif MODE == "predict":
                pixel_map_images = []
                for i, anno in enumerate(images_to_predict):
                    scr.update_progress((i/len(images_to_predict) *100))
                    img_name = anno['image'].split('/')[-1]
                    img_path = os.path.join(media_path, img_name)
                    pixel_map = predict(sess, config, data, graph, scr= scr, predict_image_path= img_path, result_predict_path=result_predict_path)
                    pixel_map_images.append(pixel_map)
                return pixel_map_images
            elif MODE == "sanityCheck":                             # Test ob Graph richtig aufgebaut
                sanityCheckMain(sess, config, data, graph)
           
