import tensorflow as tf 
import time 
import numpy as np 
import datetime
import cv2 

from LookingForSeagrassSemanticSegmentation.predict import predict

# one training epoch and validation afterwards
def doTrain(epoch, sess, graph, config, data, modelFileName, scr):
    training_status_list = [

    ]



    accuracySum = 0
    step = 1
    trainSize = int(data.config["trainSize"]/config["batchSize"])
    
    loss = []
    train_acc = []
    acc = []

    feed_dict = {}
    
    iterator = graph["preFetchIterators"][0]
    nextImgData = iterator.get_next()
    for batchIdx in range(trainSize):

        start = time.time()

        # images that create error which sets the loss to NaN
        faultyVals = list()

        if step not in faultyVals:

            if data.config["tfPrefetch"]:
                try:
                    
                    imgData  = sess.run(nextImgData)
                    if imgData[0].shape[0] == config["batchSize"]:
                        _imageData = imgData[0]
                        _labelData = imgData[1]
                    else:
                        break
                except tf.errors.OutOfRangeError:
                    break
            else:
                batch = next(data.getNextBatchTrain(config["batchSize"], step*epoch))
                _imageData = np.expand_dims(batch[1], axis=3) if data.config["imageChannels"] == 1 else batch[1]
                _labelData = batch[0]

            feed_dict = {
                graph["imagePlaceholder"] : _imageData,
                graph["labelPlaceholder"] : _labelData
            }

            graph["trainOp"].run(feed_dict=feed_dict)

            end = time.time() 
          
        else:
            print("SKIPPED INVALID STEP: ", step)  
            scr.logger.info("SKIPPED INVALID STEP: " + str(step))  
        
        # every 10 steps print training status
        # if step % int(trainSize/10) == 0:
        if ((step % int(trainSize/2)) == 0):
            summary, _loss, _train_acc = sess.run([graph["mergedLog"], graph["loss"], graph["accuracy"]], feed_dict=feed_dict)


            train_acc.append(_train_acc*100)
            loss.append(_loss)
            graph["logWriter"].add_summary(summary, step)
            status = "Epoch : "+str(epoch)+" || Step: "+str(step)+"/"+ str(trainSize)
            status += " || loss:"+str(round(np.mean(np.array(loss)), 5))+" || train_accuracy:"+ str(round(np.mean(np.array(train_acc)), 5))
            status += "% || ETA: "+str(datetime.timedelta(seconds=((end-start)*((trainSize)-step))))
            # ends with \r to delete the older line so the new line can be printed
            print(status, end="\r")              
            scr.logger.info("TRAINING STATUS")          
            scr.logger.info(status)
            training_status_list.append(
                {
                    "step": str(step),
                    "loss": str(round(np.mean(np.array(loss)), 5)),
                    "train_accuracy": str(round(np.mean(np.array(train_acc)), 5)),
                    "ETA": str(datetime.timedelta(seconds=((end-start)*((trainSize)-step))))
                }
            )
     
        if step >= trainSize:
            break

        step+=1
    

    # validate trained model after one epoch
    iterator = graph["preFetchIterators"][1]
    nextImgData = iterator.get_next()
    valSize = int(data.config["validationSize"]/config["batchSize"])
    for r in range(valSize):

        imgData  = sess.run(nextImgData)
        if imgData[0].shape[0] == config["batchSize"]:
            feed_dict={
                graph["imagePlaceholder"]: np.expand_dims(imgData[0], axis=3) if data.config["imageChannels"] == 1 else imgData[0],
                graph["labelPlaceholder"]: imgData[1] if data.config["imageChannels"] == 1 else imgData[1]
            }

            _acc = 100*(graph["accuracy"].eval(feed_dict=feed_dict))    
            acc.append(_acc)

    accuracy = round(np.mean(np.array(acc)), 3)
    print("\nvalidation_accuracy: "+str(accuracy))
    scr.logger.info("\nvalidation_accuracy: "+str(accuracy))
    return accuracy, round(np.mean(np.array(loss)), 5), training_status_list