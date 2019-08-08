import tensorflow as tf 
import importlib
import numpy as np
#from netFCN import net
from LookingForSeagrassSemanticSegmentation.nets.deeplabV3plusSS import net

def buildGraph(data, config, MODE, scr):
    # net_path = scr.get_path("LookingForSeagrassSemanticSegmentation/nets/deeplabV3plusSS")
    # net = importlib.import_module("nets."+config["neuralNetwork"]).net

    # REAL TENSORFLOW - low API

    # Main Variables
    # global step for decaying learning rate
    global_step = tf.Variable(0, trainable=False)

    # create placeholder later to be filled
    imageShape = [config["batchSize"], data.config["y"], data.config["x"], data.config["imageChannels"]]
    image = tf.placeholder(tf.float32, shape=imageShape, name="input_image")

    # has to be reshaped in case output resolution is smaller as in the unet
    labelsShape = [config["batchSize"], data.config["y"], data.config["x"]]
    labels = tf.placeholder(tf.int32, labelsShape, name="labels")

    # Neural Network
    logits, predictionNet, softmaxNet = net(image, data.config["classes"], MODE)

    if MODE == "train":
        # Training
        # sparse because one pixel == one class and not multiple
        loss = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits))

        tf.summary.scalar("loss", loss)

        # optimizer
        LR = tf.Variable(config["learningRate"], name="learningRate")
        tf.summary.scalar("learning_rate", LR)
        optimizer = tf.train.AdamOptimizer(learning_rate=LR, name="AdamOpt")
        train_op = optimizer.minimize(loss, global_step=global_step)

        correct_prediction = tf.equal(tf.cast(predictionNet, tf.int32), labels)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        tf.summary.scalar("accuracy", accuracy)

    saver = tf.train.Saver()
    merged = tf.summary.merge_all()

    writer = tf.summary.FileWriter("../logs/", graph=tf.get_default_graph())

    if data.config["tfPrefetch"]:
        with tf.device('/cpu:0'):
            # tensorflow dataset for a more efficient input pipeline through threading
            iterators = []
            for _type in ["train", "validation", "test"]:
                
                print("Creating ", _type, " dataset...")
                scr.logger.info("Creating " + str(_type) +  " dataset...")
                imageFilenames = tf.constant(data.imageData[_type])
                labelsFileNames = tf.constant(data.imageData[_type+"Label"])

                dataset = tf.data.Dataset.from_tensor_slices((imageFilenames, labelsFileNames))
                dataset = dataset.map(lambda filename, label: tf.py_func(
                                              data.getImageTuple,
                                              [filename, label],
                                              [tf.float32, tf.uint8]
                                           ),  num_parallel_calls=config["threadCount"])

                if _type == "train":
                    # data augmentation
                    #datasetFlippedV = dataset.map(lambda trainImage, labelImage:
                    #                             (tf.reverse(trainImage, axis=[1]), tf.reverse(labelImage, axis=[1]))
                    #                           , num_parallel_calls=config["threadCount"])
                    #dataset = dataset.concatenate(datasetFlippedV)

                    datasetFlippedH = dataset.map(lambda trainImage, labelImage:
                                                  (tf.reverse(trainImage, axis=[0]), tf.reverse(labelImage, axis=[0]))
                                               , num_parallel_calls=config["threadCount"])

                    dataset = dataset.concatenate(datasetFlippedH)
                    data.config[_type+"Size"] *= 2
                    
                    print("Dataset flipped vertically new ", _type, "Size: ", data.config[_type+"Size"])

                if _type == "train":
                    dataset = dataset.shuffle(buffer_size=int(1000/config["batchSize"]))
                
                dataset = dataset.batch(config["batchSize"])
                dataset = dataset.prefetch(5)
                dataset = dataset.repeat(config["epochs"])
                iterators.append(dataset.make_one_shot_iterator())

    
    if MODE == "train":
        return {
            "logits":logits,
            "loss": loss,
            "mergedLog": merged,
            "learningRate": LR,
            "prediction": predictionNet,
            "softmaxOut": softmaxNet,
            "imagePlaceholder": image,
            "labelPlaceholder": labels,
            "preFetchIterators": iterators,
            "trainOp": train_op,
            "saver": saver,
            "logWriter": writer,
            "accuracy": accuracy
        }
    else:
        return {
            "logits":logits,
            "prediction": predictionNet,
            "softmaxOut": softmaxNet,
            "imagePlaceholder": image,
            "labelPlaceholder": labels,
            "preFetchIterators": iterators,
            "saver": saver,
            "logWriter": writer,
            }