# Tensorflow for semantic segmentation. 

Tensorflow is structured in it`s components:

* graph.py - Builds the static graph of tensorflow
* main.py - Main file which calls Train, Eval or Predict and loads config files.
* data.py - Data class which loads the dataset and provides data for the machine learning algorithm
* train.py - Contains the train loop for each epoch
* nnUtils.py - Provides builders for the different layers e.g. pooling layer
* evaluate.py - Takes test data and evaluates the trained model for metrics in metricsSemseg.py
* predict.py - Takes one image("predict.png") and uses the trained model to create the segmentated  image

I recreated different network architectures from papers descriptions:

* [denseNet.py](https://arxiv.org/pdf/1611.09326.pdf)
* [dilNet.py](https://arxiv.org/pdf/1511.07122.pdf)
* [refineNet.py](http://openaccess.thecvf.com/content_cvpr_2017/papers/Lin_RefineNet_Multi-Path_Refinement_CVPR_2017_paper.pdf)
* [segNet.py](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7803544)
* [uNet.py](https://arxiv.org/pdf/1505.04597.pdf)


Usage:

```
python3 main.py <MODE> <NeuralNetwork>
```

Whereas MODE can be either train, predict or eval. The names of the neural networks are the filename without the .py extension. Information about the modes:
* train - trains the current neural network with the current dataset
* predict - creates an segmented image with the name "predict.png" and outputs it in the root folder
* eval - evaluates the test split of the dataset with the metrics in metricsSemSeg.py
* serialize - serializes the dataset(pre-processed) into an numpy object. This object can be loaded into memory thus accelerating the training process immensely. Works only when in config the attribute "serializedObject" is set to false otherwise it tries to read a serialized object.

Example call:

```
python3 main.py train refineNet
```

The model is saved into a folder named models which is two directories above relative to the main file. The same for the tensorflow log files which are located in ../../logs and the dataset folder in ../../data.

Main JSON config file:

* batchSize: Batchsize for every train round
* steps: How many steps each epoch should at least be done. Is only effective when smaller then dataset,
* dataset: A name for the dataset,
* classes: How many classes are to be differentiated with the model,
* neuralNetwork: Name of the neural network,
* learningRate: Float for the learning rate,
* epochs: How many epochs the model should be trained with,
* gpu: GPU number

Dataset JSON config file:

* name: Name of the dataset,
* size: Int, size of the dataset,
* trainSize: float, percentage for how big the train size should be e.g. 95% of the dataset -> 0.95,
* testSize: float, percentage for the test size
* x: int, x value of the to be used image,
* y: int, y value of the to be used image,
* imageChannels: int, how many channels does the input image have,
* preProcessedPath": Path of already pre-processed images,
* downsize: String, "True" when the images should be downsized, "False" when not,
* classes : How many classes are to be differentiated with the model,
* path: Relative path to the dataset foler,
* serializedObject: Wether the dataset is serialized into a numpy object,
* fileName: Filename of the serialized object,
* images: Subfolder of the train input images,
* labels: Subfolder of the label input images(Ground-truth),
* ClassToRGB: Int Array, where the index is the class and the value and RGB array that is associated with this class
 