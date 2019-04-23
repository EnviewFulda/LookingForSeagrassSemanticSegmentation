# LookingForSeagrass with Semantic Segmentation 
This is the code which corresponds to the Seagrass Semantic Segmentation paper <URL> which was puplished at the IEEE/OCENs conference 2019 in Marseille


# Documentation for usage
Tensorflow is structured in it`s components:

* graph.py - Builds the static graph of tensorflow
* main.py - Main file which calls Train, Eval or Predict and loads config files.
* data.py - Data class which loads the dataset and provides data for the machine learning algorithm
* train.py - Contains the train loop for each epoch
* nnUtils.py - Provides builders for the different layers e.g. pooling layer
* evaluate.py - Takes test data and evaluates the trained model for metrics in metricsSemseg.py
* predict.py - Takes one image and uses the trained model to create the segmentated image
* sanityCheck - Predicts 60 images from the test set and displays them. The corresponding ground-truth is also displayed.

Recreated network architectures from papers descriptions:

* [dilNet.py](https://arxiv.org/pdf/1511.07122.pdf)
* [uNet.py](https://arxiv.org/pdf/1505.04597.pdf)

Usage:

```
import main
main.deepSS(<MODE>, <Neural Network Name>)

```

Whereas MODE(first parameter) can be either train, predict, eval or sanityCheck. The names of the neural networks are the filename without the .py extension. Information about the modes:
* train - trains the current neural network with the current dataset
* predict - creates an segmented image with the name "predict.png" and outputs it in the root folder
* eval - evaluates the test split of the dataset with the metrics in metricsSemSeg.py

Example call:

```
import main
main.deepSS("eval","deeplabV3plusSS")

```

The model is saved into a folder named models which is two directories above relative to the main file. The same for the tensorflow log files which are located in ../../logs and the dataset folder in ../../data.

Main JSON config file:

* batchSize: Batchsize for every train round
* steps: How many steps each epoch should at least be done. Is only effective when smaller then dataset,
* dataset: A name for the dataset,
* classes: How many classes are to be differentiated with the model,
* neuralNetwork: Name of the neural network,
* learningRate: Float for the learning rate,
* threads: Int, how many CPU threads should be used for data provider.
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
* downsize: String, "True" when the images should be downsized, "False" when not,
* classes : How many classes are to be differentiated with the model,
* path: Relative path to the dataset foler,
* fileName: Filename of the serialized object,
* images: Subfolder of the train input images,
* labels: Subfolder of the label input images(Ground-truth),
* ClassToRGB: Int Array, where the index is the class and the value and RGB array that is associated with this class
 
