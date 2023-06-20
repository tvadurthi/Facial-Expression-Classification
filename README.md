# Facial-Expression-Classification

**Problem Statement:**

Investigated the effectiveness of a deep convolutional neural network in detecting handwritten English characters in this study.

**I. Introduction:**

In recent years, one of the most exciting and hard research areas in the realm of image processing and pattern recognition has been handwriting recognition. It makes a significant contribution to the improvement of automated processes and improves the human-machine interface in a variety of applications. Several studies have focused on developing new strategies and methodologies that would cut processing time while increasing recognition accuracy.

**A. Deep Learning**

Deep learning is a subset of machine learning that is essentially a three-layer neural network. These neural networks aim to imitate the activity of the human brain by allowing it to "learn" from enormous amounts of data, albeit they fall far short of its capabilities. While a single-layer neural network may produce approximate predictions, additional hidden layers can help to optimize and improve for accuracy.

**B. Neural Networks**

A neural network is a system that is inspired by human brain activity and is made up of neurons that are connected in parallel and can learn. The input layer, hidden layer, and output layer are the three main layers of a neural network. With their extraordinary ability to infer meaning from complex or imprecise data, neural networks may be used to identify patterns and discover trends that are too complex for people or other computer systems to notice. A trained neural network can be regarded as an "expert" in the category of data it has been trained to assess. Pattern recognition, group
categorization, series prediction, and data mining are all activities that benefit from neural networks.

**C. Convolutional Neural Networks**

The deep convolutional neural network (CNN) has been the architecture of choice for difficult vision recognition problems for several years. Deep CNN's capacity to recognize handwritten digits, English alphabets, and more broad Latin alphabets has been extensively investigated. Experiments have shown that well-constructed deep CNNs are effective instruments for dealing with these problems. The most prevalent application of neural networks is pattern recognition. The neural network is given an input that contains pattern information, which could be in the form of an image or handwritten data. The neural network then tries to figure out if the input data follows a pattern that has been indicated by the neural network. A classification neural network is meant to accept input samples and categorize them into categories. These input groupings could be hazy, with no apparent boundaries. The goal of this research is to detect free handwritten characters.
A sort of Neural Network that works similarly to a standard neural network except for the addition of a convolution layer at the start. An input and output layer, as well as numerous hidden layers, make up a convolutional neural network. Convolutional layers, RELU layers (activation function), pooling layers, fully connected layers, and softmax layers are often found in CNN's hidden layers.

**D. Keras API**

Keras is a deep learning API written in Python, running on top of the machine learning platform TensorFlow. It was developed with a focus on enabling fast experimentation.
Keras is:

Simple: but not simplistic. Keras reduces developer cognitive load to free you to focus on the parts of the problem that really matter.

Flexible: Keras adopts the principle of progressive disclosure of complexity: simple workflows should be quick and easy, while arbitrarily advanced workflows should be possible via a clear path that builds upon what you've already learned.

Powerful: Keras provides industry-strength performance and scalability: it is used by organizations and companies including NASA, YouTube, or Waymo.

**E. Keras & TensorFlow 2:**

TensorFlow 2 is an open-source machine learning platform that runs from start to finish. It can be thought of as a foundation layer for differentiable programming. It brings together four crucial abilities:
Executing low-level tensor operations efficiently on the CPU, GPU, or TPU. The gradient of arbitrary differentiable expressions is computed. Scaling computation to a large number of devices, such as hundreds of GPUs in a cluster. Exporting programs (graphs) to external runtimes like servers, browsers, mobile devices, and embedded devices. Keras is TensorFlow 2's high-level API: a user-friendly, high-productivity interface for addressing machine learning issues, with a focus on current deep learning. It provides fundamental abstractions and building elements for designing and releasing high iteration rate machine learning systems. Engineers and researchers may use Keras to fully use TensorFlow 2's scalability and cross-platform capabilities: you can run Keras on TPU or massive clusters of GPUs, and you can export Keras models to run in the browser or on a mobile device.

**II. Dataset**

Handwritten A-Z Alphabet Dataset from Kaggle was used to train the model. It was further used to develop a program to recognize Handwritten Text on paper and is totally suitable for real-life examples. There are 3,72,450 photos in the collection, with 26 labels ranging from 0 to 25 for all the English alphabets (A-Z) containing handwritten graphics. The label for every image is specified in the first column of the dataset. The size of the images are 28 X 28 pixels, each pixel is considered as a feature. Therefore, the dataset contains 784 columns and is 699 MB. Each letter in the image is center fitted to a 20 X 20-pixel box. Each image is stored as gray level. Hence the value for each pixel ranges from 0 to 255. The images are taken from NIST (https://www.nist.gov/srd/nist-special-database-19) and NMIST large dataset and few other sources which were then formatted as mentioned above.

**III. Methodology**

Initially we load the dataset into a NumPy array.
As reported earlier the dimensions of the dataset are (372450). A list of the English alphabets is created.
We extract the numeric labels of the images through the first column and map them to their corresponding alphabets using the alphabet list created above.
The data of each image is represented in a single row in the dataset where each feature represents an individual pixel intensity of the 784 image pixels (28 X 28).

**A. Pre – processing**

We reshape the dataset to a 3D format. We arrange the pixels in 28 X 28 format and label each image according to the label class. Basically, we are constructing 372,450,785 2D matrices, each representing an image from the dataset.
We count the occurrence of each alphabet image in the dataset and plot the count against the alphabet. It is observed that the most occurring alphabet in the dataset is ‘o’ followed by ‘s’. The least count is observed in letters ‘f’ and ‘I’.
To get a better understanding of the handwritten images with which we will train our model, we plot them. Having reshaped the original dataset from 372450 X 784 to 372450 X 28 X28, we plot 400 out of the 372450 to get a visual intuition of the data.

**B. Splitting the Dataset**

Next, we split the data into training and testing data. As we will be dealing with convoluted neural networks, we need a lot of training data for better model performance. For this reason, we have chosen a 90% training and 10% testing split. For the ease of convenience, we add another dimension to the data to make it linear and easier to split.

**C. Building the Convolutional Neural Network Model**

After pre-processing and splitting our dataset, we begin to implement the neural network. We are building out CNN models using Sequential API. Sequential API allows us to create a model’s layer by layer in a step-by-step fashion. A 4 convolutional layer with 1 max pooling layer after every 2 convolutional layers. As our input layer should be two dimensional, a flattening layer needs to be added between them. At the end of the fully connected layers is a softmax layer.

**D. Training The Network**

A low probability results in minimal effect and a high probability results in under learning by the network. When dropout is used on a larger network, we are likely to get better performance as the model learns more independent representations. Using dropout on visible as well as hidden units that shows good results.

**E. Dense Layer and Flatten Layer**

In a neural network, a dense layer is simply a regular layer of neurons. Each neuron in the previous layer receives information from all the neurons in the layer
above it, making it tightly linked. The reason we do this is that we're going to need to insert this data into an artificial neural network later on.

**V. Results:**

Training on the full dataset (over 3 million examples) can take many hours or days even with the fastest GPU.
Constrained by the computation resources we have access to, we ran our major experiments (for model comparison and visualization) only with one epoch.
We can increase the number of epochs to train our model more efficiently which perhaps will increase our accuracy even further.
With one epoch, the accuracy we get is, around 95% and the validation accuracy we get is, around 98%.

**VI. Future Scope:**

Considering the limitations of the project, we define the future scope to overcome them. The algorithm can be improved by making it capable of handling lowercase, uppercase and cursive RGB images without any specific orientation. Apart from this, the algorithm can be made more versatile by increasing the number of languages it can decipher.
