# Facial-Expression-Classification

**PROBLEM STATEMENT**

Images are used as input and output signals in the field of image processing. One of the most important uses of image processing is the recognition of facial expressions. Our faces convey our emotions through non-verbal face gestures known as facial expressions. Interpersonal communication depends on facial expressions. They are comparable only to voice tone in relevance to daily emotional communication. They also function as a feeling indicator, enabling a person to communicate their emotional condition. A machine's capacity to perceive human emotion may lead to a variety of incredibly beneficial applications including healthcare. Consider the development of comforting and assisting therapeutic robots for the sick and disabled. We were also motivated by the advantages of those with physical disabilities. The ability for a human or automated system to interpret someone's wishes from their expression will make it much simpler for people to convey their wishes, nevertheless. Some other usages include examining how consumers feel when shopping with a focus on the products or how they are displayed in the store, identifying shoplifters, looking at footage from crime scenes to spot potential criminal intentions, determining how users feel about educational programs, changing the learning path, and developing efficient tutoring systems.
Modern applications for automated or real-time facial expression recognition include emotion analysis, cognitive science, virtual reality, and human-computer interaction. To build a trustworthy system, it is necessary to identify several expression types that try to be recognized. This project explores the idea of employing CNN, SVM, and DeepFace approach to construct a trustworthy and reliable classifier for human emotion as revealed in still photographs and the live or real-time feed from a webcam. Using information gathered from the dataset's face images that we utilized to train our models, we evaluated how well each of the three models performed in classifying human emotion. According to Ekman's list of common emotions, which includes happy, sad, angry, disgusted, fear, surprise, and neutrality are the seven emotions that we are taking into consideration for face recognition in our project. To test how our models could be affected by an image's sharpness, contrast, and brightness while predicting emotions, we also manipulated the images and checked their emotion prediction in CNN and DeepFace models.

**INTRODUCTION**

Facial expression recognition is a technique that may be carried out by humans or computers and here, our project entails:

1. Facial Feature Extraction: Face recognition in the surroundings and extraction of facial features from the recognized face region e.g., detecting the shape of face and its components such as eyes, using Haar-cascade in OpenCV.

2. Facial Expression Interpretation: Examining how the facial features move and/or change, and then categorizing this data into several interpretive categories for emotions like a grin or a frown, like joy or rage, or attitude categories like contempt or neutrality, etc. In order to do this, we employed and trained three different types of models, as follows:

a. CNN Model: Convolutional neural networks, often known as CNNs or ConvNets, specialize in processing input with a grid-like design, such as images. The three layers of a CNN are generally convolutional, pooling, and fully connected [7].

b. SVM Model: SVMs, a type of supervised learning method that performs well in high-dimensional domains.

c. DeepFace: DeepFace is the Python face recognition and facial attribute (age, gender etc.) analysis module. This free and open-source DeepFace library wraps all the most recent AI facial recognition models [9].

3. Image Manipulation: Manipulating pictures' brightness, contrast, and sharpness using Python PIL (pillow) library’s ImageEnhance module, to see how it would impact the CNN model's and DeepFace model's ability to anticipate emotions [10].

4. Real-time/Live Emotion Prediction: Prediction of emotions in real-time (webcam) using CNN model and DeepFace with face annotation.

**DATASET**

The Kaggle Facial Expression Recognition Challenge (FER2013) provided the data set that we utilized to train our models. The columns in the dataset are:

• Emotion: The emotion variable represents emotions.

• Pixels: Pixel’s variable expresses the value per pixel in the photos.

• Usage: Usage shows which set the row it belongs to such as training and testing.

Anger, disgust, fear, happiness, sadness, surprise, and neutral are the seven emotions represented in the dataset's 48 x 48-pixel grayscale pictures. The dataset includes 3589 instances in the public testing set, 28709 examples in the training set, and 3,589 examples in the private testing set, so in total 35,887 photos are the result of the dataset. Each face is assigned to one of seven categories, with 0 denoting anger, 1 denoting disgust, 2 denoting fear, 3 denoting happiness, 4 denoting sadness, 5 denoting surprise, and 6 denoting neutralities.

**METHODOLOGY**

This section describes the techniques used with comparison, discussion of the approaches, inputs, outputs and transformations:

A.) Facial Feature Extraction using Haar-cascade with annotations is performed using OpenCV. In this ML based method, a cascade function is trained from negative and positive images to detect the objects. This however requires tons of images for training and therefore we have used a predefined Haar-cascade in OpenCV for the eyes and face annotation in our images. This is available in an XML file which we have used for our image and real-time/live webcam annotation.

B.) CNN is a neural network-based approach where our input data shape is (48,48,1). The activation function used throughout is Relu except for the final layer where the function is a softmax function. The CNN has a 6
convolutional layer with 1 max pooling layer after every 2 convolutional layers and two dense layers. Neural network is trained over varied epochs (10,15,20,50,100) to obtain the best performing neural network configuration. We export our trained model in .h5 form to further usage of predication on unseen data and real time emotion recognition.

C.) SVM is a machine learning approach that has been used for the recognition and classification of facial expressions. The pixel row in the dataset is converted to list of floating integers. Followed by reshaping 1D array to 2D array of 48*48 pixels. Then, after aligning the face and eyes through CV2, we are again reshaping the 2D array to 1D array of 2304 pixels. Principal Component Analysis is then applied on the dataset to reduce the data dimensionality. SVM is then trained with Radial basis function kernel and decision function as OVR (One vs Rest) over 10000 epochs. Correct predictions can be identified by green colored images and incorrect ones are red in color.

D.) DeepFace is an open-source python framework for facial detection, recognition, and verification. The library can be installed using a pip command (pip install deepface). DeepFace is a collection of multiple face recognition packages. Many cutting-edge face recognition models are presently supported by it, including VGG-Face, Google FaceNet, OpenFace, Facebook DeepFace, DeepID, ArcFace, Dlib, and SFace. VGG-Face is the model used by default. The library besides emotion also provides facial attribute analysis such as age, gender, emotion, and race.

E.) Real-time emotion recognition was performed using in-built stream function for the DeepFace model while the same was performed using Python’s OpenCV library for the CNN Model. A custom function was created which accessed the system’s camera. The CNN model is then called, for predicting the emotion in real-time with face and eye annotation.
