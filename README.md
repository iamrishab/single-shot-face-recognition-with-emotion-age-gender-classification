## Overview

This is a TensorFlow implementation of the face recognizer described in the paper
["FaceNet: A Unified Embedding for Face Recognition and Clustering"](http://arxiv.org/abs/1503.03832). The project also uses ideas from the paper ["Deep Face Recognition"](http://www.robots.ox.ac.uk/~vgg/publications/2015/Parkhi15/parkhi15.pdf) from the [Visual Geometry Group](http://www.robots.ox.ac.uk/~vgg/) at Oxford.

A mobilenet SSD(single shot multibox detector) based model is used for face detection, powered by tensorflow [object detection api](https://github.com/tensorflow/models/tree/master/research/object_detection), trained on [WIDERFACE dataset](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/).

Real-time face detection and emotion/gender classification using fer2013/IMDB datasets with a keras CNN model and openCV.
* IMDB gender classification test accuracy: 96%.
* fer2013 emotion classification test accuracy: 66%.

For more information please consult the [publication](https://github.com/oarriaga/face_classification/blob/master/report.pdf)

## Compatibility

The code is tested using Tensorflow r1.10 under Ubuntu 16.04 with Python 3.5. But it may work perfectly with other versions also.

## Pre-trained models
| Model name       | Training dataset | Architecture | Usage
|-----------------|------------------|-------------|---------
| [20170512-110547](https://drive.google.com/open?id=18RxJk9Pk0Mic-FpM5tDlZbdt8w6C0GzW) | [CASIA-WebFace](http://www.cbsr.ia.ac.cn/english/CASIA-WebFace-Database.html)    | [Inception ResNet v1](https://github.com/davidsandberg/facenet/blob/master/src/models/inception_resnet_v1.py) | Face Embeddings
| [frozen_inference_graph_face](https://drive.google.com/open?id=1WAP6_EiQyHdKaGtuql6PHSFKgYf3cRBw) | [WIDERFACE dataset](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/)    | SSD Mobilenet | Face Detection
|[fer2013_mini_XCEPTION.119-0.65](https://github.com/Tony607/Keras_age_gender/releases/download/V1.0/weights.18-4.06.hdf5) | [Fer2013](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data) | Inception ResNet | Emotion Classification

The prototype is designed to be implemented for different applications:
- Personalized Customer Experience
- Collect Customer Demographics
- Enhanced Store Traffic Analytics
- Tracking Customer Emotions
- Improve Store Security

NOTE: If you use any of the models, please do not forget to give proper credit to those providing the training dataset as well.

