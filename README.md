# mnist-keras

This repo holds the code for a simple handwritten digit classifier trained on MNIST database

## Contents
 1. [Dependencies](#dependencies)
 2. [Usage](#usage)
 3. [References](#references)

### Dependencies

The classifier requires the installation of the following dependencies:

1. [Tensorflow](https://www.tensorflow.org/install/)
2. [Keras](https://keras.io/#installation)
3. [Matplotlib](https://matplotlib.org/users/installing.html)
4. [h5py](http://docs.h5py.org/en/latest/build.html)

### Usage

1. To train the model run mnistTrain.py (if you want to use the pretrained model, skip this step). Note that it may take a while if you're not using a GPU:
```bash
   python mnistTrain.py
``` 

2. To test the model on ten random images from the dataset, run mnistClassify.py:
```bash
   python mnistClassify.py
``` 

### References
1. LeCun, Yann; Corinna Cortes; Christopher J.C. Burges. ["MNIST handwritten digit database, Yann LeCun, Corinna Cortes and Chris Burges"](http://yann.lecun.com/exdb/mnist/).
2. Martín Abadi, Ashish Agarwal, Paul Barham, Eugene Brevdo,
Zhifeng Chen, Craig Citro, Greg S. Corrado, Andy Davis,
Jeffrey Dean, Matthieu Devin, Sanjay Ghemawat, Ian Goodfellow,
Andrew Harp, Geoffrey Irving, Michael Isard, Rafal Jozefowicz, Yangqing Jia,
Lukasz Kaiser, Manjunath Kudlur, Josh Levenberg, Dan Mané, Mike Schuster,
Rajat Monga, Sherry Moore, Derek Murray, Chris Olah, Jonathon Shlens,
Benoit Steiner, Ilya Sutskever, Kunal Talwar, Paul Tucker,
Vincent Vanhoucke, Vijay Vasudevan, Fernanda Viégas,
Oriol Vinyals, Pete Warden, Martin Wattenberg, Martin Wicke,
Yuan Yu, and Xiaoqiang Zheng.
TensorFlow: Large-scale machine learning on heterogeneous systems, 2015. Software available from [tensorflow.org](www.tensorflow.org).
3. Chollet, François and others. Keras, 2015. Software available from [keras.io](keras.io).