# Intro to ML General Flow


# ML Data

- How do you get the data?
- What should it look like?
- How do you clean up the data?
- How do you normalize the data?
- Are the solutions knowable?
- How much bias is there in your data?
- How do you partition the data for training?

# Models

- Model Types
	- [supervised](https://en.wikipedia.org/wiki/Supervised_learning)
	- [unsupervised](https://en.wikipedia.org/wiki/Unsupervised_learning)
	- [semi-supervised](https://en.wikipedia.org/wiki/Semi-supervised_learning)
	- [reinforcement learning](https://en.wikipedia.org/wiki/Reinforcement_learning).

- Model Implementations
  - Traditional Forms
    - Decision Trees
    - SVMs
    - Ensemble methods
  - Neural Network Forms
    - MLPs
    - CNNs
    - RNNs/Transformer/Attention

- [Model Zoo - Framworks](https://modelzoo.co/frameworks)
- [Model Zoo - Categories](https://modelzoo.co/categories)


# Python Frameworks

- [Numpy](https://numpy.org/) : the fundamental package for scientific computing with Python [Github](https://github.com/numpy/numpy)
- [Pandas](https://pandas.pydata.org/) : data structures and data analysis tools [Intro](https://pythonprogramming.net/introduction-python3-pandas-data-analysis/) [Github](https://github.com/pandas-dev/pandas)
- [Scikit-Learn](https://scikit-learn.org/stable/) machine learning built on top of [SciPy](https://www.scipy.org/) [Examples](https://scikit-learn.org/stable/auto_examples/index.html#clustering) [Github](https://github.com/scikit-learn/scikit-learn)
- [OpenCV](https://opencv.org/) [OpenCV-Python](https://opencv-python-tutroals.readthedocs.io/en/latest/) [Intro](https://pythonprogramming.net/loading-images-python-opencv-tutorial/) [Github](https://github.com/opencv/opencv)

# Deep Learning Frameworks

- [Keras](https://keras.io/) [Zoo](https://modelzoo.co/framework/keras) [github](https://github.com/keras-team/keras)
- [TensorFlow](https://www.tensorflow.org/) [Zoo](https://modelzoo.co/framework/tensorflow) [github](https://github.com/tensorflow)
- [Caffe](http://caffe.berkeleyvision.org/) [Zoo](https://modelzoo.co/framework/caffe) [github](https://github.com/BVLC/caffe/)
- [Chainer](https://chainer.org) : [Zoo](https://modelzoo.co/framework/chainer) [github](https://github.com/chainer/chainer)
- [MXNet](https://mxnet.incubator.apache.org/)/[Gluon](https://mxnet.incubator.apache.org/versions/master/gluon/index.html) [Zoo](https://modelzoo.co/framework/mxnet) [github](https://github.com/apache/)
- [PyTorch](https://pytorch.org/) [Zoo](https://modelzoo.co/framework/pytorch) [github](https://github.com/pytorch/pytorch)
- [Caffe2 - NOW PyTorch](http://caffe2.ai/) [Zoo](https://modelzoo.co/framework/caffe2) [github](https://github.com/caffe2)


## Jupyter, Numpy, Pandas

- 1.1 Python Numpy Tutorial: this is the bare minimum to get you started with Python and the Numpy arrays.
- 1.2 Numpy Vectors, Arrays, and Matrices: More specific information on the data structures used in ML

- Chris Albon's Notes : The unreasonably useful notes of Chris Albon (host of the Partially Derivative podcast)

- 1.3 Intro to Pandas: the required library for data analysis and modeling : Pandas
- 1.4 More Pandas: more useful pandas snippets applied to the Titanic dataset

## Look at some ML Data

- 2.1 Exploring/visualizing Data - 1: The classic Iris dataset visualized with seaborn and pandas

- 2.3 Exploring/visualizing Data - 3: The Darwin's Finches dataset with some nice EDA - Exploratory Data Analysis
- Python Machine Learning - 2nd Edition : Another good ML book created with Github notebooks

## Look at some ML Models

- 3.1 Gradient Descent: The basics of gradient descent
- 3.2 Least squares : The simplest fitting method : Linear regression
- 3.4 ML Workflow : A full end to end project on the diabetes dataset; workflows: Google Amazon Azure Watson

## Beyond basic 'accuracy'

- 6.1 Confusion Matrix : Using confusion matrices to evaluate classifiers : Confusion_Matrix

## Intro to NN models

- 5.1 Bias and Weights: Fundamental concepts: bias, and weights
- 4.1 Training a Model 1: Exploring MNIST

## Detail on a Conventional Model

- Scikit-learn : Excellent documentation and notebooks
- 3.5 Decision Trees : An introduction to decision trees : Decision Tree Learning

## Back to NN models

- 5.2 Activation Functions: Fundamental concepts: activation functions
- 4.3 Binary functions: computing the simplest binary functions (not, xor, or, and) with neural networks.

- 5.4 Math of Neural Networks: general NN designed using just Python and Numpy.
- 4.2 Training a Model 2: Exploring FashionMNIST

## More Metrics

- 6.2 Precision Recall Curves : Precision-Recall to evaluate classifiers : Precision-Recall
- 6.3 Receiver Operating Characteristics : ROC and AUC (Area Under the Curve) to evaluate classifiers : AUC_ROC

- 6.4 Minimizing Loss and Validating Models: Making better models through hyperparameter tuning

- 3.3 Beyond Gradient Descent: Adaptive techniques that improve on ordinary gradient descent

### Model Evalution

- Confusion Matrix : correct / incorrect classed by result type
- Precision : how many of the positive results were good?
- Recall : did most of the valid results get found?
- ROC and AuROC : True positives as a function of False positives

## Advanced Visualization

- 2.4 Visualization of MLP weights on MNIST : Looking at the weight matrices; reading the tea leaves
- 2.5 Visualizing MNIST: An Exploration of Dimensionality Reduction : Another classic Colahl)) more analysis with t-SNE visualization

- 5.3 Exploding and Vanishing Gradients: Gradient problems; Fundamental concepts: gradients

## A bit of Advanced Unsupervised Learning

- 5.5 Training an AutoEncoder : Using a NN for Credit Fraud detection based on an AutoEncoder

# End.