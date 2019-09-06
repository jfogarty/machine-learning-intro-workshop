# More ML Teaching Notes





# Data

- Can you get the data?
- Can you normalize the data?
- Are the solutions knowable?
- How much bias is there in your data?


# Models


- [supervised](https://en.wikipedia.org/wiki/Supervised_learning)
- [unsupervised](https://en.wikipedia.org/wiki/Unsupervised_learning)
- [semi-supervised](https://en.wikipedia.org/wiki/Semi-supervised_learning)
- [reinforcement learning](https://en.wikipedia.org/wiki/Reinforcement_learning).


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

# Problems Gradient Descent Cannot Solve
  

  - Convex problem spaces have a single solvable correct answer

  - No valid gradient, non-convex problem spaces

  - Non-convex optimization algorithms such as evolutionary algorithms

  - Most interesting problems have a non-convex loss function space


# Things we're not going to talk about much

  - Hard core MATH
    - Gradient: $ \operatorname {grad} f=\nabla f}{\displaystyle \operatorname {grad} f=\nabla f} $ and [del](https://en.wikipedia.org/wiki/Del)
  - [Probability Based Machine Learning]()
    - [Probabilistic Machine Learning](http://inverseprobability.com/talks/notes/probabilistic-machine-learning.html)
  
  - Robotics

  - General AI


  - How to deploy ML models
	- [Deploying a Machine Learning Model as a REST API](https://towardsdatascience.com/deploying-a-machine-learning-model-as-a-rest-api-4a03b865c166)



  - [Convex Functions](https://www.math24.net/convex-functions/)

  - Training with Massive Models and Datasets
    - Python GIL Issues (Global Interpreter Lock)
    - Segmentation of models into different GPUs and Servers


 -[Scaling a massive State-of-the-Art Deep Learning model in production](https://medium.com/huggingface/scaling-a-massive-state-of-the-art-deep-learning-model-in-production-8277c5652d5f)
- [Massive Datasets And Generalization In Ml](http://cachestocaches.com/2019/1/staggering-amounts-data/)

- Facebook AI Research (FAIR) have dwarfed [ImageNet 2012](http://image-net.org/challenges/LSVRC/2012/) with a 3-billion-image dataset comprised of hashtag-labeled images from Instagram.
- [YouTube-8M dataset](https://research.google.com/youtube8m/), geared towards large-scale video understanding : audio/visual features extracted from 350,000 hours of video. 
	- The amount of computation being used in the largest AI research experiments, like AlphaGo and Neural Architecture Search, is doubling every 3 to 5 months
    - OpenAI whose multiplayer-game-playing AI is trained using a massive cluster of computers so that it can play [180 years of games against itself every day](https://blog.openai.com/openai-five/).


 - Edge Computing : Running on the Pi, Arduino, NVidia Jetson Nano, etc.

---------------------------------

## Basics

- Linear regression; SSE; gradient descent; closed form; normal equations;
features78

- Overfitting and complexity

## Data

- training
- validation
- test data


## Classification

- Classification problems
- decision boundaries
- nearest neighbor methods

## Probability

- Probability and classification
- Bayes optimal decisions
- Naive Bayes and Gaussian class-conditional distribution
- Bayes' Rule and Naive Bayes Model

- Linear classifiers
- Logistic regression
- online gradient descent
- Neural Networks

- Decision tree


- Ensemble methods:
  - Bagging
  - Random forests
  - Boosting


- Unsupervised learning:
  - clustering
  - k-means
  - hierarchical agglomeration

## Unsupervised learning

  - K-means
  - Latent space methods
    - PCA.

- support vector machines (SVM)
- large-margin classifiers

- Time series
- Markov models
- autoregressive models


### End of note.
