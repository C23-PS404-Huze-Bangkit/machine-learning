# [C23-PS404]-Huze
## Machine Learning Documentation

We use a dataset with 10 most popular dog breeds and 10 most popular cat breeds based on research in Indonesia, Japan, Malaysia, and the world. The image data resource is taken from :
* https://www.kaggle.com/zippyz/cats-and-dogs-breeds-classification-oxford-dataset
* https://www.kaggle.com/jessicali9530/stanford-dogs-dataset
* https://www.kaggle.com/datasets/ma7555/cat-breeds-dataset

Here is the Dataset and Converted Model File Link that we have processed : 
* https://drive.google.com/drive/folders/1y_LicM_emds9wmR5h_BSPZfDj5BgzevW?usp=share_link 

# Table of contents
* [Roadmap](#roadmap)
* [Prerequisites](#prerequisites)
* [Models](#models)


# Roadmap
1. We manually check each and every existing race data because the dataset obtained from Kaggle is still messy. For instance, it includes images of humans, newspapers, or logos that were not properly labeled by the dataset creator. Here is the link to the cleaned dataset: Data Cleaning and Preprocessing. The dataset consists of 10 dog classes and 10 cat classes, or alternatively, 20 dog classes and 20 cat classes.
2. After that we try to make a model and we are looking for models with which architecture has the highest level of accuracy both on the training, validation and testing sides. 
3. The model that has been trained and evaluated the results will be stored in HDF5 or H5 format and the format will be converted into tflite format. 
4. We chose Xception as the base model and added a CNN layer to the last layer of its architecture. This decision was based on its high accuracy on the testing data, with the training model achieving an accuracy of 0.8510 and the validation accuracy reaching 0.8400. Compared to models such as just CNN, MobileNet_v2, InceptionV3, and ResNet, this model demonstrated superior performance. 
  * [Notebook Xception](https://github.com/C23-PS404-Huze-Bangkit/machine-learning/blob/main/Xception-datasetfix.ipynb) 
  * [Model Results in H5 and tflite formats](https://drive.google.com/drive/folders/1vQEcVZ4Yh1R-K-twZ9bH-J_mYdZwHGF9?usp=sharing) 

# Prerequisites
Here are the technologies you should install if you are using Jupyter-notebook. If you're using Google Colab you don't need to install it just import the libraries. Because We have a capable GPU so we use jupyter-notebook and have to install this library.

* Python : You can access this link to install [python](https://www.python.org/downloads/) and using [pip](https://pypi.org/project/pip/) for installing  packages/libraries
* Tensorflow : 
  ```bash
  pip install tensorflow
* tensorflow_hub :
  ```bash
  pip install tensorflow_hub
* Keras : 
  ```bash
  pip install keras
* numPy : 
  ```bash
  pip install numpy
* Pandas : 
  ```bash
  pip install pandas
* Matplotlib :
  ```bash
  pip install matplotlib
* Scikit - learn : 
  ```bash 
  pip install scikit-learn 

# Models
* CNN
* mobilenet_v2
* inceptionv3
* RestNet
* Xception



# Notes 
We uploaded the dataset and Converted Model File via gdrive due to limited space on github only maximizing 100MB or efficiency of only 50MB per file.