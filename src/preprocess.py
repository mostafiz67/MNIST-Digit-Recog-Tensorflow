"""
Author: Md Mostafizur Rahman
File: Preprocessing MNIST train and test datasets

"""

import os, cv2
import numpy as np
import pandas as pd
#import tensorflow.compact.v1 as tf 

from sklearn.preprocessing import LabelBinarizer

from .. import config

#How can I do this for CSV???
train_img = np.ndarray((config.nb_train_samples, config.img_size, config.img_size, config.img_channel),
                       dtype=np.float32)

def normalization(x):
    x = np.divide(x, 255)
    x = np.subtract(x, 0.5)
    x = np.multiply(x, 2)

    return x

def load_train_data():
    
    # Loding CSV file in a dataframe with values
    train_img_df = pd.read_csv(os.path.join(config.dataset_path(), "mnist_train.csv")).values
    
    #Spliting image pixels from labels
    print("Loding Images including changing its shape! ...")
    train_images = train_img_df[:, 1:].reshape(train_img_df.shape[0],28, 28,1).astype( 'float32' )
    
    print("Doing Normalization Train Data")
    train_images = np.multiply(train_images, 1.0 / 255.0)
    print(train_images.shape)
    
    #Spliting labels from image pixels
    print("Loding Labels")
    train_data_lables = train_img_df[:,0]
    train_data_lables_count = np.unique(train_data_lables).shape[0]
    print(train_data_lables_count)
    
    #One hot encoding image labels 
    print("One hot encoding")
    encoder = LabelBinarizer()
    train_labels = encoder.fit_transform(train_data_lables)
    
    print(train_images.shape)
    print(train_labels.shape)
    
    return train_images, train_labels

#Loadnig Test Data
def load_test_data():
    
    # Loding CSV file in a dataframe with values
    test_img_df = pd.read_csv(os.path.join(config.dataset_path(), "mnist_test.csv")).values
    test_img_df = test_img_df[: ,1:]
    print(test_img_df.shape)
    
    #Spliting image pixels from labels
    print("Loding Test Images including changing its shape! ...")
    test_images = np.multiply(test_img_df,1.0/255.0)
    test_images = test_images.reshape(test_images.shape[0],28, 28, 1).astype( 'float32' )
    
    print("Doing Normalization Test Data")
    test_images = np.multiply(test_images, 1.0 / 255.0)
    print(test_images.shape)
    
    #return test_images, test_data_lables
    return test_images



    
if __name__ == "__main__":
    #x, y = load_test_data()
    #print(x.shape)
    #print(y.shape)
    x = load_test_data()
