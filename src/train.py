"""
Author: Md Mostafziur Rahman
File: Traning a CNN architecture using the MNIST dataset
"""

import os, keras
import numpy as np


# module packages
from .. import config
from . import preprocess, my_model

# Loding train Data
train_data, train_labels = preprocess.load_train_data()
print("Train Data shape: ", train_data.shape)
print("Train Data Labels: ", train_labels.shape)

# Loding Model
model = my_model.get_model()

#Compile
model.compile(keras.optimizers.Adam(config.lr),
              keras.losses.categorical_crossentropy,
              metrics = ['accuracy'])

# Check point
model_cp = my_model.save_model_checkpoints()
early_stopping = my_model.set_early_stopping()

#model training
model.fit(train_data, train_labels,
          batch_size = config.batch_size,
          epochs = config.nb_epocs,
          verbose=2,
          shuffle = True,
          callbacks = [early_stopping, model_cp],
          validation_split = 0.2)
