"""
Author: Md Mostafizur Rahman
File: Testing and submission Kaggle MNIST
"""

import os, keras
import pandas as pd
import numpy as np

# project modules
from .. import config
from . import preprocess, my_model

#Loading Model
model = my_model.read_model()


#loading test data
result = []
test_data = preprocess.load_test_data()
print("From Test file", test_data.shape)

#Predicting results
print("Model Predictions...")
predictions = model.predict(test_data,
                            batch_size = config.batch_size,
                            verbose = 2)
label_preditions = np.argmax(predictions, axis= 1)

print(predictions.shape)