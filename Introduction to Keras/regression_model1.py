import tensorflow as tf
import os
import pandas as pd
import numpy as np
import keras
import warnings
warnings.simplefilter('ignore', FutureWarning)

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Input

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

filepath='https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0101EN/labs/data/concrete_data.csv'
concrete_data = pd.read_csv(filepath)

#Check data structure
'print(concrete_data.head())'
# Check numbber of datasets
'print(concrete_data.shape)'

#Check for missing values
'print(concrete_data.isnull().sum())'

concrete_data_columns = concrete_data.columns
predictors = concrete_data[concrete_data_columns[concrete_data_columns != 'Strength']] # all columns except Strength
target = concrete_data['Strength'] # Strength column

#Normalize the data
predictors_norm = (predictors - predictors.mean()) / predictors.std()
predictors_norm.head()

n_cols = predictors_norm.shape[1] # number of predictors

# define regression model
def regression_model():
    input_colm = predictors_norm.shape[1] # Number of input features
    # create model
    model = Sequential()
    model.add(Input(shape=(input_colm,)))  # Set the number of input features 
    model.add(Dense(50, activation='relu'))  
    model.add(Dense(50, activation='relu'))
    model.add(Dense(50, activation='relu')) 
    model.add(Dense(50, activation='relu'))
    model.add(Dense(50, activation='relu'))  
    model.add(Dense(1))  # Output layer
    
    # compile model
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# build the model
model = regression_model()

# fit the model
model.fit(predictors_norm, target, validation_split=0.3, epochs=100, verbose=2)

loss = model.evaluate(predictors_norm, target, verbose=2)
print("Mean Squared Error on full dataset:", loss)
print(np.sqrt(loss))