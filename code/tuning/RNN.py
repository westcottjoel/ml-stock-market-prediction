import os
os.environ['PYTHONWARNINGS'] = 'ignore'
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error,r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow.keras.layers as tfkl
import keras_tuner as kt
import tensorflow as tf
from alibi.explainers import IntegratedGradients
import tensorflow.keras.backend as KB


#MODEL TAKES IN 4 CMDLINE ARGUMENTS -
#1) FILE PATH OF THE DATA
#2) DATA FORMAT - "Date" or "Datetime" (Default = "Datetime")
#3) BOOLEAN VALUE FOR WHETHER XAI PLOTS ARE TO BE PRINTED - "True" or "False" (Default = "False")

#READ IN CMDLINE ARGUMENTS

file_path = sys.argv[1]
company_name = Path(file_path).name
data_format = sys.argv[2]

#READ IN DATA

data = pd.read_csv(file_path,delimiter=',',header=0,parse_dates=True)

#FILTER OUT UNNEEDED COLUMNS

data = data.filter([data_format,'Open','Close','High','Low','Volume'])

#CLEAN UP ANY MISSING VALUES

data = data.bfill()

#EXTRACT FEATURES

#Smoothed Volume


data['Volume-Smoothed'] = data['Volume'] / data['Volume'].rolling(window=20).mean()

# Relative Strength Index (RSI)

def RSI(window):
    gains=[]
    losses=[]
    for diff in window.diff():
        if diff >= 0:
            gains.append(diff)
        if diff < 0:
            losses.append(-diff)
    mean_gains = np.mean(gains) if gains else 0
    mean_losses = np.mean(losses) if losses else 0
    if mean_losses == 0:
        return(100)
    RS = mean_gains / mean_losses
    return(100 - (100 / (1+RS)))

data['RSI'] = data['Close'].rolling(window=14).apply(RSI,raw=False)

# Simple Moving Averages (SMA)
data['SMA_10'] = data['Close'].rolling(window=10).mean()
data['SMA_50'] = data['Close'].rolling(window=50).mean()

# Exponential Moving Averages (EMA)
data['EMA_10'] = data['Close'].ewm(span=10, adjust=False).mean()
data['EMA_50'] = data['Close'].ewm(span=50, adjust=False).mean()

# Moving Average Convergence Divergence (MACD)
data['MACD'] = data['Close'].ewm(span=12, adjust=False).mean() - data['Close'].ewm(span=26, adjust=False).mean()

# Lag Features
data['LAG_3'] = data['Close'].shift(3)
data['LAG_4'] = data['Close'].shift(4)
data['LAG_5'] = data['Close'].shift(5)
data['LAG_6'] = data['Close'].shift(6)
data['LAG_7'] = data['Close'].shift(7)
data['LAG_8'] = data['Close'].shift(8)
data['LAG_9'] = data['Close'].shift(9)
data['LAG_10'] = data['Close'].shift(10)

#DROP NAN VALUES

data = data.dropna().reset_index(drop=True)

#SEPARATE DATE FOR FUTURE PLOTTING

dates = np.array(data[data_format])
dates = pd.to_datetime(dates)
data=data.drop(columns=[data_format])

window_size = 60
dates=dates[window_size:]
x_values = []
y_values = []

for i in range(window_size, len(data)):
    x_values.append(data.values[i - window_size:i, :])
    y_values.append(data.values[i, :])

x_values = np.array(x_values)
y_values = np.array(y_values)

#SPLIT INTO TRAIN/TEST/VALIDATION

train_val_set_x, test_set_x, train_val_set_y, test_set_y, dates_train_val, dates_test = train_test_split(
    x_values, y_values, dates, test_size=0.2, shuffle=False
)

train_set_x, val_set_x, train_set_y, val_set_y, dates_train, dates_val = train_test_split(
    train_val_set_x, train_val_set_y, dates_train_val, test_size=0.25,shuffle=False
)

#SCALE VALUES

train_set_x_shape = train_set_x.shape
train_set_x =  np.reshape(train_set_x,(-1,train_set_x_shape[2]))#reshape to 2d to fit scaler

test_set_x_shape = test_set_x.shape
test_set_x =  np.reshape(test_set_x,(-1,test_set_x_shape[2]))

val_set_x_shape = val_set_x.shape
val_set_x =  np.reshape(val_set_x,(-1,val_set_x_shape[2]))

scaler=MinMaxScaler()
scaler.fit(train_set_x)

train_set_x = scaler.transform(train_set_x)
test_set_x = scaler.transform(test_set_x)
val_set_x = scaler.transform(val_set_x)

train_set_x = np.reshape(train_set_x,train_set_x_shape)
test_set_x = np.reshape(test_set_x,test_set_x_shape)
val_set_x = np.reshape(val_set_x,val_set_x_shape)

train_set_y = scaler.transform(train_set_y)
test_set_y = scaler.transform(test_set_y)
val_set_y = scaler.transform(val_set_y)

#manually design RMSE function as it is not built into keras

def root_mean_squared_error(actual, predicted):
    return KB.sqrt(KB.mean(KB.square(predicted - actual)))
#FUNCTION and TUNER INSTANSIATION TAKEN FROM https://www.tensorflow.org/tutorials/keras/keras_tuner
def model_builder(hp):
    model = Sequential()
    hp_units = hp.Int('rnn_units', min_value=32, max_value=128, step=32)
    hp_regularizer = hp.Choice('regularizer_val',values=[1e-1,1e-2,1e-3,1e-4,1e-5])
    reg = regularizers.l2(hp_regularizer)
    model.add(tfkl.SimpleRNN(
        units=hp_units,return_sequences=True, 
        input_shape=(train_set_x.shape[1], train_set_x.shape[2]),
        kernel_regularizer=reg,
        recurrent_regularizer=reg,
        bias_regularizer=reg)
        )
    model.add(tfkl.Dropout(hp.Float('dropout_rate1', min_value=0.1, max_value=0.4, step=0.1)))
    model.add(tfkl.SimpleRNN(
        units=hp_units,
        return_sequences=True,
        kernel_regularizer=reg,
        recurrent_regularizer=reg,
        bias_regularizer=reg)
        )
    model.add(tfkl.Dropout(hp.Float('dropout_rate2', min_value=0.1, max_value=0.4, step=0.1)))
    model.add(tfkl.SimpleRNN(
        units=hp_units,
        kernel_regularizer=reg,
        recurrent_regularizer=reg,
        bias_regularizer=reg)
        )
    model.add(tfkl.Dropout(hp.Float('dropout_rate3', min_value=0.1, max_value=0.4, step=0.1)))
    model.add(tfkl.Dense(units=1))

    hp_learning_rate = hp.Choice('learning_rate', values=[1e-1,1e-2, 1e-3, 1e-4])
    model.compile(optimizer=Adam(learning_rate=hp_learning_rate),
                    loss='mean_squared_error',
                    metrics=[root_mean_squared_error])
    
    return(model)

def model_builder(hp):
    model = Sequential()
    hp_units = hp.Int('rnn_units', min_value=32, max_value=128, step=32)
    hp_dropout = hp.Float('dropout_rate', min_value=0.1, max_value=0.5, step=0.1)
    hp_regularizer = hp.Choice('regularizer_val',values=[1e-1,1e-2,1e-3,1e-4,1e-5])
    reg = regularizers.l2(hp_regularizer)
    model.add(tfkl.SimpleRNN(
        units=hp_units,return_sequences=True, 
        input_shape=(train_set_x.shape[1], train_set_x.shape[2]),
        kernel_regularizer=reg,
        recurrent_regularizer=reg,
        bias_regularizer=reg)
        )
    model.add(tfkl.Dropout(hp.Float('dropout_rate', min_value=0.1, max_value=0.4, step=0.1)))
    model.add(tfkl.SimpleRNN(
        units=hp_units,
        return_sequences=True,
        kernel_regularizer=reg,
        recurrent_regularizer=reg,
        bias_regularizer=reg)
        )
    model.add(tfkl.Dropout(hp.Float('dropout_rate', min_value=0.1, max_value=0.4, step=0.1)))
    model.add(tfkl.SimpleRNN(
        units=hp_units,
        kernel_regularizer=reg,
        recurrent_regularizer=reg,
        bias_regularizer=reg)
        )
    model.add(tfkl.Dropout(hp.Float('dropout_rate', min_value=0.1, max_value=0.4, step=0.1)))
    model.add(tfkl.Dense(units=1))


    hp_learning_rate = hp.Choice('learning_rate', values=[1e-1,1e-2, 1e-3, 1e-4,1e-5])
    model.compile(optimizer=Adam(learning_rate=hp_learning_rate),
                    loss='mean_squared_error',
                    metrics=['mean_squared_error'])
    
    return(model)

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True) #ensures model stops if best value does not increase for 10 epochs


tuner = kt.RandomSearch(
                    model_builder,
                    objective='val_loss',
                    max_trials=30,
                    executions_per_trial=1,
                    directory=".",
                    project_name=(company_name)
                    )





tuner.search(train_set_x, train_set_y, 
             epochs=1000, 
             validation_data=(test_set_x, test_set_y), 
             callbacks=[early_stopping])

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print("Best Hyperparameters: ", best_hps.values)