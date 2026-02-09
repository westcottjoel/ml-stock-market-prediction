import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from alibi.explainers import IntegratedGradients
import shap
import sys
import time

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

data = data.filter([data_format,'Open','Close','Low','High','Volume'])

#CLEAN UP ANY MISSING VALUES

data = data.bfill()

#EXTRACT FEATURES

#Smoothed Volume

data['Volume-Smoothed'] = data['Volume'] / data['Volume'].rolling(window=20).mean()

#Relative Strength Index (RSI)

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

# # Exponential Moving Averages (EMA)
data['EMA_10'] = data['Close'].ewm(span=10, adjust=False).mean()
data['EMA_50'] = data['Close'].ewm(span=50, adjust=False).mean()

# Moving Average Convergence Divergence (MACD)
data['MACD'] = data['Close'].ewm(span=12, adjust=False).mean() - data['Close'].ewm(span=26, adjust=False).mean()

# #LAG FEATURES
data['LAG_3'] = data['Close'].shift(3)
data['LAG_4'] = data['Close'].shift(4)
data['LAG_5'] = data['Close'].shift(5)
data['LAG_6'] = data['Close'].shift(6)
data['LAG_7'] = data['Close'].shift(7)
data['LAG_8'] = data['Close'].shift(8)
data['LAG_9'] = data['Close'].shift(9)
data['LAG_10'] = data['Close'].shift(10)

# #DROP NAN VALUES

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

train_set_y = train_set_y[:,1] #just the close price column is being predicted
test_set_y = test_set_y[:,1]
val_set_y = val_set_y[:,1]



#DEFINE HYPERPARAMETERS

groups = {
    "growth": [
        "Tesla",
        "Nvidia",
        "Meta"
    ],
    "declining":[
        "GlobalFounderies",
        "Intel",
        "Walgreens Boots Alliance"
    ],
    "stable":[
        "Procter & Gamble",
        "Johnson & Johnson",
        "Coca-Cola Co",
    ]
}

if company_name in groups["growth"] :
    lstm_units = 128
    dropout_rate = 0.2
    learning_rate = 1e-3
if company_name in groups["declining"] :
    lstm_units = 128
    dropout_rate = 0.4
    learning_rate = 1e-3
if company_name in groups["stable"] :
    lstm_units = 128
    dropout_rate = 0.1
    learning_rate = 1e-4
else:
    print("default hyperparameter values used")
    lstm_units = 128
    dropout_rate = 0.1
    learning_rate = 1e-3

#BUILD LSTM MODEL

training_start = time.time()

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,            
    restore_best_weights=True 
)

lstm_model = Sequential()
lstm_model.add(LSTM(units=lstm_units, return_sequences=True,
    input_shape=(train_set_x.shape[1], train_set_x.shape[2])))
lstm_model.add(Dropout(dropout_rate))
lstm_model.add(LSTM(units=lstm_units,return_sequences=True))
lstm_model.add(Dropout(dropout_rate))
lstm_model.add(LSTM(units=lstm_units))
lstm_model.add(Dropout(dropout_rate))
lstm_model.add(Dense(units=1))

optimizer = Adam(learning_rate=learning_rate)

lstm_model.compile(optimizer=optimizer, loss='mean_squared_error')

print(lstm_model.summary())

lstm_model.fit(train_set_x,
               train_set_y,
               epochs=1000,
               batch_size=32,
               verbose=2,
               validation_data=(val_set_x,val_set_y),
               callbacks=[early_stop]
               ) 

training_end = time.time()

#MAKE PREDICTIONS

prediction_y = lstm_model.predict(test_set_x)

#RESCALE PREDICTION, ACTUAL VALUES

features=test_set_x.shape[2]

rescale_df = pd.DataFrame(np.zeros((len(prediction_y),features)))
rescale_df.iloc[:,1] = prediction_y 
prediction_y = scaler.inverse_transform(rescale_df)[:,1]

rescale_df = pd.DataFrame(np.zeros((len(test_set_y),features)))
rescale_df.iloc[:,1] = test_set_y

test_set_y = scaler.inverse_transform(rescale_df)

#EVALUATION AGAINST CLOSING VALUE

RMSE_value = math.sqrt(mean_squared_error(test_set_y[:,1],prediction_y))

print("Evaluation of LSTM stock prediction for " + company_name)
print("----------------------------------------------------")
print("Time Taken to Train Model")
print(training_end - training_start)
print("RMSE: " + str(RMSE_value))

#PLOT PREDICTION vs TEST SET Y

plt.plot(test_set_y[:,1], label='Actual Closing Price')
plt.plot(prediction_y, label='Predicted')

tick_positions = np.linspace(0, len(dates_test) - 1, 6, dtype=int)
tick_labels = [pd.to_datetime(dates_test[i]).strftime('%d/%m/%Y') for i in tick_positions]
plt.xticks(tick_positions, tick_labels)

plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.title("Actual vs predicted stock prices of " + company_name + " using LSTM prediction")
plt.legend()
plt.show()

#XAI GRAPHS (IF REQUESTED)

if len(sys.argv)>=4 and sys.argv[3] == "True":

    #INITIALIZE LSTM EXPLAINER

    IG_explainer = IntegratedGradients(lstm_model, layer=None)
    explanation = IG_explainer.explain(test_set_x, baselines=np.zeros_like(test_set_x),target=None)
    relevance_scores = explanation.attributions
    relevance_scores = np.squeeze(relevance_scores, axis=0)

    #PLOT AVERAGE RELEVANCE OVER TIME STEPS

    avg_time_relevance = relevance_scores.mean(axis=(0, 2))
    plt.plot(avg_time_relevance, marker='o')
    plt.title("Average Relevance Across Time Steps")
    plt.xlabel("Time Step")
    plt.ylabel("Relevance")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    #PLOT AVERAGE RELEVANCE OVER TIME STEPS PER FEATURE

    mean_over_samples = relevance_scores.mean(axis=0)  
    for i in range(0,features):
        plt.plot(mean_over_samples[:, i], label=data.columns[i])
    plt.title("Average Relevance of Each Feature Across Time Steps")
    plt.xlabel("Time Step")
    plt.ylabel("Relevance")
    plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1.0))
    plt.tight_layout()
    plt.show()

    #PLOT HEATMAP OF FEATURE RELEVANCE USING SHAP
    n_samples = min(100, train_set_x.shape[0])
    background = train_set_x[np.random.choice(train_set_x.shape[0], n_samples, replace=False)]

    explainer = shap.GradientExplainer(lstm_model, background)
    shap_values = explainer.shap_values(test_set_x)
    shap_values = shap_values.reshape(len(test_set_x), -1)

    per_feature_shap_values =[]
    labels=data.columns
    for i in range(0,features):
        feature_shap_values = shap_values[:,i::features]
        per_feature_shap_values.append(feature_shap_values.flatten())
    per_feature_shap_values = np.array(per_feature_shap_values).T
    df = pd.DataFrame(per_feature_shap_values,columns=labels)
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f', cbar=True, linewidths=1)
    plt.title('Correlation Heatmap')
    plt.gca().invert_yaxis()
    plt.show()

    #PLOT RESIDUALS IN HISTOGRAM
    residuals = test_set_y[:,1] - prediction_y
    sns.histplot(residuals, kde=True)
    plt.title('Residuals Distribution')
    plt.xlabel('Residual')
    plt.show()