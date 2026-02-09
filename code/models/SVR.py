import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
import sys
from pathlib import Path
import shap
import time
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from datetime import datetime

#MODEL TAKES IN 4 CMDLINE ARGUMENTS -
#1) FILE PATH OF THE DATA
#2) DATA FORMAT - "Date" or "Datetime" (Default = "Datetime")
#3) BOOLEAN VALUE FOR WHETHER XAI PLOTS ARE TO BE PRINTED - "True" or "False" (Default = "False")
#READ IN DATA

file_path = sys.argv[1]
company_name = Path(file_path).name
data_format = sys.argv[2]

data = pd.read_csv(file_path,delimiter=',',header=0,parse_dates=True)

#READ IN CMDLINE ARGUMENTS

file_path = sys.argv[1]
company_name = Path(file_path).name
data_format = sys.argv[2]

#READ IN DATA

data = pd.read_csv(file_path,delimiter=',',header=0,parse_dates=True)

#FILTER OUT UNNEEDED COLUMNS

data = data.filter([data_format,'Close'])

#CLEAN UP DATA - BACKFILL EMPTY CELLS,

data = data.bfill()

#SEPARATE DATE FOR FUTURE PLOTTING

dates = np.array(data[data_format])
dates = pd.to_datetime(dates)
data= data.drop(columns=[data_format])

# APPLY SLIDING WINDOWS ALGORITHM
window_size = 60
dates=dates[window_size:]
x_values = []
y_values = []

for i in range(window_size, len(data)):
    x_values.append(data.values[i - window_size:i, 0])
    y_values.append(data.values[i, 0])

x_values = np.array(x_values)
y_values = np.array(y_values)

# SPLIT INTO TRAIN/TEST

train_set_x, test_set_x, train_set_y, test_set_y, dates_train, dates_test = train_test_split(
    x_values, y_values, dates, test_size=0.2, shuffle=False
)

#SCALE VALUES

scaler = MinMaxScaler()
scaler.fit(train_set_x)

train_set_x = scaler.transform(train_set_x)
test_set_x = scaler.transform(test_set_x)



# FIT MODEL
model = SVR(
    kernel='linear', 
    C=0.5, 
    epsilon=0.01)

training_start = time.time()
model.fit(train_set_x, train_set_y)
training_end = time.time()

# PREDICT
final_prediction = model.predict(test_set_x)

#EVALUATE

RMSE_value = math.sqrt(mean_squared_error(test_set_y,final_prediction))

print("Evaluation of SVR stock prediction for " + company_name)
print("----------------------------------------------------")
print("Time Taken to Train Model")
print(training_end - training_start)
print("RMSE: " + str(RMSE_value))

#PLOT PREDICTION vs TEST SET Y

plt.plot(test_set_y, label='Actual Closing Price')
plt.plot(final_prediction, label='Predicted')

tick_positions = np.linspace(0, len(dates_test) - 1, 6, dtype=int)
tick_labels = [pd.to_datetime(dates_test[i]).strftime('%d/%m/%Y') for i in tick_positions]
plt.xticks(tick_positions, tick_labels)

plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.title("Actual vs predicted stock prices of " + company_name + " using SVR prediction")
plt.legend()
plt.show()

#XAI GRAPHS (IF REQUESTED)

if len(sys.argv)>=4 and sys.argv[3] == "True":
    #INITIALIZE SHAP EXPLAINER FOR MODEL

    shap_explainer = shap.Explainer(model.predict, train_set_x)
    shap_values = shap_explainer(train_set_x)
    relevance_scores = shap_values.values
    
    avg_time_relevance = np.mean(np.abs(relevance_scores),axis=0)
    plt.plot(avg_time_relevance, marker='o')
    plt.title("Average Relevance Across Time Steps")
    plt.xlabel("Time Step")
    plt.ylabel("Relevance")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    #PLOT HISTOGRAM OF RESIDUALS
    residuals = test_set_y - final_prediction
    sns.histplot(residuals, kde=True)
    plt.title("Residuals Distribution")
    plt.xlabel('Residual')
    plt.show()

