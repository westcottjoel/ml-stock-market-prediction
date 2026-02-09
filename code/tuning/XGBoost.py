import pandas as pd
import numpy as np
import xgboost
import sys
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import GridSearchCV

#MODEL TAKES IN 4 CMDLINE ARGUMENTS -
#1) FILE PATH OF THE DATA
#2) DATA FORMAT - "Date" or "Datetime" (Default = "Datetime")

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

#SCALE DATA
scaler = MinMaxScaler()
scaler.fit(train_set_x)

train_set_x = scaler.transform(train_set_x)
test_set_x = scaler.transform(test_set_x)

print("----")

print(company_name)

parameter_grid = {
    "booster":["gblinear","gbtree","dart"],
    "n_estimators":[100,250,500,1000,2500,5000,10000],
    "learning_rate":[0.001, 0.01, 0.1, 0.25,0.5],
    "reg_alpha" : [0,1e-5,1e-4,1e-3,1e-2,1e-2],
    "reg_lambda" :[0,1e-5,1e-4,1e-3,1e-2,1e-2]
}

model = xgboost.XGBRegressor()

grid_search = GridSearchCV(
    estimator=model, param_grid=parameter_grid,
    scoring='neg_mean_squared_error', cv=3,
    verbose=2, n_jobs=-1
)
grid_search.fit(train_set_x,train_set_y)

print("Best Parameters:", grid_search.best_params_)