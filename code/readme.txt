All required external packages are outlined in the report

Models take in 3 command line arguments:

#1) FILE PATH OF THE DATA
#2) DATA FORMAT - "Date" or "Datetime" (Default = "Datetime")
#3) BOOLEAN VALUE FOR WHETHER XAI PLOTS ARE TO BE PRINTED - "True" or "False" (Default = "False")

Tuning files take in 2 command line arguments:

#1) FILE PATH OF THE DATA
#2) DATA FORMAT - "Date" or "Datetime" (Default = "Datetime")

Example command to run RNN model on the General Nvidia stock data:

py models/RNN.py data/general/Nvidia Datetime False
