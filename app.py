### Data Collection
import streamlit as st
import pandas_datareader as pdr
import yfinance as yf
import pandas as pd

import moment

def getGMTTime(utcString):
  date = moment.date(utcString, "%Y-%m-%d %H:%M:%S+00:00")
  return date.locale('Asia/Kolkata').date.strftime("%Y-%m-%d %H:%M:%S")


def load_data(ticker, previous_days, previous_data_freq):
    data = yf.download(ticker, period=previous_days, interval=previous_data_freq)    
    data.reset_index(inplace=True)
    for row in data.itertuples():
        date = data.at[row.Index, 'Datetime']   
        data.at[row.Index, 'Datetime'] = pd.Timestamp(str(getGMTTime(date)), tz=None)        
    return data

sidebar = st.sidebar

stocks = ('BTC-USD','EURUSD=X','GOOG', 'AAPL', 'MSFT', 'GME')
ticker = sidebar.selectbox('Select dataset for prediction', stocks)

period = sidebar.selectbox('Select Days', ('1d','2d','3d','4d','5d','6d'), index=1)
interval = sidebar.selectbox('Select Timeframe', ('1m', '2m', '5m', '15m', '30m', '60m'))
predTime = sidebar.selectbox('Select Prediction Time (in minutes)', (10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60), index=4)
epoch = sidebar.selectbox('Select Epochs', (1,10,20,30,40,50,60,70,80,90,100), index=2)

df = load_data(ticker, period, interval)

startTime=df['Datetime'].iloc[0]
endTime=df['Datetime'].iloc[-1]

csvName=(ticker+'-'+str(startTime)+'-to-'+str(endTime)+'.csv').replace(':', '_')

# df.to_csv(csvName)
st.write('Downloaded '+ticker+" Data")
st.table( df.loc[:, 'Datetime':'Close'].tail(8))

df1=df.reset_index()['Close']

import numpy as np

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))
df1=scaler.fit_transform(np.array(df1).reshape(-1,1))


##splitting dataset into train and test split
training_size=int(len(df1)*0.65)
test_size=len(df1)-training_size
train_data,test_data=df1[0:training_size,:],df1[training_size:len(df1),:1]

import numpy
# convert an array of values into a dataset matrix
def create_dataset(dataset, time_step=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-time_step-1):
		a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
		dataX.append(a)
		dataY.append(dataset[i + time_step, 0])
	return numpy.array(dataX), numpy.array(dataY)


# reshape into X=t,t+1,t+2,t+3 and Y=t+4
time_step = 100
X_train, y_train = create_dataset(train_data, time_step)
X_test, ytest = create_dataset(test_data, time_step)


# reshape input to be [samples, time steps, features] which is required for LSTM
X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)


# Create the Stacked LSTM model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM


model=Sequential()
model.add(LSTM(50,return_sequences=True,input_shape=(100,1)))
model.add(LSTM(50,return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam')

model.summary()

model.fit(X_train,y_train,validation_data=(X_test,ytest),epochs=epoch,batch_size=64,verbose=1)

# Lets Do the prediction and check performance metrics
train_predict=model.predict(X_train)
test_predict=model.predict(X_test)


# ##Transformback to original form
train_predict=scaler.inverse_transform(train_predict)
test_predict=scaler.inverse_transform(test_predict)


### Calculate RMSE performance metrics
import math
from sklearn.metrics import mean_squared_error
math.sqrt(mean_squared_error(y_train,train_predict))

### Test Data RMSE
math.sqrt(mean_squared_error(ytest,test_predict))


last_hundred=len(test_data)-100
x_input=test_data[last_hundred:].reshape(1,-1)

temp_input=list(x_input)
temp_input=temp_input[0].tolist()


# demonstrate prediction for next 10 days
from numpy import array

lst_output=[]
n_steps=100
i=0
while(i<predTime):
    
    if(len(temp_input)>100):        
        x_input=np.array(temp_input[1:])        
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        yhat = model.predict(x_input, verbose=0)
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]        
        lst_output.extend(yhat.tolist())
        i=i+1
    else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        temp_input.extend(yhat[0].tolist())
        lst_output.extend(yhat.tolist())
        i=i+1
    

day_new=np.arange(1,101)
day_pred=np.arange(101,131)

predictedRawValue=scaler.inverse_transform(lst_output)

from datetime import timedelta
import re

def printPrediction(rawData):
    data=[]
    tempTime=endTime
    for val in rawData:        
        offset=re.findall(r'-?\d+\.?\d*', interval)
        Minutes = int(offset[0])
        tempTime = tempTime + timedelta(minutes=Minutes)
        data.append({
            "Datetime": str(tempTime),
            "Close": val[0]
        })
    return data

st.write('Predicted future '+str(predTime)+' minutes forcast')
st.table(printPrediction(predictedRawValue))    





