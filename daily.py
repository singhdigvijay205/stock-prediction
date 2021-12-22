# pip install streamlit fbprophet yfinance plotly
import streamlit as st
from datetime import date

import yfinance as yf
from fbprophet import Prophet
import moment
import pytz

st.title('Stock Forecast App')

local = pytz.timezone('Asia/Kolkata')

stocks = ('BTC-USD','EURUSD=X','GOOG', 'AAPL', 'MSFT', 'GME')
selected_stock = st.selectbox('Select dataset for prediction', stocks)

previous_days = st.selectbox('Select days', ('1d','2d','3d','4d','5d','6d','7d','8d','9d'))
previous_data_freq = st.selectbox('Select Data timeframe', ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d'))

def getGMTTime(utcString):
  date = moment.date(utcString, "%Y-%m-%d %H:%M:%S+00:00")
  return date.locale('Asia/Kolkata').date.strftime("%Y-%m-%d %H:%M:%S")

# def getGMTTimeForcast(utcString):
#   date = moment.date(utcString, "%Y-%m-%dT%H:%M:%S")
#   return date.strftime("%Y-%m-%d   %H:%M:%S")   

@st.cache
def load_data(ticker, previous_days, previous_data_freq):
    data = yf.download(ticker, period=previous_days, interval=previous_data_freq)    
    data.reset_index(inplace=True)
    for row in data.itertuples():
        date = data.at[row.Index, 'Datetime']   
        temp = str(getGMTTime(date)).replace("+00:00", "")
        st.text(temp)
        data.at[row.Index, 'Datetime'] = '1234'
    return data

	
data_load_state = st.text('Loading data...')
data = load_data(selected_stock, previous_days, previous_data_freq)
data_load_state.text('')
st.subheader('Previous '+previous_days+ ' of data with '+previous_data_freq+' frequency')
st.write(data.tail(8))


n_hours = st.slider('Choose Hours of prediction:', 4, 24)
period = n_hours

# Predict forecast with Prophet.
df_train = data[['Datetime','Close']]
df_train = df_train.rename(columns={"Datetime": "ds", "Close": "y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period, freq='H')
forecast = m.predict(future)

# for row in forecast.itertuples():
#     date = forecast.at[row.Index, 'ds']   
#     forecast.at[row.Index, 'ds'] = str(getGMTTimeForcast(str(date)))

# Show and plot forecast
st.subheader('Forecast data for '+str(n_hours)+' Hours')
st.write(forecast.tail(n_hours))