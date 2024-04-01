
# Method implemented here: https://github.com/jsyoon0823/TimeGAN/blob/master/data_loading.py
# Originally used in TimeGAN research

import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import yfinance as yf



def real_data_loading(stock_ticker, start_date ='2009-01-02', end_date = '2023-12-31', seq_len = 24):
    """Load and preprocess stock data from yahoo finance.
    Args:
      - stock_ticker: the name of the stock to download from yahoo finance
      - start_date: the start date of the stock data 
      - end_date: the end date of the stock data
      - seq_len: sequence length

    Returns:
      - data: preprocessed data.
    """
    #download stock data
    stock_df = yf.download(stock_ticker, start = start_date, end = end_date)
    stock_df = stock_df[['Open', 'High', 'Low', 'Close', 'Adj Close']]

    scaler = MinMaxScaler()
    ori_data = scaler.fit_transform(stock_df)

    # Preprocess the dataset
    temp_data = []
    # Cut data by sequence length
    for i in range(0, len(ori_data) - seq_len):
        _x = ori_data[i:i + seq_len]
        temp_data.append(_x)

    # Mix the datasets (to make it similar to i.i.d)
    idx = np.random.permutation(len(temp_data))
    data = []
    for i in range(len(temp_data)):
        data.append(temp_data[idx[i]])
    return data

