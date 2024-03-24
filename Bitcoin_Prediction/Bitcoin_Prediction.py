import numpy as np
import pandas as pd
import os


from keras.models import Sequential
from keras.layers import LSTM, Dense, Bidirectional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt

def read_data():
    script_dir = os.path.dirname(os.path.realpath(__file__))
    file_path = 'Data/bitcoin_2017_to_2023.csv'
    os.chdir(script_dir)
    df = pd.read_csv(file_path)
    return df

# TODO: 將資料進行分割。
# 利用前 split_num (10)天的資料來預測第 split_num (11)天的收盤價
def split_data(datas, labels, split_num = 10):
    max_len = len(datas)
    x, y = [], []
    for i in range(max_len - split_num - 1):
        x.append(datas[i: i+split_num])
        y.append(labels[split_num+i+1])

    return np.array(x), np.array(y)

def plot_data(df: type[pd.DataFrame]):
    df_copy = df.copy()
    df_copy['timestamp'] = pd.to_datetime(df_copy['timestamp'])

    df_copy.index = df_copy['timestamp']

    # Resampling to daily frequency
    df_copy = df_copy.resample('D').mean()
    
    # Plots
    fig = plt.figure(figsize=[15, 7])
    plt.suptitle('Bitcoin exchanges, mean USD', fontsize=22)

    # plt.subplot(221)
    plt.plot(df_copy.open, '-', label='By Days')
    plt.legend()

    plt.show()

if __name__ == "__main__":
    df = read_data()
    # 將資料以日期重新 sample
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.index = df['timestamp']
    df = df.resample('D').mean()
    # 去除日期
    df = df.drop(columns=['timestamp'])
    sc = MinMaxScaler()
    print(df)
    x = sc.fit_transform(df)
    # 取出收盤價當作標籤
    y = x[:,3]

    x_train, x_valid, y_train, y_valid = train_test_split(x, y, train_size=0.8, random_state=22, shuffle=False)

    x_train, y_train = split_data(x_train, y_train)
    x_valid, y_valid = split_data(x_valid, y_valid)

    model = Sequential()

    model.add(Bidirectional
                (LSTM(
                        64,
                        input_shape=(x_train.shape[1],x_train.shape[-1]),
                        return_sequences=True,
                        activation='tanh')
                )
             )
    model.add(Bidirectional(LSTM(128, return_sequences=False, activation='relu')))
    model.add(Dense(1))

    model.compile(loss='mean_squared_error', optimizer='adam')
    history = model.fit(x_train, y_train,
                        batch_size=64,
                        epochs=10,
                        verbose=1,
                        validation_data=(x_valid, y_valid))
    
    y_pred = model.predict(x_valid)

    # 反正規化 (用途：顯示預測的比特幣美元值)
    # Note : 需先 fit 原本的 2D array, 才可以執行反正規化
    y_pred_new = np.zeros(shape=(len(y_pred), 9))
    y_pred_new[:,3] = y_pred[:,0]
    y_pred_ori = sc.inverse_transform(y_pred_new)[:,3]
    y_valid_new = np.zeros(shape=(len(y_valid), 9))
    y_valid_new[:,3] = y_valid
    y_valid_ori = sc.inverse_transform(y_valid_new)[:,3]

    plt.figure(figsize=(15,7))
    plt.plot(y_valid_ori)
    plt.plot(y_pred_ori, color='r', ls='--')

    plt.title('Bitcoin exchanges, by day')
    plt.ylabel('Mean USD')
    plt.xlabel('Days')

    plt.legend(['Real open price', 'Predicted price'], loc='upper left')
    plt.show()