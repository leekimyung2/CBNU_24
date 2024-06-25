import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # '0': 모든 로그, '1': INFO 로그, '2': WARN 로그, '3': ERROR 로그만 출력

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# 1. 데이터 준비
def prepare_data():
    np.random.seed(0)
    time_steps = 1000
    data = np.sin(np.linspace(0, 50, time_steps)) + np.random.normal(0, 0.5, time_steps)
    df = pd.DataFrame(data, columns=['value'])
    plt.plot(df['value'])
    plt.title('Synthetic Log Data')
    plt.xlabel('Time Steps')
    plt.ylabel('Value')
    plt.show()
    return df

# 2. 데이터 전처리
def create_sequences(data, sequence_length):
    sequences = []
    labels = []
    for i in range(len(data) - sequence_length):
        sequences.append(data[i:i + sequence_length])
        labels.append(data[i + sequence_length])
    return np.array(sequences), np.array(labels)

def preprocess_data(df, sequence_length=50):
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(df)
    X, y = create_sequences(data_scaled, sequence_length)
    split_ratio = 0.8
    split_index = int(len(X) * split_ratio)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    return X_train, X_test, y_train, y_test, scaler

# 3. LSTM 모델 구축 및 훈련
def build_and_train_model(X_train, y_train, sequence_length=50, epochs=20, batch_size=32):
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(sequence_length, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1)
    return model

# 4. 이상 탐지
def detect_anomalies(model, X_test, y_test, scaler, threshold_factor=3):
    y_pred = model.predict(X_test)
    y_test_inv = scaler.inverse_transform(y_test)
    y_pred_inv = scaler.inverse_transform(y_pred)
    errors = np.abs(y_test_inv - y_pred_inv)
    threshold = np.mean(errors) + threshold_factor * np.std(errors)
    anomalies = errors > threshold
    plt.figure(figsize=(15, 5))
    plt.plot(y_test_inv, label='Actual')
    plt.plot(y_pred_inv, label='Predicted')
    plt.scatter(np.where(anomalies)[0], y_test_inv[anomalies], color='r', label='Anomalies')
    plt.legend()
    plt.title('Anomaly Detection in Log Data')
    plt.xlabel('Time Steps')
    plt.ylabel('Value')
    plt.show()

# 전체 파이프라인 실행
if __name__ == "__main__":
    df = prepare_data()
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)
    model = build_and_train_model(X_train, y_train)
    detect_anomalies(model, X_test, y_test, scaler)