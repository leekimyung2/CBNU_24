import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, LeakyReLU, Input, Reshape, Flatten
from tensorflow.keras.optimizers import Adam


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


# 3. LSTM 기반 생성자 구축
def build_generator(sequence_length, latent_dim):
    model = Sequential()
    model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(sequence_length, latent_dim)))
    model.add(LSTM(50, activation='relu'))
    model.add(Dense(sequence_length))
    model.add(Reshape((sequence_length, 1)))
    return model


# 4. 판별자 구축
def build_discriminator(sequence_length):
    model = Sequential()
    model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(sequence_length, 1)))
    model.add(LSTM(50, activation='relu'))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    return model


# 5. GAN 모델 구축
def build_gan(generator, discriminator):
    discriminator.trainable = False
    gan_input = Input(shape=(sequence_length, latent_dim))
    x = generator(gan_input)
    gan_output = discriminator(x)
    gan = Model(gan_input, gan_output)
    return gan


# 6. GAN 학습
def train_gan(generator, discriminator, gan, X_train, latent_dim, epochs=10000, batch_size=64):
    half_batch = int(batch_size / 2)
    sequence_length = X_train.shape[1]

    for epoch in range(epochs):
        idx = np.random.randint(0, X_train.shape[0], half_batch)
        real_data = X_train[idx]

        noise = np.random.normal(0, 1, (half_batch, sequence_length, latent_dim))
        generated_data = generator.predict(noise)

        d_loss_real = discriminator.train_on_batch(real_data, np.ones((half_batch, 1)))
        d_loss_fake = discriminator.train_on_batch(generated_data, np.zeros((half_batch, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        noise = np.random.normal(0, 1, (batch_size, sequence_length, latent_dim))
        valid_y = np.array([1] * batch_size)
        g_loss = gan.train_on_batch(noise, valid_y)

        if epoch % 1000 == 0:
            print(f"{epoch} [D loss: {d_loss}] [G loss: {g_loss}]")


# 7. 이상 탐지
def detect_anomalies(generator, X_test, scaler, latent_dim, threshold_factor=3):
    sequence_length = X_test.shape[1]
    noise = np.random.normal(0, 1, (X_test.shape[0], sequence_length, latent_dim))
    generated_data = generator.predict(noise)
    generated_data = scaler.inverse_transform(generated_data.reshape(-1, 1)).reshape(-1, sequence_length)
    X_test_inv = scaler.inverse_transform(X_test.reshape(-1, 1)).reshape(-1, sequence_length)

    errors = np.mean(np.abs(X_test_inv - generated_data), axis=1)
    threshold = np.mean(errors) + threshold_factor * np.std(errors)
    anomalies = errors > threshold

    plt.figure(figsize=(15, 5))
    plt.plot(X_test_inv.flatten(), label='Actual')
    plt.plot(generated_data.flatten(), label='Generated')
    plt.scatter(np.where(anomalies)[0], X_test_inv.flatten()[anomalies], color='r', label='Anomalies')
    plt.legend()
    plt.title('Anomaly Detection in Log Data using LSTM-based GAN')
    plt.xlabel('Time Steps')
    plt.ylabel('Value')
    plt.show()


# 전체 파이프라인 실행
if __name__ == "__main__":
    df = prepare_data()
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)

    sequence_length = X_train.shape[1]
    latent_dim = 10

    generator = build_generator(sequence_length, latent_dim)
    discriminator = build_discriminator(sequence_length)
    discriminator.compile(optimizer=Adam(0.0002, 0.5), loss='binary_crossentropy', metrics=['accuracy'])

    gan = build_gan(generator, discriminator)
    gan.compile(optimizer=Adam(0.0002, 0.5), loss='binary_crossentropy')

    train_gan(generator, discriminator, gan, X_train, latent_dim)

    detect_anomalies(generator, X_test, scaler, latent_dim)
