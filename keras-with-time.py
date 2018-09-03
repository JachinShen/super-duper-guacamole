# author: JachinShen(jachinshen@foxmail.com)
from datetime import datetime, timedelta

import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras import backend as K
from keras.layers import Add, Dense, Dropout, Input, Concatenate
from keras.models import Model

import quantize
from dataset import get_hist_with_time
from evaluate import deploy_model, test_model
from submit import get_hour_density, submit_csv

is_test = True
img_size = (quantize.lat_ctr() - 1) * (quantize.lon_ctr() - 1)
noise_size = 100

np.random.seed(2333)

def preprocess_data():
    if is_test:
        density, weekday, hours = get_hist_with_time(
            datetime(2017, 2, 6), datetime(2017, 3, 5))
    else:
        density, weekday, hours = get_hist_with_time(
            datetime(2017, 2, 6), datetime(2017, 3, 12))

    noise_samples = np.random.uniform(size=(density.shape[0], noise_size))
    hours = (hours.astype("float32") - 8) / 14.0
    weekday = (weekday.astype("float32") + 1) / 7.0
    #density = density.reshape((*density.shape, 1))
    train_img = np.array([
        img.flatten().astype("float32")/100.0 for img in density])

    X = [noise_samples, hours, weekday]
    y = train_img
    return X, y


def build_model():
    inputs_noise_img = Input(shape=(noise_size, ), name="noise_img")
    inputs_hour = Input(shape=(1, ), name="hour")
    inputs_weekday = Input(shape=(1, ), name="weekday")

    print(inputs_noise_img)

    #noise_dense = Dense(32, activation="relu")(inputs_noise_img)
    hour_dense_16 = Dense(16, activation="relu")(inputs_hour)
    hour_dense_32 = Dense(32, activation="relu")(hour_dense_16)
    hour_dense_64 = Dense(64, activation="relu")(hour_dense_32)
    weekday_dense_16 = Dense(16, activation="relu")(inputs_weekday)
    weekday_dense_32 = Dense(32, activation="relu")(weekday_dense_16)
    weekday_dense_64 = Dense(64, activation="relu")(weekday_dense_32)
    mainline = Concatenate()([hour_dense_16, weekday_dense_16])
    mainline = Dense(32, activation="relu")(mainline)
    mainline = Add()([mainline, hour_dense_32, weekday_dense_32])
    mainline = Dense(64, activation="relu")(mainline)
    mainline = Dropout(0.2)(mainline)
    mainline = Concatenate()([mainline, hour_dense_64, weekday_dense_64])
    predictions = Dense(img_size, activation="relu")(mainline)
    model = Model(inputs=[inputs_noise_img, inputs_hour, inputs_weekday],
                  outputs=predictions)
    return model


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


if __name__ == "__main__":
    model = build_model()
    model.compile(optimizer="adam", loss='mean_squared_error')
    X, y = preprocess_data()
    model.fit(x=X, y=y, epochs=40, batch_size=7)

    if is_test:
        errors = []
        date = datetime(2017, 3, 6)
        delta_day = timedelta(days=1)
        while date <= datetime(2017, 3, 12):
            for hour in range(9, 23):
                errors.append(test_model(model, date, hour))
            date += delta_day
        print("Avearage RMSE:{}".format(np.array(errors).mean()))
    else:
        frames = []
        date = datetime(2017, 3, 13)
        delta_day = timedelta(days=1)
        while date <= datetime(2017, 3, 26):
            for hour in range(9, 23):
                hist = deploy_model(model, date, hour)
                X_sub = get_hour_density(hist, date, hour)
                frames.append(X_sub)
            date += delta_day
        submission = pd.concat(frames).drop(
            ['latitude_range', 'longitude_range'], axis=1)
        submission.to_csv("./submission.csv", index=False)
