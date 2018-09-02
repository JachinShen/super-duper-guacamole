# author: JachinShen(jachinshen@foxmail.com)
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential

import quantize
from dataset import get_hist_as_image
from evaluate import deploy_model, test_model
from submit import get_hour_density, submit_csv

is_test = False
img_size = ((quantize.lat_ctr() - 1), (quantize.lon_ctr() - 1))
noise_size = (10, 10)
frames_per_sample = 14


def preprocess_data():
    if is_test:
        density = get_hist_as_image(
            datetime(2017, 2, 6), datetime(2017, 3, 5))
    else:
        density = get_hist_as_image(
            datetime(2017, 2, 6), datetime(2017, 3, 12))

    train_img = density.astype("float32") / 100.0

    X = np.array([frames[:-1] for frames in train_img])
    y = np.array([frames[1:] for frames in train_img])
    return X, y


def build_model():
    seq = Sequential()
    seq.add(ConvLSTM2D(filters=32, kernel_size=(3, 3),
                       input_shape=(None, *img_size, 1),
                       padding='same', return_sequences=True))
    seq.add(BatchNormalization())
    seq.add(Conv3D(filters=1, kernel_size=(3, 3, 3),
                   activation="sigmoid", padding='same', data_format='channels_last'))
    return seq

if __name__ == "__main__":
    model = build_model()
    model.compile(optimizer="adam", loss='mean_squared_error')
    for epochs in range(1):
        X, y = preprocess_data()
        print(X.shape)
        model.fit(x=X, y=y, epochs=5, batch_size=1)

'''
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

'''
