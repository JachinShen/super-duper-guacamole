# author: JachinShen(jachinshen@foxmail.com)
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
from keras import backend as K
from keras.layers import Input, Dense, ConvLSTM2D
from keras.models import Model
from datetime import datetime
from evaluate import test_model, deploy_model
from submit import submit_csv
from dataset import get_hist_with_time
import quantize
#%%
is_test = True
img_size = (quantize.lat_ctr() - 1) * (quantize.lon_ctr() - 1)
noise_size = (10, 10, 1)
#%%
def preprocess_data():
    if is_test:
        density, weekday, hours = get_hist_with_time(datetime(2017, 3, 2))
    else:
        density, weekday, hours = get_hist_with_time(datetime(2017, 3, 12))

    noise_samples = np.random.uniform(size=(density.shape[0], *noise_size))
    hours = (hours.astype("float32") - 8) / 4.0
    weekday = (weekday.astype("float32") + 1) / 7.0
    density = density.reshape((*density.shape, 1))
    train_img = np.array([
        img.astype("float32")/100.0 for img in density])

    X = [noise_samples, hours, weekday]
    y = train_img
    return X, y

def build_model():
    inputs_noise_img = Input(shape=(10, 10, 1), name="noise_img")
    inputs_hour = Input(shape=(None, 1), name="hour")
    inputs_weekday = Input(shape=(None, 1), name="weekday")

    print(inputs_noise_img)

    x = ConvLSTM2D(filters=32, kernel_size=(3,3),
        padding="same", return_sequences=True)(inputs_noise_img)
    #x = keras.layers.concatenate([inputs_weekday, x])
    #x = keras.layers.concatenate([inputs_hour, x])
    predictions = Dense(img_size)(x)
    model = Model(inputs=[inputs_noise_img, inputs_hour, inputs_weekday],
        outputs=predictions)
    return model

def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1)) 

if __name__ == "__main__":
    X, y = preprocess_data()
    print(X[0].shape)
    print(y.shape)
    model = build_model()
    model.compile(optimizer = "adam", loss = root_mean_squared_error)
    model.fit(x=X, y=y, epochs=20, batch_size=10)

    if is_test:
        for day in range(3, 13):
            for hour in range(9, 13):
                test_model(model, 3, day, hour) 
    else:
        hists = []
        for hour in range(9, 13):
            hist = deploy_model(model, 3, 12, hour)
            hists.append(hist)
        submit_csv(hists)