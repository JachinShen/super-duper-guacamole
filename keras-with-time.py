# author: JachinShen(jachinshen@foxmail.com)
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
from keras import backend as K
from keras.layers import Input, Dense, Add, Dropout
from keras.models import Model
from datetime import datetime, timedelta
from evaluate import test_model, deploy_model
from submit import submit_csv, get_hour_density
from dataset import get_hist_with_time
import quantize
is_test = True
img_size = (quantize.lat_ctr() - 1) * (quantize.lon_ctr() - 1)
noise_size = 100
def preprocess_data():
    if is_test:
        density, weekday, hours = get_hist_with_time(
            datetime(2017, 2, 6), datetime(2017, 3, 5))
    else:
        density, weekday, hours = get_hist_with_time(
            datetime(2017, 2, 6), datetime(2017, 3, 12))

    np.random.seed(233)
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

    noise_dense = Dense(64, activation="relu")(inputs_noise_img)
    hour_dense = Dense(64, activation="relu")(inputs_hour)
    x = Add()([noise_dense, hour_dense])
    x = Dense(64, activation="relu")(x)
    x = keras.layers.concatenate([inputs_weekday, x])
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.2)(x)
    predictions = Dense(img_size)(x)
    model = Model(inputs=[inputs_noise_img, inputs_hour, inputs_weekday],
        outputs=predictions)
    return model

def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1)) 

#%%
if __name__ == "__main__":
    model = build_model()
    model.compile(optimizer = "adam", loss = root_mean_squared_error)
    for epochs in range(5):
        X, y = preprocess_data()
        model.fit(x=X, y=y, epochs=5, batch_size=7)

    if is_test:
        errors = []
        date = datetime(2017, 3, 6)
        delta_day = timedelta(days = 1)
        while date <= datetime(2017, 3, 12):
            for hour in range(9, 23):
                errors.append(test_model(model, date, hour))
            date += delta_day
        print("Avearage RMSE:{}".format(np.array(errors).mean()))
    else:
        frames = []
        date = datetime(2017, 3, 13)
        delta_day = timedelta(days = 1)
        while date <= datetime(2017, 3, 26):
            for hour in range(9, 23):
                hist = deploy_model(model, date, hour)
                X_sub = get_hour_density(hist, date, hour)
                frames.append(X_sub)
            date += delta_day
        submission = pd.concat(frames).drop(
            ['latitude_range', 'longitude_range'], axis=1)
        submission.to_csv("./submission.csv", index=False)