#%%
import numpy as np
import pandas as pd
import h5py
import pickle
import matplotlib.pyplot as plt
import keras
from keras import backend as K
from keras.layers import Input, Dense
from keras.models import Model
from datetime import datetime, timedelta
from evaluate import test_model
#%%
def get_data_with_time():
    frames = []
    hours = []
    days_from_12 = []
    date = datetime(2017, 1, 2)
    delta_day = timedelta(days=1)
    #merge_file = h5py.File("./data/merge.h5", "w")
    while date <= datetime(2017, 3, 2):
        date_str = datetime.strftime(date, "%Y%m%d")
        hist_file = h5py.File("./data/hist/{}.h5"
            .format(date_str), "r")
        delta_day_from_12 = date - datetime(2017, 1, 2)
        for hour in range(9, 13):
            key = "hour: {}".format(hour)
            frames.append(hist_file[key][:])
            hours.append(hour)
            days_from_12.append(delta_day_from_12.days)
        hist_file.close()
        date += delta_day

    frames = np.array(frames)
    days_from_12 = np.array(days_from_12)
    hours = np.array(hours)
    return frames, days_from_12, hours
    #dataset = [frames, days_from_12, hours]
    #print(dataset)
    #merge_file.create_dataset("bot", data=dataset)
    #merge_file.close()

density, days, hours = get_data_with_time()
#%%
img_size = density[0].shape[0]*density[0].shape[1]
noise_size = 100
#inputs_real_img = Input(shape=(img_size, ), name="real_img")
inputs_noise_img = Input(shape=(noise_size, ), name="noise_img")
inputs_hour = Input(shape=(1, ), name="hour")

x = Dense(64, activation='relu')(inputs_noise_img)
x = keras.layers.concatenate([inputs_hour, x])
predictions = Dense(img_size)(x)
#%%
model = Model(inputs=[inputs_noise_img, inputs_hour],
    outputs=predictions)

def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1)) 

model.compile(optimizer = "adam", loss = root_mean_squared_error)
#%%
noise_samples = np.random.uniform(size=(density.shape[0], noise_size))
train_img = np.array([
    img.flatten().astype("float32")/100.0 for img in density])
#%%
model.fit(x=[noise_samples, hours], y=train_img, epochs=20, batch_size=10)
#%%   
test_model(model, 3, 12, 9)
test_model(model, 3, 12, 10)
test_model(model, 3, 12, 11)
test_model(model, 3, 12, 12)