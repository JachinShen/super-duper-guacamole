#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
from keras import backend as K
from keras.layers import Input, Dense
from keras.models import Model
from evaluate import test_model, deploy_model
from submit import submit_csv
from dataset import get_hist_with_time
#%%
density, weekday, hours = get_hist_with_time()
#%%
img_size = density[0].shape[0]*density[0].shape[1]
noise_size = 100
inputs_noise_img = Input(shape=(noise_size, ), name="noise_img")
inputs_hour = Input(shape=(1, ), name="hour")
inputs_weekday = Input(shape=(1, ), name="weekday")

x = Dense(64, activation='relu')(inputs_noise_img)
x = keras.layers.concatenate([inputs_weekday, x])
x = keras.layers.concatenate([inputs_hour, x])
predictions = Dense(img_size)(x)
#%%
model = Model(inputs=[inputs_noise_img, inputs_hour, inputs_weekday],
    outputs=predictions)

def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1)) 

model.compile(optimizer = "adam", loss = root_mean_squared_error)
#%%
noise_samples = np.random.uniform(size=(density.shape[0], noise_size))
hours = (hours.astype("float32") - 8) / 4.0
weekday = (weekday.astype("float32") + 1) / 7.0
train_img = np.array([
    img.flatten().astype("float32")/100.0 for img in density])
#%%
model.fit(x=[noise_samples, hours, weekday], y=train_img, epochs=20, batch_size=10)
#%%  
#for day in range(3, 13):
    #for hour in range(9, 13):
        #test_model(model, 3, day, hour) 
hists = []
for hour in range(9, 13):
    hist = deploy_model(model, 3, 12, hour)
    hists.append(hist)
submit_csv(hists)