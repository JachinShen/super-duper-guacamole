# author:JachinShen(jachinshen@foxmail.com)
from datetime import datetime, timedelta

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import quantize
from submit import get_hour_density
from dataset import get_weather_dict

weather_dict = get_weather_dict()

def RMSE(target, outputs):
    return np.sqrt(np.mean(np.square(target - outputs)))


def get_test_img(month, day, hour):
    key = "hour: {}".format(hour)
    date = datetime(2017, month, day)
    date_str = datetime.strftime(date, "%Y%m%d")
    hist_file = h5py.File("./data/hist/{}.h5"
                          .format(date_str), "r")
    test_img = hist_file[key][:]
    hist_file.close()
    return test_img


def generate_test_X(date, hour):
    np.random.seed(233)
    sample = np.random.uniform(size=(10, 10))
    weekday = date.weekday()
    hour_input = (np.array([hour]).astype("float32") - 8) / 14.0
    day_input = (np.array([weekday]).astype("float32") + 1) / 7.0
    sample_input = sample.reshape((1, 100))
    weather_input = np.array([weather_dict[date.strftime("%Y-%m-%d")]])
    return [sample_input, hour_input, day_input, weather_input]


def test_model(model, date, hour, is_show=False):
    # input
    X = generate_test_X(date, hour)

    # predict
    pred = model.predict(X)

    # renormalize
    pred = (pred * 100).astype('int32')

    # get image
    pred_img = pred.reshape((quantize.lat_ctr()-1, quantize.lon_ctr()-1))
    real_img = get_test_img(date.month, date.day, hour)

    if is_show:
        print("Predict Image")
        plt.imshow(pred_img)
        plt.show()
        print("Real Image")
        plt.imshow(real_img)
        plt.show()
        print("Difference:")
        plt.imshow(pred_img - real_img)
        plt.show()

    pred_sub = get_hour_density(pred_img, date, hour)['car_number']
    real_sub = get_hour_density(real_img, date, hour)['car_number']
    error = RMSE(pred_sub, real_sub)

    print("Predict {} {}:00".format(datetime.strftime(date, "%Y%m%d"), hour),
          "RMSE: {}".format(error))

    return error


def deploy_model(model, date, hour, is_show=False):

    # input
    X = generate_test_X(date, hour)

    # predict
    pred = model.predict(X)

    # renormalize
    pred = (pred * 100).astype('int32')

    # get image
    pred_img = pred.reshape((quantize.lat_ctr()-1, quantize.lon_ctr()-1))

    if is_show:
        plt.imshow(pred_img)
        plt.show()

    return pred_img
