import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import quantize
from submit import get_all_density
from datetime import datetime, timedelta

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

def test_model(model, month, day, hour, is_show=False):
    np.random.seed(233)

    # input
    sample = np.random.uniform(size=(10, 10))
    hour_input = np.array([hour])
    sample_input = sample.reshape((1, 100))

    # predict
    pred = model.predict([sample_input, hour_input])

    # renormalize
    pred = (pred * 100).astype('int32')

    # get image
    pred_img = pred.reshape((quantize.lat_ctr()-1, quantize.lon_ctr()-1))
    real_img = get_test_img(month, day, hour)

    if is_show:
        plt.imshow(pred_img)
        plt.show()

    pred_sub = get_all_density(pred_img)['car_number']
    real_sub = get_all_density(real_img)['car_number']
    print("Predict {}-{} {}:00".format(month, day, hour),
        "RMSE: {}".format(RMSE(pred_sub, real_sub)))