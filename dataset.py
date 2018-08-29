import numpy as np
import pandas as pd
import h5py
from datetime import datetime, timedelta

def get_hist_with_time():
    frames = []
    hours = []
    days_from_12 = []
    date = datetime(2017, 1, 2)
    delta_day = timedelta(days=1)
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
