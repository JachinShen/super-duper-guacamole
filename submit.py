import numpy as numpy
import pandas as pd
from datetime import datetime
from quantize import lat_quantize, lon_quantize

def convert_str_range(string):
    string_split = string.split('~')
    return list(map(float, string_split))

def get_submission_range():
    X_sub = pd.read_csv("./A-test.csv")
    X_sub['latitude_range'] = X_sub['latitude_range'].apply(convert_str_range)
    X_sub['longitude_range'] = X_sub['longitude_range'].apply(convert_str_range)
    return X_sub

def get_range_density(hist, range_xy):
    center_lat = lat_quantize((range_xy[0][0] + range_xy[0][1]) / 2)
    center_lon = lon_quantize((range_xy[1][0] + range_xy[1][1]) / 2)
    return (int)(hist[center_lat][center_lon])

def get_hour_density(hist, date, hour):
    X_sub = get_submission_range()
    X_sub['day'] = datetime.strftime(date, "%Y%m%d")
    X_sub['hour'] = hour
    X_sub['car_number'] = X_sub.apply(
        lambda row: (get_range_density(hist,
            [row['latitude_range'], row['longitude_range']])), axis=1)
    return X_sub

def get_all_density(hists):
    frames = []
    #for hist, hour in zip(hists, range(9, 13)):
        #X_sub = get_hour_density(hist, hour)
        #frames.append(X_sub)
    return pd.concat(frames)

def submit_csv(hists):
    submission = get_all_density(hists).drop(
        ['latitude_range', 'longitude_range'], axis=1)
    submission.to_csv("./submission.csv", index=False)