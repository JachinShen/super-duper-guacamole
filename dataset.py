# author: JachinShen(jachinshen@foxmail.com)
import numpy as np
import pandas as pd
import h5py
from datetime import datetime, timedelta
from quantize import lat_quantize, lon_quantize
import requests  #导入requests
from bs4 import BeautifulSoup  #导入bs4中的BeautifulSoup
import os
import re
import csv
import time
import json
#coding=utf-8

lat_min, lat_max = 31.1, 31.4
lon_min, lon_max = 121.3, 121.8
lat_ctr = (int)((lat_max - lat_min) / 0.005)
lon_ctr = (int)((lon_max - lon_min) / 0.005)

def extract_hist(data):
    hist, _, _ = np.histogram2d(data['lat'], data['lon'],
        bins = [range(lat_ctr), range(lon_ctr)])
    return hist

def get_hist_hour(data, hour_begin, hour_end):
    data_hour = data[(hour_begin <= data['date_time']) 
        & (data['date_time'] < hour_end)]
    data_hour['lat'] = data_hour['lat'].apply(lat_quantize)
    data_hour['lon'] = data_hour['lon'].apply(lon_quantize)

    # the grid only counts one car one times
    # remove the samples with the same car_id and lat and lon
    data_hour = data_hour.drop_duplicates(['car_id', 'lat', 'lon'])
    hist = extract_hist(data_hour)
    return hist

def get_raw_data(month, day):
    frames = []
    features_need = ['car_id', 'date_time', 'lat', 'lon']

    # give dtype to save the time for pd to guess
    rcar_dtype = {'car_id': str,
                  'date_time': str,
                  'lat': float,
                  'lon': float,
                  'power_mode': str,
                  'mileage': float,
                  'speed': float,
                  'fuel_consumption': float}
    ecar_dtype = {'car_id': str,
                  'date_time': str,
                  'lat': float,
                  'lon': float,
                  'work_mode': str,
                  'mileage': float,
                  'speed': float,
                  'avg_fuel_consumption': float,
                  'system_mode': str}
    date = datetime.strftime(datetime(2017, month, day), "%Y%m%d")
    for part in range(3):
        filename = ("./data/rcar/BOT_data_rcar_{1}_{1}_part{0}.csv"
                    .format(part, date))
        data_part = pd.read_csv(filename, dtype=rcar_dtype)[features_need]
        data_part = data_part[(data_part['lat'] > 0.1) 
            & (data_part['lon'] > 0.1)]
        frames.append(data_part)
        
        filename = ("./data/ecar/BOT_data_ecar_{1}_{1}_part{0}.csv"
                    .format(part, date))
        data_part = pd.read_csv(filename, dtype=ecar_dtype)[features_need]
        data_part = data_part[(data_part['lat'] > 0.1) 
            & (data_part['lon'] > 0.1)]
        frames.append(data_part)
        
    data = pd.concat(frames)
    data["date_time"] = data["date_time"].apply(
        lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))
    return data

def extract_all_hists(begin_date, end_date, folder_path="./data/hist/"):
    date = begin_date
    delta_day = timedelta(days=1)
    while date <= end_date:
        date_str = datetime.strftime(date, "%Y%m%d")
        print("Processing {}".format(date_str))
        with h5py.File("{}{}.h5"
            .format(folder_path, date_str), "w") as f:
            data = get_raw_data(date.month, date.day)
            for hour in range(9, 23):
                hist = get_hist_hour(data,
                    datetime(2017, date.month, date.day, hour),
                    datetime(2017, date.month, date.day, hour+1))
                f.create_dataset('hour: {}'.format(hour), data=hist)
        date += delta_day

def get_hist_with_time(begin_date, end_date):
    frames = []
    hours = []
    weekday = []
    workday = []
    date = begin_date
    delta_day = timedelta(days=1)
    while date <= end_date:
        date_str = datetime.strftime(date, "%Y%m%d")
        hist_file = h5py.File("./data/hist/{}.h5"
            .format(date_str), "r")
        for hour in range(9, 23):
            key = "hour: {}".format(hour)
            frames.append(hist_file[key][:])
            hours.append(hour)
            weekday.append(date.weekday())
            workday.append((is_workday(date)))
        hist_file.close()
        date += delta_day

    frames = np.array(frames)
    weekday = np.array(weekday)
    hours = np.array(hours)
    workday = np.array(workday)
    return frames, weekday, hours, workday


class get_history_weather():
    def __init__(self):
        self.request_url = 'http://lishi.tianqi.com/shanghai/index.html'
    def get_url(self):
        # 获取所有月份的url
        html = requests.get(self.request_url).text
        Soup = BeautifulSoup(html, 'lxml') # 解析文档
        all_li = Soup.find('div', class_='tqtongji1').find_all('li')
        url_list = []
        for li in all_li:
            url_list.append([li.get_text(), li.find('a')['href']])       
        return url_list
    

    def get_month_weather(self, year_number, month_number):
        # month_url = 'http://lishi.tianqi.com/beijing/201712.html'
        url_list = self.get_url()
        for i in range(len(url_list)-1, -1, -1):
            year_split = int(url_list[i][0].encode('utf-8')[:4])
            month_split = int(url_list[i][0].encode('utf-8')[7:9])
            if year_split == year_number and month_split == month_number:
                month_url = url_list[i][1]
        html = requests.get(month_url).text
        Soup = BeautifulSoup(html, 'lxml') # 解析文档
        all_ul = Soup.find('div', class_='tqtongji2').find_all('ul')
        month_weather = []
        for i in range(1, len(all_ul)):
            ul = all_ul[i]
            li_list = []
            for li in ul.find_all('li'):
                li_list.append(li.get_text().encode('utf-8'))
            month_weather.append(li_list)
        return month_weather       

example = get_history_weather()
weather_result = example.get_month_weather(2017,1)+example.get_month_weather(2017,2)+example.get_month_weather(2017,3)
name=['date','high_temprature','low_temprature','weather_condition','wind_direction','wind_force']
weather=pd.DataFrame(columns=name,data=weather_result)
weather.to_csv(r'C:/Users/sindy123/Documents')

if __name__ == "__main__":
    extract_all_hists(datetime(2017, 1, 2),
        datetime(2017, 1, 2), "./data/test/")
is