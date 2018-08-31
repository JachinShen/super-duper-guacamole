def lat_quantize(lat):
    lat_min = 31.1
    return (int)((lat - lat_min) / 0.005)


def lon_quantize(lon):
    lon_min = 121.3
    return (int)((lon - lon_min) / 0.005)


def lat_ctr():
    lat_min, lat_max = 31.1, 31.4
    return (int)((lat_max - lat_min) / 0.005)


def lon_ctr():
    lon_min, lon_max = 121.3, 121.8
    return (int)((lon_max - lon_min) / 0.005)
