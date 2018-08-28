def lat_quantize(lat):
    lat_min = 31.1
    return (int)((lat - lat_min) / 0.005)
def lon_quantize(lon):
    lon_min = 121.3
    return (int)((lon - lon_min) / 0.005)