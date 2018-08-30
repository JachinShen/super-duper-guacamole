from dataset import extract_all_hists
from datetime import datetime

extract_all_hists(datetime(2017, 2, 7),
    datetime(2017, 2, 20), "./data/hist_all/")