from dataset import extract_all_hists
from datetime import datetime

extract_all_hists(datetime(2017, 2, 21),
    datetime(2017, 3, 12), "./data/hist_all/")