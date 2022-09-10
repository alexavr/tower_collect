import netCDF4 as nc
import numpy as np
from datetime import datetime, timedelta
import sqlite3
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator  # for axis integer
import pandas as pd
import configparser
import time
import tower_lib as tl

start_time = time.time()

CONFIG_NAME = "../tower.conf"
DEBUG = False


def read_config(path):

    config = configparser.ConfigParser(
        interpolation=configparser.ExtendedInterpolation()
    )
    config.read(path)

    # settings = dict(config.items("paths"))
    settings = dict(config["paths"])
    # # Flatten the structure and convert the types of the parameters
    # settings = dict(
    #     wwwpath=config.get("paths", "wwwpath"),
    #     datapath=config.get("paths", "datapath"),
    #     dbpath=config.get("paths", "dbpath"),
    #     dbfile=config.getint("paths", "dbfile"),
    #     npzpath=config.getfloat("paths", "npzpath"),
    #     L1path=config.getint("paths", "l1path"),
    #     L2path=config.get("paths", "l2path"),
    #     rrdpath=config.get("paths", "rrdpath"),
    # )

    return settings


def smooth(x, window_len=11, window='hanning'):
    """
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")
    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")
    if window_len < 3:
        return x

    s = np.r_[2*x[0]-x[window_len-1::-1], x, 2*x[-1]-x[-1:-window_len:-1]]

    if window == 'flat':  # moving average
        w = np.ones(window_len)
    elif window == 'hanning':
        w = np.hanning(window_len)
    elif window == 'hamming':
        w = np.hamming(window_len)
    elif window == 'bartlett':
        w = np.bartlett(window_len)
    elif window == 'blackman':
        w = np.blackman(window_len)
    else:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    y = np.convolve(w/w.sum(), s, mode='same')

    return y[window_len:-window_len+1]


def quality_info(datetimes, window_len, frequency):
    """ Computes gaps in data in every window_len (seconds) step
        with window_len step (not running value)
    """
    normal_amount = window_len*frequency
    mask = np.full_like(datetimes, True)

    series = pd.Series(mask, index=datetimes)

    quality = (1 - series.resample(timedelta(seconds=window_len), label="left").sum()/normal_amount)*100.

    return quality.index[1:-1], quality.values[1:-1]


# First read sonfiguration
config = read_config(CONFIG_NAME)

dbfile = "%s/%s" % (config['dbpath'], config['dbfile'])
conn = sqlite3.connect(dbfile)
cur = conn.cursor()
cur.execute('SELECT short_name FROM towers')
trows = cur.fetchall()
for trow in trows:
    tower_name = trow[0]
    print("Working on tower named: %s" % (tower_name))

    cur.execute('SELECT equipment_name,Hz FROM equipment WHERE tower_name=? AND show=1', (tower_name,))
    erows = cur.fetchall()
    for erow in erows:
        equipment_name = erow[0]
        frequency = float(erow[1])
        print("    Working on tower %s, equipment %s:" % (tower_name, equipment_name))

        # if equipment_name != "A12":
        #     continue

        buffer_file = Path(config['buffer_path'],'%s_%s_BUFFER.npz' % (tower_name, equipment_name))

        try:
            data = tl.reader.buffer(buffer_file)
        except:
            data = tl.data.create_empty_pd()

        cur.execute('SELECT name,short_name,long_name,units,missing_value FROM variables WHERE tower_name=? AND equipment_name=?', (tower_name,equipment_name))
        vrows = cur.fetchall()

        for vrow in vrows:

            var_name = vrow[0]
            long_name = vrow[2]
            var_units = vrow[3]
            var_MissingValue = vrow[4]

            attrs = {"name": var_name,
                "long_name": long_name, 
                    "units": var_units, 
               "time_lim_h": int(config['buffer_time_lim_h'])}

            print('        doing %s ...' % (var_name) )

            window_min = 30  # in minutes
            figname_data = "%s/static/%s_%s_%s_data24hr.png" % (config['wwwpath'], tower_name, equipment_name, var_name)
            figname_stat = "%s/static/%s_%s_%s_stat24hr.png" % (config['wwwpath'], tower_name, equipment_name, var_name)

            if frequency >= 1:
                tl.plot.web_accustic(data[var_name], window_min=window_min, 
                    frequency=frequency, attrs=attrs, figname=figname_data)
            else:
                tl.plot.web_meteo(data[var_name], window_min=14, 
                    frequency=frequency, attrs=attrs, figname=figname_data)

 
print("{}: Running {:.2f} seconds, {:.2f} minutes ".format( datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    (time.time() - start_time),
    (time.time() - start_time)/60.  ) )

