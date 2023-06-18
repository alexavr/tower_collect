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
    # # Flatten the structure and convert the groups of the parameters
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


window=30 # rolling and resampling window (min)

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

    plt, ax0, ax1 = tl.plot.web_accustic_stat_prep()
    cur.execute('SELECT equipment_name,Hz,height FROM equipment WHERE tower_name=? AND name="Acoustic anemometer" AND show=1', (tower_name,))
    erows = cur.fetchall()
    for erow in erows:
        equipment_name = erow[0]
        frequency = float(erow[1])
        hgt = float(erow[2])
        print("    Working on tower %s, equipment %s:" % (tower_name, equipment_name))

        buffer_file = Path(config['buffer_path'],'%s_%s_BUFFER.npz' % (tower_name, equipment_name))

        if buffer_file.is_file():

            data = tl.reader.buffer(buffer_file)

            data = tl.data.clean(data)
            data = tl.math.primes(data,int(window*frequency*60), detrend='mean')
            data = tl.math.tke(data)
            plt, ax0, ax1 = tl.plot.web_accustic_stat(data, frequency=frequency, window=window*frequency*60, 
                plt=plt, ax0=ax0, ax1=ax1, label=f"{equipment_name} ({hgt} m)")
            ax0.legend()
            ax1.legend()

    figname_data = "%s/static/%s_STAT_data24hr_spec.png" % (config['wwwpath'], tower_name)
    plt.savefig(figname_data, dpi=150)
    plt.show()

 
print("{}: Running {:.2f} seconds, {:.2f} minutes ".format( datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    (time.time() - start_time),
    (time.time() - start_time)/60.  ) )

