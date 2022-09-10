# Converts BUFFER into nc file, but leaves last buffer_time_lim_h of data
# for the web to plot pictures.
#
# IMPORTANT!
# Data converts into NetCDF file only if it's older then buffer_time_lim_h!
#
# Alogo:
# 1. Read buffer
# 2. Sort the data
# 3. Loop through days ):
    # 3.1 Get the start and end indexes
    # 3.2 Check if nc file is already exists
        # if NO: create a new one
    # 3.3 Fill the nc file (append the data)
    # 3.4 remove saved data from the BUFFER

import netCDF4 as nc
import numpy as np
from datetime import datetime, timedelta
# import datetime
import sqlite3
from pathlib import Path
import pandas as pd
import configparser
from collections import defaultdict
import pandas as pd
import time

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

def create_netcdf(tower_name, equipment_name, fout):

    Path(fout.parents[0]).mkdir(parents=True, exist_ok=True)

    with nc.Dataset(fout, mode="w", clobber=False, format='NETCDF4_CLASSIC') as ncout:
        # get some tower information for global variables

        cur.execute('SELECT id,long_name,lat,lon FROM towers WHERE short_name=?', (tower_name,))
        row = cur.fetchone()

        ncout.tower = tower_name
        ncout.tower_description = row[1]
        ncout.tower_longitude = row[2]
        ncout.tower_latitude = row[3]

        # get some equipment information for global variables

        cur.execute('SELECT type,name,height,model,Hz,install_date FROM equipment WHERE equipment_name=?', (equipment_name,))
        row = cur.fetchone()

        ncout.equipment_name = equipment_name
        ncout.equipment_type = row[0]
        ncout.equipment_description = row[1]
        ncout.equipment_height = row[2]
        ncout.equipment_model = row[3]
        ncout.equipment_frequency = row[4]
        ncout.equipment_InstallDate = row[5]

        ncout.history = 'Created {0}'.format(datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC'))

        time = ncout.createDimension("time", None)
        times = ncout.createVariable("time", "f8", ("time",))
        times.units = "seconds since 1970-01-01 00:00:00.0"
        times.calendar = "Gregorian"


        cur.execute('SELECT name,short_name,long_name,units,description,missing_value,coordinates  FROM variables WHERE tower_name=? AND equipment_name=?', (tower_name,equipment_name))
        vrows = cur.fetchall()

        for vrow in vrows:

            var_name = vrow[0]
            var_short_name = vrow[1]
            var_long_name = vrow[2]
            var_units = vrow[3]
            var_description = vrow[4]
            var_MissingValue = vrow[5]
            var_coordinatese = vrow[6]

            # print("        %s %s %s %s %s %s %s " % (var_name, var_short_name, var_long_name, var_units, var_description, var_MissingValue, var_coordinatese))

            var = ncout.createVariable(var_name, "f4", ("time",), fill_value=var_MissingValue)
            var.short_name = var_short_name
            var.long_name = var_long_name
            var.description = var_description
            var.units = var_units
            var.missing_value = var_MissingValue
            var.coordinates = var_coordinatese
            # var.multiplier = row[6]



# First read configuration
config = read_config(CONFIG_NAME)

# Since all is written in the BD file -- go there
dbfile = "%s/%s" % (config['dbpath'], config['dbfile'])
conn = sqlite3.connect(dbfile)
cur = conn.cursor()
cur.execute('SELECT short_name FROM towers')
trows = cur.fetchall()
for trow in trows:
    tower_name = trow[0]
    print("Working on tower named: %s" % (tower_name))

    cur.execute('SELECT equipment_name FROM equipment WHERE tower_name=?', (tower_name,))
    erows = cur.fetchall()
    for erow in erows:

        equipment_name = erow[0]

        print("    Working on tower %s, equipment %s:" % (tower_name, equipment_name))

        # Read the BUFFER
        buffer_file = Path(config['buffer_path'],'%s_%s_BUFFER.npz' % (tower_name, equipment_name))
        if buffer_file.is_file():
            data = np.load(buffer_file)
        else:
            print("    -> No BUFFER file found. Skipping... ")
            continue

        # Sort the data
        data_sorted = defaultdict(list)
        sorted_indxs = np.argsort(data['time'])
        for k,v in data.items():
            data_sorted[k] = data[k][sorted_indxs]

        # Trim the data. Cut the last buffer_time_lim_h from BUFFER in order to
        # avoid doubling or extra heavy checks during writing into NetCDF file/
        # First find the index to trim up to
        buffer_time_lim_sec = float(config['buffer_time_lim_h'])*60*60
        now_time_sec = datetime.utcnow().timestamp()
        trim_datetime = now_time_sec - buffer_time_lim_sec
        trim_dtime = np.abs(data_sorted['time'] - trim_datetime)
        trim_dtime_ind = np.argmin(trim_dtime)
        # Then trim the array
        data_sorted_trim = defaultdict(list)
        for k,v in data_sorted.items():
            data_sorted_trim[k] = data_sorted[k][:trim_dtime_ind]

        # If the (now - buffer_time_lim_h) hits the age of the BUFFER array
        # It means the data stopped coming buffer_time_lim_h hours ago and we
        # need to save all in the BUFFER and delete the BUFFER file
        EndOfBuffer = False
        if trim_dtime_ind == ( len(data_sorted['time']) - 1 ):
            EndOfBuffer = True

        start_date = datetime.fromtimestamp(data_sorted_trim['time'][0]).date()
        # end_date = datetime.fromtimestamp(data_sorted_trim['time'][-1]).date()
        end_date = datetime.now().date() # TODAY
        delta = timedelta(days=1)

        datetime_dt = np.array([datetime.utcfromtimestamp(ts) for ts in data_sorted_trim['time']])
        datetime_pd = pd.Series(data=datetime_dt,index=datetime_dt)

        print(f"    -> Doing timeseries from {start_date} to {end_date}...")


        # находим индексы для каждой ДАТЫ, вынимаем данные и пишем их в nc файл
        # Works for long data (more than 2 days)
        while start_date < end_date:


            tmp = ( datetime_pd.dt.date == start_date )
            indexes = np.where(tmp)[0]

            if len(indexes)==0:
                print(f"    --> {start_date} has no data. Skipping...")
                start_date += delta
                continue

            print(f"    --> {start_date} has {len(indexes)} measures. Saving into NetCDF...")

            # print("    -> Doing {} istart = {} iend = {}".format(start_date, indexes[0],indexes[-1]))

            fout = Path("%s/%s/%s/%04d/%02d/%s_%s_%s.nc"%(config['l0_path'],
                tower_name,equipment_name,start_date.year,start_date.month,
                tower_name, equipment_name,start_date))


            if not fout.is_file():
                print("    --> Creating     %s"%(fout))
                create_netcdf(tower_name, equipment_name, fout) # 
            else:
                print("    --> Appending to %s"%(fout))

            with nc.Dataset(fout, mode="a") as ncout:
                it_start = len(ncout.dimensions['time'])
                it_size = len(indexes)

                for k,v in data_sorted_trim.items():
                    print("    ---> Doing %s " %(k))
                    # print(it_start,it_size, len(data_sorted_trim[k][indexes[0]:indexes[-1]]))

                    ncout[k][it_start:(it_start+it_size-1)] = data_sorted_trim[k][indexes[0]:indexes[-1]]

            start_date += delta

        data.close()

        # If BUFFER is empty - delete the BUFFER file
        # Else rewrite new BUFFER with data left from trim (clean the buffer)
        print("    -> Re-buffering...")
        if EndOfBuffer:
            buffer_file.unlink()
        else:
            data_save = defaultdict(list)
            for k,v in data_sorted.items():
                data_save[k] = data_sorted[k][trim_dtime_ind+1:]
            buffer_file.unlink()
            np.savez_compressed(buffer_file, **data_save)



print("{}: Running {:.2f} seconds, {:.2f} minutes ".format( datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    (time.time() - start_time),
    (time.time() - start_time)/60.  ) )

