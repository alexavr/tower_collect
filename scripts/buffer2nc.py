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
import tower_lib as tl

start_time = time.time()

CONFIG_NAME = "../tower.conf"
DEBUG = False

def create_netcdf(tower_name, equipment, fout):

    Path(fout.parents[0]).mkdir(parents=True, exist_ok=True)

    with nc.Dataset(fout, mode="w", clobber=False, format='NETCDF4_CLASSIC') as ncout:
        # get some tower information for global variables

        dbfile = f"{config['dbpath']}/{config['dbfile']}"
        cur = tl.reader.db_init(dbfile)
        towers = tl.reader.bd_get_table_df(cur,f"SELECT id,long_name,lat,lon FROM towers WHERE short_name='{tower_name}'")
        ncout.tower = tower_name
        ncout.tower_description = towers['long_name'].values[0]
        ncout.tower_longitude = towers['lon'].values[0]
        ncout.tower_latitude = towers['lat'].values[0]

        equipments = tl.reader.bd_get_table_df(cur,f"SELECT type,name,height,model,Hz,install_date FROM equipment WHERE equipment='{equipment}'")
        ncout.history = f"Created {format(datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC'))}"
        ncout.name = equipments['name'].values[0]
        ncout.type = equipments['type'].values[0]
        ncout.model = equipments['model'].values[0]
        ncout.frequency = equipments['Hz'].values[0]
        ncout.height = equipments['height'].values[0]
        ncout.install_date = equipments['install_date'].values[0]

        time = ncout.createDimension("time", None)
        times = ncout.createVariable("time", "f8", ("time",))
        times.units = "seconds since 1970-01-01 00:00:00.0"
        times.calendar = "Gregorian"

        # create variables
        variables = tl.reader.bd_get_table_df(cur, f"SELECT name,short_name,long_name,units,description,missing_value,coordinates,height \
            FROM variables \
            WHERE tower_name='{tower_name}' AND equipment='{equipment}'")

        for index_var,vrow in variables.iterrows():

            var = ncout.createVariable(vrow[0], "f4", ("time",), fill_value=vrow[5])
            var.short_name    = vrow['short_name']
            var.long_name     = vrow['long_name']
            var.description   = vrow['description']
            var.units         = vrow['units']
            var.coordinates   = vrow['coordinates']
            var.missing_value = vrow['missing_value']



CONFIG_NAME = "../tower.conf"
config = tl.reader.config(CONFIG_NAME)
dbfile = f"{config['dbpath']}/{config['dbfile']}" 
cur = tl.reader.db_init(dbfile)

towers = tl.reader.bd_get_table_df(cur,f"SELECT short_name FROM towers")

for index_tower,trow in towers.iterrows():

    tower_name = trow['short_name']
    print(f"Working on tower named: {tower_name}")


    equipment = tl.reader.bd_get_table_df(cur,f"SELECT equipment FROM equipment WHERE tower_name='{tower_name}' ")

    for index_eq, erow in equipment.iterrows():

        equipment = erow['equipment']

        print(f"    Working on tower {tower_name}, equipment {equipment}:")

        # Read the BUFFER
        buffer_file = Path(config['buffer_path'],f'{tower_name}_{equipment}_BUFFER.npz')
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
                tower_name,equipment,start_date.year,start_date.month,
                tower_name, equipment,start_date))


            if not fout.is_file():
                print(f"    --> Creating     {fout}")
                create_netcdf(tower_name, equipment, fout) #
            else:
                print(f"    --> Appending to {fout}")

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

