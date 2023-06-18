# import sys
import os
import sqlite3
from pathlib import Path
import configparser


CONFIG_NAME = "../tower.conf"
# DEBUG = False

################################################################################


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

# def save_buffer(tower_name, equipment_name, fin):

#     config = read_config(CONFIG_NAME)

#     fout = '{0}/{1}_{2}.npy'.format(config['buffer_path'], tower_name, equipment_name)

#     # Load the data
#     data = np.load(fin)




def convert_file(tower_name, equipment_name, fin):

    import netCDF4 as nc
    import numpy as np
    from datetime import datetime, time

    config = read_config(CONFIG_NAME)

    status = False

    # Load the data
    data = np.load(fin)

    datetimes = np.array([datetime.fromtimestamp(ts) for ts in data['time']])
    dates = [datetime.combine(dt, time.min) for dt in datetimes]
    unique_dates = np.unique(dates + [datetime.max])

    for i in range(len(unique_dates) - 1):
        mask = (datetimes >= unique_dates[i]) & (datetimes < unique_dates[i + 1])
        # subset = datetimes[mask]

        # seconds = (unique_dates[i] - datetime(1970,1,1)).total_seconds

        fout = '{0}/{1}_{2}_{3}'.format(config['l0_path'], tower_name, equipment_name, unique_dates[i].strftime('%Y-%m-%d.nc'))
        nmask = mask.sum()

        ####### new part start
        if os.path.isfile(fout):
            # print("file ... %s"%(fout))
            params = dict(mode="a")
            init = False
        else:
            # print("new file")
            params = dict(mode="w", clobber=False, format='NETCDF4_CLASSIC')
            init = True

        with nc.Dataset(fout, **params) as ncout:
            if init:

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
                times.calendar = "gregorian"
                for iv in data.files:

                    if iv == 'time' or iv == 'level':
                        continue

                    t = (iv, tower_name, equipment_name)
                    cur.execute('SELECT short_name,long_name,units,description,missing_value,coordinates,multiplier FROM variables WHERE name=? AND tower_name=? AND equipment_name=?', t)
                    row = cur.fetchone()

                    var = ncout.createVariable(iv, "f4", ("time",), fill_value=row[4])
                    var.short_name = row[0]
                    var.long_name = row[1]
                    var.description = row[3]
                    var.units = row[2]
                    var.missing_value = row[4]
                    var.coordinates = row[5]
                    var.multiplier = row[6]

            ntime = len(ncout.dimensions['time'])

            for iv in data.files:

                if iv == 'level':
                    continue

                if iv == 'time':
                    multiplier = 1
                else:
                    t = (iv, tower_name, equipment_name)
                    cur.execute('SELECT multiplier FROM variables WHERE name=? AND tower_name=? AND equipment_name=?', t)
                    row = cur.fetchone()
                    multiplier = row[0]

                ncout[iv][ntime:(ntime+nmask)] = data[iv][mask]*multiplier

    data.close()

    status = True

    return status

################################################################################


config = read_config(CONFIG_NAME)

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
        print("    Working on equipment: %s" % (equipment_name))

        p = Path(config['npz_path'])
        files = list(sorted(p.glob("%s_%s_*.npz" % (tower_name, equipment_name))))
        print("    %s files to process" % (len(files)))
        for fname in files:
            filesize = os.path.getsize(fname)
            print("        %s ; size = %d" % (fname,filesize))
            if filesize == 0:
                print("        The file is empty, skipping!")
            else:
                status = convert_file(tower_name, equipment_name, fname)
                # save_buffer(tower_name, equipment_name, fname)

            # # if status: shutil.move(fname, "%s/trash"%npzpath )
            # head_tail = os.path.split(fname)
            # if status: os.rename(fname, "%s/_%s"%(head_tail[0],head_tail[1]) )
            os.remove(fname)


conn.close()
