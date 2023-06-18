# import sys
# import os
import sqlite3
from pathlib import Path
import configparser
from netCDF4 import Dataset
import numpy as np
import re
import shutil
from datetime import datetime, timedelta

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


def clean_data(tower_name,equipment_name,fin):

    print(fin.name)

    tmp = re.split('-',fin.name[-13:-3])
    year = tmp[0]
    month = tmp[1]
    day = tmp[2]

    src = Dataset(fin, mode="r")

    time = src.variables['time'][:]

    # dt = time[1:] - time[:-1]
    # if any(dt < 0):
    #     print(" -> ",np.argmin(dt),fin)

    indxs = np.argsort(time)

    path = "{}/{}/{}/{}/{}".format(config['l0_path'],tower_name,equipment_name,year,month)

    # print(indxs)
    Path(path).mkdir(parents=True, exist_ok=True)

    try:
        dst = Dataset(Path(path,fin.name), mode="w")
    except OSError:
        Path(path,fin.name).unlink()
        dst = Dataset(Path(path,fin.name), mode="w")


    dst.setncatts(src.__dict__)
    # copy dimensions
    for name, dimension in src.dimensions.items():
        dst.createDimension(
            name, (len(dimension) if not dimension.isunlimited() else None))

    # copy all file data
    for name, variable in src.variables.items():
        x = dst.createVariable(name, variable.datatype, variable.dimensions)
        # copy variable attributes all at once via dictionary
        # print(" copying %s ..."%(name))
        dst[name].setncatts(src[name].__dict__)
        # copy var data with proper order
        var_data = src[name][:]
        dst[name][:] = var_data[indxs]

    src.close()
    dst.close()


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

        p = Path(config['l0_path'])
        files = list(sorted(p.glob("%s_%s_*.nc" % (tower_name, equipment_name))))
        print("    %s files to process" % (len(files)))
        for fname in files:

            today = datetime.now()
            today_filename = "%s_%s_%d-%02d-%02d.nc"%(tower_name,equipment_name,today.year, today.month, today.day)

            if fname.name != today_filename:

                print(fname.name,today_filename)

                status = clean_data(tower_name,equipment_name,fname)

                path = "{}/temporary_data".format(config['l0_path'])
                Path(path).mkdir(parents=True, exist_ok=True)
                shutil.copy2(fname, path) ### DELETE THIS ASA SORTING PROOVED TO BE OPERATIONAL
                fname.unlink()
                # os.remove(fname)


conn.close()
