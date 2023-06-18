import numpy as np
# import sys
from pathlib import Path
import sqlite3
from collections import defaultdict
import configparser
from datetime import datetime, timedelta
import time
import tower_lib as tl

start_time = time.time()

CONFIG_NAME = "../tower.conf"
config = tl.reader.config(CONFIG_NAME)
dbfile = f"{config['dbpath']}/{config['dbfile']}"
cur = tl.reader.db_init(dbfile)

towers = tl.reader.bd_get_table_df(cur,f"SELECT short_name FROM towers")

for indext, trow in towers.iterrows():
    tower_name = trow['short_name']
    print(f"Working on tower named: {tower_name}" )

    equipments = tl.reader.bd_get_table_df(cur,f"SELECT equipment FROM equipment WHERE tower_name='{tower_name}' GROUP BY equipment")
    for indexe, erow in equipments.iterrows():
        equipment = erow['equipment']
        print(f"    Working on tower {tower_name}, equipment {equipment}:")

        p = Path(config['npz_path'])
        file_list = list(sorted(p.glob(f"{tower_name}_{equipment}_*.npz")))

        if len(file_list) != 0: # if npz files exists, than merge them

            # upload previous buffer (if exists) and all npz files into RAM

            npz_list = list(sorted(p.glob(f"{tower_name}_{equipment}_*.npz")))
            buf_list = Path(config['buffer_path'],f"{tower_name}_{equipment}_BUFFER.npz")

            if buf_list.is_file():
                file_list = [ buf_list ] + npz_list
            else:
                file_list = npz_list


            merged_data = defaultdict(list)


            # Merge and apply scale coefficient to the new data
            for fname in file_list:
                if Path(fname).stat().st_size != 0:
                    with np.load(fname) as data:
                        for k, v in data.items():
                            if k == 'time' or k == 'level':
                                scale = 1.
                            else:
                                # if np.any(npz_list==fname):
                                if any(x == fname for x in npz_list):
                                    t = (k, tower_name, equipment)
                                    multiplier = tl.reader.bd_get_table_df(cur,f"SELECT multiplier FROM variables \
                                        WHERE name='{k}' AND tower_name='{tower_name}' AND equipment='{equipment}'")
                                    scale = float(multiplier.values)
                                else:
                                    scale = 1.
                            merged_data[k].append(v*scale)

            for k, v in merged_data.items():
                merged_data[k] = np.concatenate(v)


            # # Apply scale coefficient
            # for k,v in merged_data.items():
            #     if k == 'time' or k == 'level':
            #         continue
            #     t = (k, tower_name, equipment)
            #     cur.execute('SELECT multiplier FROM variables WHERE name=? AND tower_name=? AND equipment=?', t)
            #     row = cur.fetchone()
            #     print("          -> {} {} {} scale = {} ".format(tower_name, equipment, k, row[0]))
            #     # merged_data[k] = merged_data[k]*float(row[0])
            #     merged_data[k] = merged_data[k]*row[0]


            # sort by time
            merged_data_sorted = defaultdict(list)
            sorted_indxs = np.argsort(merged_data['time'])
            for k,v in merged_data.items():
                merged_data_sorted[k] = merged_data[k][sorted_indxs]


            # SAVE merged ans sorted data
            p = Path(config['buffer_path'],f"{tower_name}_{equipment}_BUFFER.npz")
            np.savez_compressed(p, **merged_data_sorted)

            # rm temporary npz files
            for fname in npz_list:
                fname.unlink()

        else:
            print(f"          -> {equipment} has no npz files. Skipping...")


print("{}: Running {:.2f} seconds, {:.2f} minutes ".format( datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    (time.time() - start_time),
    (time.time() - start_time)/60.  ) )

