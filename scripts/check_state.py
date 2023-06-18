import tower_lib as tl
from pathlib import Path
from datetime import datetime, timedelta

CONFIG_NAME = "../tower.conf"

config = tl.reader.config(CONFIG_NAME)
dbfile = f"{config['dbpath']}/{config['dbfile']}"

conn, cur = tl.reader.create_connection(dbfile)

towers = tl.reader.bd_get_table_df(cur,"SELECT city,short_name, token FROM towers")

for indext, trow in towers.iterrows():

    levels = tl.reader.bd_get_table_df(cur,f"SELECT height \
        FROM equipment WHERE tower_name='{trow['short_name']}' AND status='online' \
        GROUP BY height \
        ORDER BY height ASC")

    for indext,lrow in levels.iterrows():


        states = tl.reader.bd_get_table_df(cur,f"SELECT tower_name,equipment,type,name,height,state \
            FROM equipment WHERE tower_name='{trow['short_name']}' AND height='{lrow['height']}' AND status='online' ")

        for indext,erow in states.iterrows():
            

            buffer_file = Path(config['buffer_path'],f"{trow['short_name']}_{erow['equipment']}_BUFFER.npz" )
            data, status = tl.reader.read_buffer(buffer_file)

            current_status = tl.data.check4dataflow(data,status)

            if erow['state'] != current_status:

                if current_status == 'good':
                    msg = f"{datetime.utcnow().replace(microsecond=0)}\nTower {trow['short_name']} in {trow['city']} \nEquipment: \'{erow['name']}\' \nat {lrow['height']} m level\is back online again!"
                else:
                    msg = f"{datetime.utcnow().replace(microsecond=0)}\nTower {trow['short_name']} in {trow['city']} \nEquipment: \'{erow['name']}\' \nat {lrow['height']} m level\nhas gone OFFLINE!"

                # UPDATE current_status into towers.db
                tl.reader.bd_update(conn, cur,f"UPDATE equipment SET state = '{current_status}' \
                    WHERE tower_name='{trow['short_name']}' AND height='{lrow['height']}' AND status='online'")

                # Sending notification
                res = tl.notification.send2bot(trow['token'], 269679622, msg)
                
                print(f"    UPDATING {trow['short_name']} {erow['equipment']:<3} with {erow['state']} to {current_status}")

            