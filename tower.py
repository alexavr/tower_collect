#!/home/tower/anaconda3/envs python3.6
from flask import Flask, render_template, request
from datetime import datetime, timedelta
import rrdtool
from pathlib import Path
# import sqlite3
import glob
import os
import netCDF4 as nc
# import configparser
import numpy as np

import sys
sys.path.append(Path(os.path.join(os.getcwd(),"scripts/")).__str__())
import tower_lib as tl


app = Flask(__name__)  # initialize

CONFIG_NAME = "./tower.conf"


# for main Flask run (no debug)
# if __name__ == "__main__":
    # app.run(debug=True)
    # app.run(host='0.0.0.0')
    # app.run()

def dict_factory(cursor, row):
    d = {}
    for idx, col in enumerate(cursor.description):
        d[col[0]] = row[idx]
    return d

def newest(path):
    files = glob.glob(path)

    if len(files) == 0:
        return None
    else:
        return max(files, key=os.path.getctime)


@app.after_request
def add_header(r):
    """
    Avoid caching
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r


config = tl.reader.config(CONFIG_NAME)


@app.route('/')
@app.route('/home')
def home():

    dbfile = "%s/%s" % (config['dbpath'], config['dbfile'])
    cur = tl.reader.db_init(dbfile)
    towers = tl.reader.bd_get_table_df(cur,"SELECT city,short_name FROM towers")

    return render_template('towers.html', towers=towers.values)


@app.route('/rtdata', methods=['GET'])
def rtdata():
    tower_name = request.args.get('tower', default=1)
    tower_city = request.args.get('city', default=1)
    active = float(request.args.get('active', default=-1))
    hbactive = request.args.get('hbactive', default=1)
    # var = request.args.get('var', default=1)
    hbvar = request.args.get('hbvar', default=1)
    section = request.args.get('section', default=1)
    level = request.args.get('level', default=1)

# setting some defaults
    if hbactive == 1:
        hbactive = 'hbboxtemp'

    if hbvar == 1:
        hbvar = 'hbboxtemp'

    mtimed = datetime.utcnow()
    mtime = int(mtimed.timestamp())


# reading sqlite info
    dbfile = f"{config['dbpath']}/{config['dbfile']}"
    cur = tl.reader.db_init(dbfile)

    details = tl.reader.bd_get_table(cur,f"SELECT description FROM towers WHERE short_name='{tower_name}' AND status='online'")

    variables = (0,)
    levels = []
    lastupdate, dlastupdate, alert = 0, 0, 0

    types = tl.reader.bd_get_table_df(cur,f"SELECT type FROM equipment \
        WHERE tower_name='{tower_name}' AND height='{active}' AND status='online'  \
        GROUP BY type \
        ORDER BY type DESC")
    levels = tl.reader.bd_get_table_df(cur,f"SELECT height FROM equipment \
        WHERE tower_name='{tower_name}' AND status='online' \
        GROUP BY height \
        ORDER BY height ASC")
    equipment = [] # FOR 'HB' SECTION


# ALERTS (HB and RTDATA)
    hb_lastupdate, hb_dlastupdate, hb_alert = tl.data.get_lastupdate_rrd(tower_name)
    # if active == -1:
    #     lastupdate, dlastupdate, alert = 0, 0, 0
    # else:
    #     lastupdate, dlastupdate, alert = tl.data.get_lastupdate_data_l0(tower_name, active)

    hb_variables = [
        {'name': 'hbtemp', 'long_name': 'BOX & CPU temperature'},
        {'name': 'hbmem',  'long_name': 'Memory consumption'},
        {'name': 'hbnet',  'long_name': 'Network usage'}
    ]

    equipment = []
    for index,lev in levels.iterrows():
        tmp = tl.reader.bd_get_table_df(cur,f"SELECT name FROM equipment WHERE tower_name='{tower_name}' AND height='{lev['height']}' AND status='online' ORDER BY height ASC")
        tmp = [tpl[0] for tpl in tmp.values]
        tmp = ', '.join(tmp)
        equipment.append(tmp)

    variables = []
    for indexl,lev in levels.iterrows():
        tmp = tl.reader.bd_get_table_df(cur,f"SELECT short_name FROM variables \
            WHERE tower_name='{tower_name}' AND height='{lev['height']}' AND status='online'  AND plot=1 \
            ORDER BY equipment ASC")
        tmp = [tpl[0] for tpl in tmp.values]
        tmp = ', '.join(tmp)
        variables.append(tmp)

    # # rm unnecessary variables (currently the device temperature)
    # variables = [d for d in variables
    #                  if d["name"] not in {"e1", "e2"}
    #                  ]

    levels = levels.values
    types = types.values

    level = level

    var_longname = ''


    return render_template('rtdata.html', 
        tower=tower_name, 
        city=tower_city,
        details=details,
        section=section,
        types=types,
        levels=levels,
        level=level,
        equipment=equipment,
        active=active,
        hbactive=hbactive,
        # var=var,
        # var_longname=var_longname,
        variables=variables,
        hb_variables=hb_variables,
        hbvar=hbvar,
        # lastupdate=lastupdate,
        # dlastupdate=dlastupdate,
        # alert=alert,
        hb_lastupdate=hb_lastupdate,
        hb_dlastupdate=hb_dlastupdate,
        hb_alert=hb_alert,
        mtime=mtime,
        title="Real-time data")



@app.route('/description')
def description():
    return render_template('description.html')


@app.route('/references')
def references():
    return render_template('references.html')


@app.route('/download')
def download():
    return render_template('download.html')

@app.route('/layout')
def layout():
    dbfile = f"{config['dbpath']}/{config['dbfile']}"
    cur = tl.reader.db_init(dbfile)
    news = tl.reader.bd_get_table_df(cur,f"SELECT date,news_en FROM news")


    return render_template('layout.html', news=news)


################################################################################
# read HeartBeat data from mast
@app.route('/hb_put', methods=['POST'])
def get_data():
    name = request.form['name']
    dt = request.form['datetime']
    boxtemp_str = request.form['boxtemp']
    cputemp_str = request.form['cputemp']
    hdd = request.form['hdd']
    ram = request.form['ram']
    netin = request.form['in']
    netout = request.form['out']

    cputemp = float(cputemp_str)/1000.
    boxtemp = float(boxtemp_str)-4.
#    date = datetime.utcfromtimestamp(int(dt)).strftime('%Y-%m-%d %H:%M:%S UTC')

    rrdtool.update('data/hb/%s.rrd' % name, '-t',
        'boxtemp:cputemp:hdd:ram:in:out',
        '%s:%.3f:%.3f:%s:%s:%s:%s' % (dt, boxtemp, cputemp, hdd, ram, netin, netout))

    return name
