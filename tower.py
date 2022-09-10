from flask import Flask, render_template, request
from datetime import datetime, timedelta
import rrdtool
from pathlib import Path
import sqlite3
import glob
import os
import netCDF4 as nc
import configparser
import numpy as np


app = Flask(__name__)  # initialize

CONFIG_NAME = "./tower.conf"


# for main Flask run (no debug)
# if __name__ == "__main__":
    # app.run(debug=True)
    # app.run(host='0.0.0.0')
    # app.run()


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


config = read_config(CONFIG_NAME)


def dict_factory(cursor, row):
    d = {}
    for idx, col in enumerate(cursor.description):
        d[col[0]] = row[idx]
    return d


def get_lastupdate_rrd(stname):
    lastupdate = rrdtool.lastupdate(config['rrdpath']+"/"+stname+".rrd")
    lastupdate_time = lastupdate["date"]
    lastupdate_dtime = datetime.utcnow() - lastupdate_time
    lastupdate_dtime -= timedelta(microseconds=lastupdate_dtime.microseconds)
    lastupdate_time_str = lastupdate_time.strftime("%Y-%m-%d %H:%M:%S UTC")
    minutes = lastupdate_dtime.total_seconds() / (60)

    alert = False
    if minutes > 3:
        alert = True

    return [lastupdate_time_str, lastupdate_dtime, alert]


def newest(path):
    files = glob.glob(path)

    if len(files) == 0:
        return None
    else:
        return max(files, key=os.path.getctime)


def get_lastupdate_data_l0(stname, equipment_name):


    buffer_file = Path(config['buffer_path'],'%s_%s_BUFFER.npz' % (stname, equipment_name))
    if buffer_file.is_file():

        data = np.load(buffer_file)
        time_from_data = data['time'][-1]

    else:

        files = sorted(Path("%s/%s/%s/"%(config['l0_path'],stname,equipment_name)).rglob('%s_%s*.nc'%(stname,equipment_name)))

        if len(files) != 0:
            fin = files[-1] # newest(config['l0_path']+"/"+stname+"_"+equipment_name+"*.nc")
            f = nc.Dataset(fin)
            time_from_data = f.variables['time'][-1]
            f.close()
        else: # No files found at all (SHOULD NEVER HAPPENED!)
            time_from_data = 0


    lastupdate_time = datetime.fromtimestamp(time_from_data)

    lastupdate_dtime = datetime.utcnow() - lastupdate_time
    lastupdate_dtime -= timedelta(microseconds=lastupdate_dtime.microseconds)
    lastupdate_time_str = lastupdate_time.strftime("%Y-%m-%d %H:%M:%S UTC")
    hours = lastupdate_dtime.total_seconds() / (60 * 60)

    alert = False
    if hours > 1:
        alert = True


    return [lastupdate_time_str, lastupdate_dtime, alert]


# def get_varname(var):
#     if var == "temp":
#         varname = "Temperature"
#     elif var == "mem":
#         varname = "Memory consumption"
#     elif var == "net":
#         varname = "Traffic consumption"
#     else:
#         varname = "unknown"
#     return varname


# def render_tower_hb(stname):

#     datapath="./data/hb/"

#     for sched in ['1d' , '1w', '1m', '1y']:

#         rrdtool.graph("./static/%s_temp_%s.png"%(stname,sched),
#             "--start", "-%s"%(sched),
#             # "--vertical-label=temperature",
#             "--width","800",
#             "--height","130",
#             "DEF:temp=%s%s.rrd:temp:AVERAGE"%(datapath,stname),
#             'AREA:temp#FA8072:CPU temperature',
#             'LINE:temp#B22222:')

#         rrdtool.graph("./static/%s_mem_%s.png"%(stname,sched),
#             "--start", "-%s"%(sched),
#             # "--vertical-label=%",
#             "--width","800",
#             "--height","130",
#             "DEF:hdd=%s%s.rrd:hdd:AVERAGE"%(datapath,stname),
#             "DEF:ram=%s%s.rrd:ram:AVERAGE"%(datapath,stname),
#             'AREA:hdd#A9A9A9:HDD',
#             'LINE:hdd#696969:',
#             'LINE:ram#FF0000:RAM')

#         rrdtool.graph("./static/%s_net_%s.png"%(stname,sched),
#             "--start", "-%s"%(sched),
#             "--slope-mode",
#             # "--border", "0",
#             # "--color", "BACK#ffffff",
#             # "--color", "CANVAS#ffffff",
#             # "--vertical-label=bytes/sec",
#             "--width","800",
#             "--height","130",
#             "DEF:in=%s%s.rrd:in:AVERAGE"%(datapath,stname),
#             "DEF:out=%s%s.rrd:out:AVERAGE"%(datapath,stname),
#             "CDEF:out_neg=out,-1,*",
#             'AREA:in#32CD32:Incoming',
#             'LINE:in#336600:',
#             'AREA:out_neg#4169E1:Outgoing',
#             'LINE:out_neg#0033CC:')


@app.route('/')
@app.route('/home')
def home():

    dbfile = "%s/%s" % (config['dbpath'], config['dbfile'])

    con = sqlite3.connect(dbfile)
    con.row_factory = dict_factory
    cur = con.cursor()
    cur.execute('SELECT city,short_name FROM towers')
    towers = cur.fetchall()
    con.close()

    return render_template('towers.html', towers=towers)


# @app.route('/tower', methods=['GET'])
# def tower():
#     tower_name = request.args.get('tower', default=1)
#     tower_city = request.args.get('city', default=1)

#     dbfile = "%s/%s" % (config['dbpath'], config['dbfile'])

#     con = sqlite3.connect(dbfile)
#     con.row_factory = dict_factory
#     cur = con.cursor()
#     cur.execute('SELECT description FROM towers WHERE short_name=?', (tower_name, ))
#     qresult = cur.fetchone()
#     con.close()

#     return render_template('tower.html',
#         tower=tower_name,
#         city=tower_city,
#         details=qresult)

# @app.route('/hb', methods=['GET'])
# def hb():
#     tower_name = request.args.get('tower',default=1)
#     tower_city = request.args.get('city',default=1)
#     var = request.args.get('var',default=1)
#     varname = get_varname(var)
#     lastupdate, dlastupdate, alert = get_lastupdate_rrd(tower_name)
#     # render_tower_hb(tower_name)
#     return render_template('hb.html', tower=tower_name, city=tower_city,
#         var=var,
#         varname=varname,
#         lastupdate=lastupdate,
#         dlastupdate=dlastupdate,
#         alert=alert,
#         title="Heartbeat viewer")


@app.route('/rtdata', methods=['GET'])
def rtdata():
    tower_name = request.args.get('tower', default=1)
    tower_city = request.args.get('city', default=1)
    active = request.args.get('active', default=1)
    hbactive = request.args.get('hbactive', default=1)
    var = request.args.get('var', default=1)
    hbvar = request.args.get('hbvar', default=1)
    section = request.args.get('section', default=1)

# setting some defaults
    if hbactive == 1:
        hbactive = 'hbboxtemp'

    if hbvar == 1:
        hbvar = 'hbboxtemp'


# reading sqlite info
    dbfile = "%s/%s" % (config['dbpath'], config['dbfile'])

    con = sqlite3.connect(dbfile)
    con.row_factory = dict_factory
    cur = con.cursor()
    cur.execute('SELECT equipment_name,type,name,height,model,install_date FROM equipment WHERE tower_name=? AND show=1 ORDER BY equipment_name', (tower_name,))
    qresult = cur.fetchall()

    cur.execute('SELECT description FROM towers WHERE short_name=?', (tower_name, ))
    details = cur.fetchone()

    # # Equipment button width on the Tower page
    # bwidth = 100./len(list(qresult))-len(list(qresult))

    variables = (0,)
    lastupdate, dlastupdate, alert = 0, 0, 0

    if section == 'sonic' or section == 'meteo' or section == 'stat':
        if active == 1:
            lastupdate, dlastupdate, alert = 0, 0, 0
        else:
            lastupdate, dlastupdate, alert = get_lastupdate_data_l0(tower_name, active)
            cur.execute('SELECT name,short_name,long_name,units FROM variables WHERE tower_name=? AND equipment_name=?', (tower_name, active))
            variables = cur.fetchall()

            # new_variables = [d for d in variables
            #                  if d["name"] not in {"e1", "e2", "e3", "e4", "e5"}
            #                  ]
            # if len(new_variables) < len(variables):
            #     variables = new_variables
            #     variables.append({'name': 'vibr', 'short_name': 'vibrations', 'long_name': 'Vibrations as inclinometer data', 'units': 'deg'})

            # rm unnecessary variables (currently the device temperature)
            variables = [d for d in variables
                             if d["name"] not in {"e1", "e2"}
                             ]

            # update e3 and e4 with pitch and roll
            for d in variables:
                if d["name"] == "e3":
                    d.update({'name': 'e3', 'short_name': 'roll', 'long_name': 'roll', 'units': 'deg'})
                elif d["name"] == "e4":
                    d.update({'name': 'e4', 'short_name': 'pitch', 'long_name': 'pitch', 'units': 'deg'})

    elif section == 'hb':
        lastupdate, dlastupdate, alert = get_lastupdate_rrd(tower_name)
        variables = [
            {'name': 'hbtemp', 'long_name': 'BOX & CPU temperature'},
            {'name': 'hbmem',  'long_name': 'Memory consumption'},
            {'name': 'hbnet',  'long_name': 'Network usage'}
        ]

    var_longname = ''
    if section != 1 and section != 'stat':
        if var != 1:
            if var not in [v['name'] for v in variables]:
                var = list(variables[0].values())[0]  # variables['name'][0]
        else:
            var = list(variables[0].values())[0]

        for d in variables:
            if d['name'] == var:
                var_longname = d['long_name']
                break

    con.close()

    return render_template('rtdata.html', tower=tower_name, city=tower_city,
        details=details,
        section=section,
        data=qresult,
        active=active,
        hbactive=hbactive,
        var=var,
        var_longname=var_longname,
        variables=variables,
        hbvar=hbvar,
        lastupdate=lastupdate,
        dlastupdate=dlastupdate,
        alert=alert,
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
