from datetime import datetime, timedelta
import rrdtool
import sqlite3
# import glob
# import os
# import netCDF4 as nc
import configparser
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from string import Template

CONFIG_NAME = "../tower.conf"
MY_ADDRESS = "gavr@ocean.ru"
MY_PASSWORD = "fG7HYnYH"

# message_template = "Dear group of ${PERSON_NAME} users. "

def read_config(path):

    config = configparser.ConfigParser(
        interpolation=configparser.ExtendedInterpolation()
    )
    config.read(path)

    settings = dict(config["paths"])

    return settings


def get_lastupdate_rrd(stname):
    lastupdate = rrdtool.lastupdate(config['rrdpath']+"/"+stname+".rrd")
    lastupdate_time = lastupdate["date"]
    lastupdate_dtime = datetime.utcnow() - lastupdate_time
    lastupdate_dtime -= timedelta(microseconds=lastupdate_dtime.microseconds)
    # lastupdate_time_str = lastupdate_time.strftime("%Y-%m-%d %H:%M:%S UTC")
    minutes = lastupdate_dtime.seconds/60.

    status = 'online'
    if minutes > 3:
        status = 'offline'

    return status


def get_emails(stname):

    dbfile = "%s/%s" % (config['dbpath'], config['dbfile'])

    con = sqlite3.connect(dbfile)
    cur = con.cursor()
    cur.execute('SELECT alerts FROM towers WHERE short_name=?', (stname,))
    temails = cur.fetchone()
    con.close()

    emails = temails[0].split()

    return emails


def get_status(stname, equipment_name='hb'):

    dbfile = "%s/%s" % (config['dbpath'], config['dbfile'])

    con = sqlite3.connect(dbfile)
    cur = con.cursor()
    if equipment_name == 'hb':
        cur.execute('SELECT status FROM towers WHERE short_name=?', (stname,))
    else:
        cur.execute('SELECT status FROM equipment WHERE tower_name=? AND equipment_name=?', (stname, equipment_name))

    status_tmp = cur.fetchone()
    con.close()

    status = status_tmp[0]

    return status


def update_status(stname, new_status, equipment_name='hb'):

    dbfile = "%s/%s" % (config['dbpath'], config['dbfile'])

    con = sqlite3.connect(dbfile)
    cur = con.cursor()

    if equipment_name == 'hb':
        cur.execute('UPDATE towers SET status=? WHERE short_name=?', (new_status, stname))
    else:
        cur.execute('UPDATE equipment SET status=? WHERE tower_name=? AND equipment_name=? ', (new_status, stname, equipment_name))

    con.commit()
    con.close()

    # return 0


def send_emails(tower_name, device_name, status, emails):

    s = smtplib.SMTP(host='mail.ocean.ru', port=465)  # 465 SSL; 587 TLS
    s.starttls()
    # s = smtplib.SMTP_SSL('mail.ocean.ru')
    s.login(MY_ADDRESS, MY_PASSWORD)

    for email in emails:
        msg = MIMEMultipart()
        #
        message = "See subject"
        msg['From'] = MY_ADDRESS
        msg['To'] = email

        if device_name == 'hb':
            msg['Subject'] = "Tower %s has changed status to %s"%(tower_name, status)
            message_template = read_template("./checklink.template_hb")
            message = message_template.substitute(
                TOWER_NAME=tower_name.upper(),
                STATUS=status.upper()
                )
        else:
            msg['Subject'] = "The %s at tower %s has changed status to %s"%(device_name, tower_name, status)
            message_template = read_template("./checklink.template_general")
            message = message_template.substitute(
                TOWER_NAME=tower_name.upper(),
                DEVICE_NAME=device_name.upper(),
                STATUS=status.upper()
                )

        # add in the message body
        msg.attach(MIMEText(message, 'plain'))

        print(email)
        print(msg)

        # send the message via the server set up earlier.
        s.set_debuglevel(1)
        s.send_message(msg)

        del msg


def read_template(filename):
    with open(filename, 'r', encoding='utf-8') as template_file:
        template_file_content = template_file.read()
    return Template(template_file_content)

### MAIN PROGRAM


stname = 'MSU'

config = read_config(CONFIG_NAME)
status_old = get_status(stname)
emails = get_emails(stname)

status_new = get_lastupdate_rrd(stname)

# print(status_old)
# print(status_new)
# print(time_str)
# print(emails)

send_emails(stname, 'hb', status_new, emails)


if status_new != status_old:
    print("Status has changed. Sending alerts...")
    if status_new == 'offline':
        print("Send emails. SYSTEM WENT OFFLINE")
    if status_new == 'online':
        print("Send emails. SYSTEM WENT ONLINE")
        print("     -> Change status in sqlite3")

    # send_emails(emails, status_new, time_str)
    update_status(stname, status_new)



