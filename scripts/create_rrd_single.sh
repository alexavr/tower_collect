#!/bin/bash
name=$1
cdate=$(date +%s)
sleep 1

FILE=${name}.rrd
if [ ! -f "$FILE" ]; then
        rrdtool create ${FILE} --start=$cdate --step=120 \
                DS:boxtemp:GAUGE:300:-30:100 \
                DS:cputemp:GAUGE:300:-30:100 \
                DS:hdd:GAUGE:300:0:100 \
                DS:ram:GAUGE:300:0:100 \
                DS:in:COUNTER:300:0:U \
                DS:out:COUNTER:300:0:U \
                RRA:AVERAGE:0.4:1:730 \
                RRA:AVERAGE:0.4:10:576 \
                RRA:AVERAGE:0.4:30:744 \
                RRA:AVERAGE:0.4:360:740
else
        echo "DataBase exists!"
fi
