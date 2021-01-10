#!/bin/bash
name='msu'
cdate=$(date +%s)
sleep 1

FILE=${name}_temp.rrd
if [ ! -f "$FILE" ]; then
	rrdtool create ${FILE} --start=$cdate --step=300 \
		DS:temperature:GAUGE:600:-30:200 \
		RRA:AVERAGE:0.3:1:60 \
		RRA:AVERAGE:0.3:30:48 \
		RRA:AVERAGE:0.3:60:168 \
		RRA:AVERAGE:0.3:1440:30 \
		RRA:AVERAGE:0.3:7200:72
else
	echo "DataBase exists!"
fi

FILE=${name}_hdd.rrd
if [ ! -f "$FILE" ]; then
        rrdtool create ${FILE} --start=$cdate --step=300 \
                DS:temperature:GAUGE:600:-30:200 \
                RRA:AVERAGE:0.3:1:60 \
                RRA:AVERAGE:0.3:30:48 \
                RRA:AVERAGE:0.3:60:168 \
                RRA:AVERAGE:0.3:1440:30 \
                RRA:AVERAGE:0.3:7200:72
else
        echo "DataBase exists!"
fi

FILE=${name}_ram.rrd
if [ ! -f "$FILE" ]; then
	rrdtool create ${FILE} --start=$cdate --step=300 \
                DS:temperature:GAUGE:600:-30:200 \
                RRA:AVERAGE:0.3:1:60 \
                RRA:AVERAGE:0.3:30:48 \
                RRA:AVERAGE:0.3:60:168 \
                RRA:AVERAGE:0.3:1440:30 \
                RRA:AVERAGE:0.3:7200:72
else
        echo "DataBase exists!"
fi

