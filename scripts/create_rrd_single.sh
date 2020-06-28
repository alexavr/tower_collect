#!/bin/bash
name='msu'
cdate=$(date +%s)
sleep 1

FILE=${name}.rrd
if [ ! -f "$FILE" ]; then
	rrdtool create ${FILE} --start=$cdate --step=120 \
		DS:temp:GAUGE:300:-30:200 \
		DS:hdd:GAUGE:300:0:100 \
		DS:ram:GAUGE:300:0:100 \
		DS:in:COUNTER:300:0:U \
		DS:out:COUNTER:300:0:U \
		RRA:AVERAGE:0.4:1:300 \
		RRA:AVERAGE:0.4:5:150 \
		RRA:AVERAGE:0.4:30:192 \
		RRA:AVERAGE:0.4:90:256 \
		RRA:AVERAGE:0.4:720:370
else
	echo "DataBase exists!"
fi

