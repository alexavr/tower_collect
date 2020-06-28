#!/bin/bash

src="/var/www/data/domains/tower.ocean.ru/html/flask/data/hb/"
dst="/var/www/data/domains/tower.ocean.ru/html/flask/static/"

sched=( '1d' '1w' '1m' '1y' )
width='800'
height='150'
xgrid=('HOUR:1:HOUR:3:HOUR:3:0:%H:00' \
       'HOUR:12:DAY:1:DAY:1:0:%d/%b' \
       'DAY:1:DAY:7:DAY:3:0:%d/%b' \
       'MONTH:1:MONTH:1:MONTH:1:0:%m/%Y')

cd ${src}

for i in $(ls *.rrd); do
    stname="${i%.*}"

    ii=0
    for d in ${sched[@]}; do

        rrdtool graph \
            "${dst}${stname}_hbtemp_${d}.png" \
            --start -${d} \
            --width ${width} \
            --height ${height} \
            DEF:temp=./${stname}.rrd:temp:AVERAGE \
            AREA:temp#FA8072:'CPU temperature, degC' \
            LINE:temp#B22222: \
            --x-grid ${xgrid[ii]} \
            --vertical-label='degC'
            # --horizontal-label='time (UTC)'

        echo "$ii : ${xgrid[ii]}"

        rrdtool graph \
            "${dst}${stname}_hbmem_${d}.png" \
            --start -${d} \
            --width ${width} \
            --height ${height} \
            DEF:hdd=./${stname}.rrd:hdd:AVERAGE \
            DEF:ram=./${stname}.rrd:ram:AVERAGE \
            AREA:hdd#A9A9A9:'HDD, %'                \
            LINE:hdd#696969:                    \
            LINE:ram#FF0000:'RAM, %' \
            --x-grid ${xgrid[ii]} \
            --vertical-label=%

        rrdtool graph \
            "${dst}${stname}_hbnet_${d}.png" \
            --start -${d} \
            --width ${width} \
            --height ${height} \
            DEF:in=./${stname}.rrd:in:AVERAGE \
            DEF:out=./${stname}.rrd:out:AVERAGE \
            CDEF:out_neg=out,-1,*               \
            AREA:in#32CD32:Incoming             \
            LINE:in#336600:                     \
            AREA:out_neg#4169E1:Outgoing        \
            LINE:out_neg#0033CC: \
            --x-grid ${xgrid[ii]} \
            --vertical-label='bytes per second'


        ii=$((ii+1))



    done


done

#
