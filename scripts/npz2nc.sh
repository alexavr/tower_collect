#!/bin/bash

if [ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]; then
	source ${HOME}/anaconda3/etc/profile.d/conda.sh
elif [[ condition ]]; then
	source /storage/kubrick/gavr/anaconda3/etc/profile.d/conda.sh
else
	echo "No conda environment found!"
	exit 1
fi

conda activate tower

cd /var/www/data/domains/tower.ocean.ru/html/flask/scripts/
python ./npz2nc.py
