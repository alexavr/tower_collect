# Fixing dublicates it data coused by connection gaps
# conda activate tower
#
# Gavrikov 2023-06-08

import xarray as xr
import sys
from pathlib import Path
import numpy as np

if len(sys.argv) != 2:
	print("Usage:")
	print("       python fix_netcdf_doubles.py FILENAME.nc")
	exit()
else:
	in_file = str(sys.argv[1])

ds_new = xr.Dataset()

with xr.open_dataset(in_file) as ds:
	# copy attrs
	for name, attr in ds.attrs.items():
		ds_new.attrs[name] = f"{attr}"

	# sort, drop and copy vars
	for iv in ds.keys():
		print(iv)
		ds_new[iv] = ds[iv].drop_duplicates(dim='time')


print("#######################################################################")
print("#######################################################################")
print(ds_new['u'])
print("#######################################################################")
print("#######################################################################")

# ENCODING

# comp = dict(zlib=True, complevel=5)
# encoding = {var: comp for var in ds_new.keys()}

encoding = {}
encoding_keys = ("_FillValue", "dtype")
for data_var in ds_new.data_vars:
    encoding[data_var] = {key: value for key, value in ds_new[data_var].encoding.items() if key in encoding_keys}
    encoding[data_var].update(zlib=True, complevel=5)

print(encoding)

print("#######################################################################")
print("#######################################################################")

if Path(in_file).is_file(): Path(in_file).unlink()
ds_new.to_netcdf(in_file, unlimited_dims=['time'], format="NETCDF4")
# ds[iv].to_netcdf(outfile ,mode='a')
