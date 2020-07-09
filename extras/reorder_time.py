import netCDF4 as nc
import numpy as np
# from datetime import datetime, time
# from pathlib import Path

fout = "out.nc"

ncin = nc.MFDataset('*.nc')

times = ncin.variables['time'][:]
u = ncin.variables['u'][:]
v = ncin.variables['v'][:]
w = ncin.variables['w'][:]
temp = ncin.variables['temp'][:]
# e1 = ncin.variables['e1'][:]
# e2 = ncin.variables['e2'][:]
# e3 = ncin.variables['e3'][:]
# e4 = ncin.variables['e4'][:]


indxs = np.argsort(times)


ncout = nc.Dataset(fout, 'w', clobber=False, format='NETCDF4_CLASSIC')
timed = ncout.createDimension("time", None)

timev = ncout.createVariable("time", "f8", ("time",))
timev.units = "seconds since 1970-01-01 00:00:00.0"
timev.calendar = "gregorian"

tempvar = ncout.createVariable("temp", "f4", ("time",), fill_value=-999.)
tempvar.short_name = "temperature"
tempvar.long_name = "Acoustic temperature"
tempvar.description = "Acoustic temperature"
tempvar.units = "degC"
tempvar.missing_value = -999.
tempvar.coordinates = "time"

uvar = ncout.createVariable("u", "f4", ("time",), fill_value=-999.)
uvar.short_name = "u"
uvar.long_name = "Earth-relative zonal wind"
uvar.description = "Earth-relative zonal wind"
uvar.units = "m s-1"
uvar.missing_value = -999.
uvar.coordinates = "time"

vvar = ncout.createVariable("v", "f4", ("time",), fill_value=-999.)
vvar.short_name = "v"
vvar.long_name = "Earth-relative meridional wind"
vvar.description = "Earth-relative meridional wind"
vvar.units = "m s-1"
vvar.missing_value = -999.
vvar.coordinates = "time"

wvar = ncout.createVariable("w", "f4", ("time",), fill_value=-999.)
wvar.short_name = "w"
wvar.long_name = "Vertical wind component"
wvar.description = "Vertical wind component"
wvar.units = "m s-1"
wvar.missing_value = -999.
wvar.coordinates = "time"

# ### Inclinometer
# e1var = ncout.createVariable("e1", "f4", ("time",), fill_value=-999.)
# e1var.short_name = "e1"
# e1var.long_name = ""
# e1var.description = ""
# e1var.units = ""
# e1var.missing_value = -999.
# e1var.coordinates = "time"

# e2var = ncout.createVariable("e2", "f4", ("time",), fill_value=-999.)
# e2var.short_name = "e2"
# e2var.long_name = ""
# e2var.description = ""
# e2var.units = ""
# e2var.missing_value = -999.
# e2var.coordinates = "time"

# e3var = ncout.createVariable("e3", "f4", ("time",), fill_value=-999.)
# e3var.short_name = "e3"
# e3var.long_name = ""
# e3var.description = ""
# e3var.units = ""
# e3var.missing_value = -999.
# e3var.coordinates = "time"

# e4var = ncout.createVariable("e4", "f4", ("time",), fill_value=-999.)
# e4var.short_name = "e4"
# e4var.long_name = ""
# e4var.description = ""
# e4var.units = ""
# e4var.missing_value = -999.
# e4var.coordinates = "time"

ncout['time'][:] = times[indxs]
ncout['u'][:] = u[indxs]
ncout['v'][:] = v[indxs]
ncout['w'][:] = w[indxs]
ncout['temp'][:] = temp[indxs]
# ncout['e1'][:] = e1[indxs]
# ncout['e2'][:] = e2[indxs]
# ncout['e3'][:] = e3[indxs]
# ncout['e4'][:] = e4[indxs]
ncout.close()
