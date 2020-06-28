This is server part of turbulence measurement project. Receives data (`.npz` files) from mast measurements, converts into NetCDF, draws preliminary plots.  The comprehensive description will be provided later.

## Crontab settings:
```bash
15,45 * * * * /path/to/scripts/plot_rtdata.sh > /path/to/scripts/plot_rtdata.log 2>&1
10,40 * * * * /path/to/scripts/npz2nc.sh > /path/to/scripts/npz2nc.log 2>&1
*/2   * * * * /path/to/scripts/plot_hb.sh > /path/to/scripts/plot_hb.log 2>&1
```
