This is server part of turbulence measurement project. Receives data (`.npz` files) from mast measurements, converts into NetCDF, draws preliminary plots.  The comprehensive description will be provided later.

## Crontab settings:
```bash
# Prep data for web plots
5,35 * * * * /var/www/data/domains/tower.ocean.ru/html/flask/scripts/npz2buffer.sh > /var/www/data/domains/tower.ocean.ru/html/flask/scripts/npz2buffer.log 2>&1
10,45 * * * * /var/www/data/domains/tower.ocean.ru/html/flask/scripts/check_state.sh > /var/www/data/domains/tower.ocean.ru/html/flask/scripts/check_state.log 2>&1 # Telegram notifications
# Generate plots for the web page
10,40 * * * * /var/www/data/domains/tower.ocean.ru/html/flask/scripts/plot_rtdata.sh > /var/www/data/domains/tower.ocean.ru/html/flask/scripts/plot_rtdata.log 2>&1
# Refresh plots of HeartBeats info
*/2 * * * * /var/www/data/domains/tower.ocean.ru/html/flask/scripts/plot_hb.sh > /var/www/data/domains/tower.ocean.ru/html/flask/scripts/plot_hb.log 2>&1
# Convert BUFFER into NetCDF (+sorting) 
50 0 * * * /var/www/data/domains/tower.ocean.ru/html/flask/scripts/buffer2nc.sh > /var/www/data/domains/tower.ocean.ru/html/flask/scripts/buffer2nc.log 2>&1
```
