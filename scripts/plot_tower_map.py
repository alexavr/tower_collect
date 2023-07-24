import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt 
import pandas as pd
import tower_lib as tl
# import cartopy.io.img_tiles as cimgt

CONFIG_NAME = "../tower.conf"
config = tl.reader.config(CONFIG_NAME)
dbfile = "%s/%s" % (config['dbpath'], config['dbfile'])
cur = tl.reader.db_init(dbfile)
# towers = tl.reader.bd_get_table(cur,"SELECT city,short_name,lat,lon FROM towers")
towers = tl.reader.bd_get_table_df(cur,"SELECT city,short_name,lat,lon FROM towers")
print(towers)

# lats = [item[2] for item in towers]
# lons = [item[3] for item in towers]
# names = [item[1] for item in towers]
# cities = [item[0] for item in towers]

# d = {'cities': cities, 'names': names, 'lons': lons, 'lats': lats}
# towers = pd.DataFrame(data=d)

fig = plt.figure(constrained_layout=True, figsize=(7, 4))

ax = plt.axes(projection=ccrs.PlateCarree())
ax.coastlines('50m',linewidth=0.5, alpha=0.5)

ax.add_feature(cfeature.LAND, alpha=0.5)
ax.add_feature(cfeature.OCEAN, alpha=0.5)
ax.add_feature(cfeature.BORDERS, alpha=0.5, linewidth=0.5 )

ax.stock_img()

prj = ccrs.PlateCarree()
gl = ax.gridlines(crs=prj,
                  draw_labels=True, 
                  # x_inline=False, y_inline=False, 
                  linewidth=0.5, linestyle=":", color='gray', alpha=0.5)
gl.xlabels_top = False
gl.ylabels_right = False
# gl.xlines = False
# gl.xlocator = mticker.FixedLocator([-180, -45, 0, 45, 180])
# gl.xformatter = LONGITUDE_FORMATTER
# gl.yformatter = LATITUDE_FORMATTER
gl.xlabel_style = {'size': 6} # , 'color': 'gray'
gl.ylabel_style = {'size': 6} # , 'color': 'gray'
gl.rotate_labels=0

ax.set_extent([5, 55, 40, 70], crs=ccrs.PlateCarree())


for index, row in towers.iterrows():
    ax.plot(row['lon'], row['lat'], markersize=13, marker="*", color="tab:orange", markeredgecolor="black", markeredgewidth=0.5, transform=ccrs.PlateCarree())
    ax.plot(row['lon'], row['lat'], markersize=13, marker="*", color="tab:orange", markeredgecolor="black", markeredgewidth=0.5, transform=ccrs.PlateCarree())
    ax.text(row['lon'], row['lat']+0.7, row['city'], fontsize='small', horizontalalignment='center', transform=ccrs.PlateCarree()) # bbox=dict(facecolor='red', alpha=0.7), 

plt.savefig('../static/plot_tower_map.png', bbox_inches='tight', dpi=150)
# plt.show()

