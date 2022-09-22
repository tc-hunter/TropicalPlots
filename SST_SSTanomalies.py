"""
Script to plot daily sea surface temperature (SST) as contours overlaid on SST anomalies (shaded).
This version relies on the NOAA NCEI THREDDS server hosting the optimum interpolated SST daily 0.25-degree dataset.

Daily fields are *usually* available around 13 UTC the day after. For example, SSTs for 20 September would appear around
13 UTC 21 September. However, delays happen! You can explore the available files on the catalog here:
    https://www.ncei.noaa.gov/thredds/catalog/OisstBase/NetCDF/V2.1/AVHRR/catalog.html
(can be slow to load)

Daily long-term mean fields are provided by NOAA's PSL:
    https://psl.noaa.gov/thredds/dodsC/Datasets/noaa.oisst.v2.highres/sst.day.mean.ltm.1991-2020.nc
"""
from datetime import datetime, timedelta
# from metpy.plots import USCOUNTIES  # use this import statement if you'd like to display US county borders
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cftime
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import os
import sys
import xarray as xr


def addcolorbar(figure, axis, imagedata, ticks):
    """
    Custom function to add a colorbar sized to the map size.
    If your image width is too small, weird things can happen due to the use of plt.tight_layout(). Remove this call as
        needed.
    """
    axes_bbox = axis.get_position()
    left = axes_bbox.x1 + 0.015
    bottom = axes_bbox.y0
    width = 0.015
    height = axes_bbox.y1 - bottom
    cax = figure.add_axes([left, bottom, width, height])
    cbar = plt.colorbar(imagedata, cax=cax, ticks=ticks, orientation='vertical', extendfrac='auto')
    cbar.ax.tick_params(labelsize=10)
    cbar_label = 'daily sea surface temperature anomaly [\N{DEGREE SIGN}C]'
    cbar.set_label(cbar_label, size=11, weight='bold')


if len(sys.argv) < 2:
    exit('    Provide the desired date for SST in YYYYMMDD format when executing this script.')

#### DEFAULT SETTINGS
RequestedDate = datetime.strptime(sys.argv[1], '%Y%m%d')  # date for which to display SST + SST anomaly data
domain = [-105., -50., 5., 40.]  # western North Atlantic [min longitude, max longitude, min latitude, max latitude]

#### Check that the output map doesn't already exist
PNGname = RequestedDate.strftime('NOAA_OISST_%Y%m%d.png')
if os.path.exists(PNGname) is True:  # (comment this if statement when you want to be able to overwrite an existing map)
    exit('    Output map for this date already exists! Exiting...')

#### Load the remote dataset via the NOAA NCEI THREDDS server
#    NOTE: there is an anomaly field provided in this file, but it's relative to 1971-2000.
baseURL = 'https://www.ncei.noaa.gov/thredds/dodsC/OisstBase/NetCDF/V2.1/AVHRR/'
try:
    ## Preliminary SST fields (most recent)
    ds = xr.open_dataset(baseURL + RequestedDate.strftime('%Y%m/oisst-avhrr-v02r01.%Y%m%d_preliminary.nc'))
except OSError:
    try:
        ## Non-preliminary SST fields (older)
        ds = xr.open_dataset(baseURL + RequestedDate.strftime('%Y%m/oisst-avhrr-v02r01.%Y%m%d.nc'))
    except OSError:
        exit('    remote netCDF file for desired date is not available! Exiting...')

#### Now access the 1991-2020 long-term mean file from NOAA PSL
#    NOTE: the time stamps are weird in this file. Lines after opening the dataset help address this issue.
#    NOTE 2: there is no long-term mean entry for February 29
dsLTM = xr.open_dataset('https://psl.noaa.gov/thredds/dodsC/Datasets/noaa.oisst.v2.highres/sst.day.mean.ltm.1991-2020.nc')
month = int(RequestedDate.strftime('%m'))  # store month as an integer
day = int(RequestedDate.strftime('%d'))  # store day as an integer
timeLTM = cftime.DatetimeGregorian(1,month,day)

#### Pull the desired fields from the remote dataset (nothing is downloaded yet!)
latslice = slice(domain[2]-0.25,domain[3]+0.25)
lonslice = slice(domain[0]-0.25+360.,domain[1]+0.25+360.)
sst = ds['sst'].sel(time=np.datetime64(RequestedDate+timedelta(hours=12)),zlev=0,lat=latslice,lon=lonslice)
sstLTM = dsLTM['sst'].sel(time=timeLTM,lat=latslice,lon=lonslice)

#### Compute anomalies
anom = sst - sstLTM

#### Create custom colormap for anomaly shading so that values near 0 are white
clevels = np.append(np.arange(-4., -0.4, 0.5), np.arange(0.5, 4.1, 0.5))  # -4C to 4C
cmap1 = plt.get_cmap('bwr')(np.linspace(0., 0.45, int(clevels.size/2)))  # colors for negative values
cmap2 = plt.get_cmap('Greys')(0)
cmap3 = plt.get_cmap('bwr')(np.linspace(0.55, 1., int(clevels.size/2)))  # colors for positive values
colors = np.vstack((cmap1, cmap2, cmap3))
cmap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)

#### Create map of SST and SST anomalies
levels = np.arange(0.,35.,1.)  # levels for SST contours in degrees Celsius
pc = ccrs.PlateCarree()
merc = ccrs.Mercator()
coasts = cfeature.COASTLINE.with_scale('50m')
countries = cfeature.BORDERS.with_scale('50m')
states = cfeature.STATES.with_scale('50m')
land = cfeature.NaturalEarthFeature('physical', 'land', '50m', edgecolor='k', facecolor='silver')  # make land silver
lat = sst['lat'].values
lon = sst['lon'].values
lon[lon>180] -= 360.  # convert western longitudes to negative values
print(PNGname)
fig, ax = plt.subplots(figsize=(12,7.5), subplot_kw={'projection': merc})
ax.set_extent(domain, pc)
ax.add_feature(land, zorder=5)
ax.add_feature(coasts, linewidth=1., edgecolor='k', facecolor='none', zorder=5)
ax.add_feature(countries, linewidth=1., edgecolor='k', facecolor='none', zorder=5)
ax.add_feature(states, linewidth=1., edgecolor='k', facecolor='none', zorder=5)
im = ax.contourf(lon, lat, anom, clevels, cmap=cmap, extend='both', transform=pc)
cl = ax.contour(lon, lat, sst, levels, colors='k', linewidths=1, transform=pc)
ax.clabel(cl, cl.levels[::2], inline=True, fmt='%d', fontsize=8)
gl = ax.gridlines(crs=pc, draw_labels=True, x_inline=False, linewidth=0.25)
gl.top_labels, gl.right_labels, gl.rotate_labels = [False] * 3
gl.xlocator = mticker.FixedLocator(np.arange(-180., 180., 10.))
gl.ylocator = mticker.FixedLocator(np.arange(-90., 90., 5.))
gl.xlabel_style = {'size': 11}
gl.ylabel_style = {'size': 11}
ax.set_title(RequestedDate.strftime('NOAA OISST (contours) and anomalies (shading) for %d %b %Y'), loc='right',
             weight='bold')
ax.set_title('anomalies relative to 1991-2020 climatology', loc='left', size=9)
ax.spines['geo'].set_zorder(100)
plt.tight_layout()
addcolorbar(fig, ax, im, clevels)
# plt.show()  # uncomment to display the figure
plt.savefig(PNGname, dpi=120)
plt.close()

ds.close()

