"""
Script to plot single-level fields from the Global Ensemble Forecast System (GEFS) using 0.5-degree data.
    Contours show the ensemble mean, shading shows the interquartile range among the 30 members.

Available maps: MSLP, 500-hPa geopotential height, 10-m wind speed, 200-hPa wind speed
    NOTE: the first member ("1") corresponds with the ensemble mean. Entries 2-31 = members 1-30.

Credit goes to Tomer Burg for the original map concept.
This version relies on the NOMADS GrADS Data Server.

Running the script:
    python GEFS_singleLevel_maps.py [InitializationTime] [field]
    Example--
        python GEFS_singleLevel_maps.py 2022092312 wind10m

NOTE:
    If you get an error that looks like the following:
        libffi.so.7: cannot open shared object file: No such file or directory
    one solution is to look for 'libffi.so.7' elsewhere in your conda installation and then create a symbolic link
    pointing to the file you found.
    Example search:
        find /home/[user]/miniconda3/ -name "libffi.so.*"   # change [user] to your username, without brackets
    This might return a file such as
        /home/[user]/miniconda3/lib/libffi.so.7.1.0
    To create a symbolic link pointing to that file, you'd use the following command:
        ln -s /home/[user]/miniconda3/lib/libffi.so.7.1.0 /home/[user]/miniconda3/envs/[envname]/lib/libffi.so.7
    where [envname] is the name of your conda environment.
"""
from datetime import datetime, timedelta
# from metpy.plots import USCOUNTIES  # use this import statement if you'd like to display US county borders
from scipy import stats
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import os
import sys
import xarray as xr


def addcolorbar(figure, axis, imagedata, ticks, cbar_label):
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
    cbar.set_label(cbar_label, size=11, weight='bold')


if len(sys.argv) < 2:
    exit('    Provide the GEFS initialization time in YYYYMMDDHH format when executing this script.')

#### DEFAULT SETTINGS
field = 'hgtprs'
level = 500.
if len(sys.argv) > 2:
    field = sys.argv[2].lower()
    if field == 'windprs':
        level = 200.
    validFields = 'hgtprs prmslmsl wind10m windprs'
    if validFields.find(field) == -1:
        print('Provided field is not set up for plotting in this script! ')
        print('    Valid field options: %s' % validFields)
        exit('    Exiting...')

InitTime = datetime.strptime(sys.argv[1], '%Y%m%d%H')
timeList = [np.datetime64(InitTime+timedelta(hours=h)) for h in range(0,169,6)]  # every 6 hours out to 120 hours
domain = [-105., -50., 5., 40.]  # western North Atlantic [min longitude, max longitude, min latitude, max latitude]

#### Load the remote dataset via the NOMADS GrADS Data Server
URL = 'http://nomads.ncep.noaa.gov:80/dods/gefs/gefs' + InitTime.strftime('%Y%m%d/gefs_pgrb2ap5_all_%Hz')
try:
    ds = xr.open_dataset(URL)
except OSError:
    exit('    URL for desired GEFS initialization time not available.')

#### Create dictionary of dimensions for pulling data
#    NOMADS-provided GEFS files have consistent dimensions, which is nice
dims = {'lat': slice(domain[2],domain[3]), 'lon': slice(domain[0]+360.,domain[1]+360.), 'time': timeList, 'lev': level}
if field == 'prmslmsl' or field == 'wind10m':
    dims = {'lat': slice(domain[2],domain[3]), 'lon': slice(domain[0]+360.,domain[1]+360.), 'time': timeList}

#### Request data and compute interquartile range (IQR) from the ensemble members
if field.startswith('wind') is True:
    suffix = field[4:]  # get 'prs' or '10m'
    U = ds['ugrd%s'%suffix].sel(**dims)
    V = ds['vgrd%s'%suffix].sel(**dims)
    subset = np.sqrt(U**2. + V**2.)  # this step takes a while ('iqr' step below is faster after this)
else:
    subset = ds[field].sel(**dims)  # 'iqr' step below takes a while after this call

iqr = stats.iqr(subset.sel(ens=slice(2,31)), axis=0)  # axis=0 corresponds with the ensembles
mean = subset.sel(ens=1)

#### Create maps for these time steps
#    mtype = PNG file name for the given field (short for 'map type')
#    titles = the field-specific string to add to the map title
#    cbartitles = the field-specific string to add to the colorbar
#    levels = the field-specific set of intervals to use for IQR shading
#    lines = the field-specific set of intervals to use for ensemble mean contours
#    adjust = the field-specific value to multiply by for nicer display units (example: Pa -> hPa)
mtype = {'hgtprs': 'height%d'%level, 'prmslmsl': 'MSLP', 'wind10m': 'wind10m', 'windprs': 'wind%d'%level}
titles = {'hgtprs': 'GEFS %d-hPa geopotential height [gpm]'%level, 'prmslmsl': 'GEFS mean sea level pressure [hPa]',
          'wind10m': 'GEFS 10-m wind speed [kt]', 'windprs': 'GEFS %d-hPa wind speed [kt]'%level}
cbartitles = {'hgtprs': '%d-hPa geopotential height interquartile range [gpm]'%level,
              'prmslmsl': 'mean sea level pressure interquartile range [hPa]',
              'wind10m': '10-m wind speed interquartile range [kt]',
              'windprs': '%d-hPa wind speed interquartile range [kt]'%level}
levels = {'hgtprs': np.arange(5., 61., 5.), 'prmslmsl': np.arange(2.,21.,1.), 'wind10m': np.arange(4.,41.,2.),
          'windprs': np.arange(8., 49., 4.)}
lines = {'hgtprs': np.arange(5000.,6300.,20.), 'prmslmsl': np.arange(900.,1060.,4), 'wind10m': np.arange(10.,101.,10.),
         'windprs': np.arange(10.,151.,20.)}
adjust = {'hgtprs': 1., 'prmslmsl': 1e-2, 'wind10m': 1.943844, 'windprs': 1.943844}
cmap = plt.get_cmap('viridis_r')  # colormap for IQR shading
pc = ccrs.PlateCarree()
merc = ccrs.Mercator()
coasts = cfeature.COASTLINE.with_scale('50m')
countries = cfeature.BORDERS.with_scale('50m')
states = cfeature.STATES.with_scale('50m')
lat = mean['lat'].values
lon = mean['lon'].values
lon[lon>180] -= 360.  # convert western longitudes to negative values
lon2d, lat2d = np.meshgrid(lon, lat)  # create 2-D lat/lon grids
for t in range(len(timeList)):
    dt = datetime.strptime(str(timeList[t])[0:13],'%Y-%m-%dT%H')
    diff = dt - InitTime
    forecastHour = int(diff.days*24 + diff.seconds/(60*60))
    PNGname = 'maps/GEFS_%sspread_%s_f%03d.png' % (mtype[field],sys.argv[1],forecastHour)  # output map name
    if os.path.exists(PNGname) is True:
        continue  # skip this time if the output map PNG file already exists!
    print(datetime.now().strftime('%H:%M:%S'), PNGname)
    fig, ax = plt.subplots(figsize=(12,7.5), subplot_kw={'projection': merc})
    ax.set_extent(domain, pc)
    ax.add_feature(coasts, linewidth=1., edgecolor='k', facecolor='none', zorder=5)
    ax.add_feature(countries, linewidth=1., edgecolor='k', facecolor='none', zorder=5)
    ax.add_feature(states, linewidth=1., edgecolor='k', facecolor='none', zorder=5)
    im = ax.contourf(lon2d, lat2d, iqr[t,:,:]*adjust[field], levels[field], cmap=cmap, extend='max', transform=pc,
                     transform_first=True)
    cl = ax.contour(lon2d, lat2d, mean[t,:,:]*adjust[field], lines[field], colors='blue', linewidths=0.75, transform=pc,
                    transform_first=True)
    ax.clabel(cl, cl.levels[::2], inline=True, fmt='%d', fontsize=8)
    gl = ax.gridlines(crs=pc, draw_labels=True, x_inline=False, linewidth=0.25)
    gl.top_labels, gl.right_labels, gl.rotate_labels = [False] * 3
    gl.xlocator = mticker.FixedLocator(np.arange(-180., 180., 10.))
    gl.ylocator = mticker.FixedLocator(np.arange(-90., 90., 5.))
    gl.xlabel_style = {'size': 11}
    gl.ylabel_style = {'size': 11}
    ax.set_title('forecast hour %03d    [init: %s]\nvalid at %s'
                 % (forecastHour,InitTime.strftime('%HZ %d %b %Y'),dt.strftime('%HZ %d %b %Y')), loc='left')
    ax.set_title(titles[field]+'\nensemble mean (contours) and IQR (shading)', loc='right', weight='bold')
    ax.spines['geo'].set_zorder(100)
    plt.tight_layout()
    addcolorbar(fig, ax, im, levels[field], cbartitles[field])
    # plt.show()  # uncomment to display the figure
    # break  # uncomment for testing
    plt.savefig(PNGname, dpi=120)
    plt.close()

ds.close()
