"""
Script to plot pressure-weighted-mean relative humidity (RH) and winds along with mean sea level pressure.
Credit goes to Dr. Levi Cowan (Tropical Tidbits) for the original map concept.
This version relies on the NOMADS GrADS Data Server.

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
from metpy.units import units
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import metpy.calc as mpcalc
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
    cbar_label = '850-hPa relative vorticity [$\mathregular{10^{-5}}$ $\mathregular{s^{-1}}$]'
    cbar.set_label(cbar_label, size=11, weight='bold')


if len(sys.argv) < 2:
    exit('    Provide the GFS initialization time in YYYYMMDDHH format when executing this script.')

#### DEFAULT SETTINGS
InitTime = datetime.strptime(sys.argv[1], '%Y%m%d%H')
timeList = [np.datetime64(InitTime+timedelta(hours=h)) for h in range(0,121,6)]  # every 6 hours out to 120 hours
domain = [-105., -50., 5., 40.]  # western North Atlantic [min longitude, max longitude, min latitude, max latitude]

#### Load the remote dataset via the NOMADS GrADS Data Server
URL = 'http://nomads.ncep.noaa.gov:80/dods/gfs_0p25/gfs' + InitTime.strftime('%Y%m%d/gfs_0p25_%Hz')
try:
    ds = xr.open_dataset(URL)
except OSError:
    exit('    URL for desired GFS initialization time not available.')

#### Pull the desired fields from the remote dataset (nothing is downloaded yet!)
U = ds['ugrdprs']  # zonal wind on pressure levels [m/s]
V = ds['vgrdprs']  # meridional wind on pressure levels [m/s]
ABSV = ds['absvprs']  # absolute vorticity on pressure levels [1/s]
HGT = ds['hgtprs']  # geopotential height [gpm]

#### NOMADS-provided GFS files have consistent dimensions, which is nice
dims = {'lat': slice(domain[2],domain[3]), 'lon': slice(domain[0]+360.,domain[1]+360.), 'time': timeList, 'lev': 200.}
Usubset = U.sel(**dims)
Vsubset = V.sel(**dims)
dims = {'lat': slice(domain[2],domain[3]), 'lon': slice(domain[0]+360.,domain[1]+360.), 'time': timeList, 'lev': 500.}
HGTsubset = HGT.sel(**dims)
dims = {'lat': slice(domain[2],domain[3]), 'lon': slice(domain[0]+360.,domain[1]+360.), 'time': timeList, 'lev': 850.}
ABSVsubset = ABSV.sel(**dims)

#### Compute relative vorticity from absolute vorticity using MetPy's coriolis_parameter() function
lon2d, lat2d = np.meshgrid(ABSVsubset['lon'].values, ABSVsubset['lat'].values)  # create 2-D lat/lon grids
planetaryVorticity = mpcalc.coriolis_parameter(lat2d * units('deg')).m  # retain the magnitude of f [1/s]
relativeVorticity = (ABSVsubset.values - planetaryVorticity) * 1e5  # convert units from 1/s to 1e-5/s

#### Convert m/s to kt using MetPy!
Uvalues = (Usubset.values * units('m/s')).to(units('kt')).m  # retain NumPy array by dropping units (can keep if desired)
Vvalues = (Vsubset.values * units('m/s')).to(units('kt')).m  # retain NumPy array by dropping units (can keep if desired)

#### Create maps for these time steps
levels = np.arange(5.,71.,5.)  # levels for vorticity shading
cmap = plt.get_cmap('magma_r')  # colormap for vorticity shading
HGTlines = np.arange(500.,630.,2.)  # values for 500-hPa height contours [dam]
barbSize = {'emptybarb': 0., 'width': 0.2, 'height': 0.3}
pc = ccrs.PlateCarree()
merc = ccrs.Mercator()
coasts = cfeature.COASTLINE.with_scale('50m')
countries = cfeature.BORDERS.with_scale('50m')
states = cfeature.STATES.with_scale('50m')
lat = Usubset['lat'].values
lon = Usubset['lon'].values
lon[lon>180] -= 360.  # convert western longitudes to negative values
for t in range(len(timeList)):
    dt = datetime.strptime(str(timeList[t])[0:13],'%Y-%m-%dT%H')
    diff = dt - InitTime
    forecastHour = int(diff.days*24 + diff.seconds/(60*60))
    PNGname = 'GFS_vort850-height500-wind200_%s_f%03d.png' % (sys.argv[1],forecastHour)  # output map name
    if os.path.exists(PNGname) is True:
        continue  # skip this time if the output map PNG file already exists!
    print(PNGname)
    fig, ax = plt.subplots(figsize=(12,7.5), subplot_kw={'projection': merc})
    ax.set_extent(domain, pc)
    ax.add_feature(coasts, linewidth=1., edgecolor='k', facecolor='none', zorder=5)
    ax.add_feature(countries, linewidth=1., edgecolor='k', facecolor='none', zorder=5)
    ax.add_feature(states, linewidth=1., edgecolor='k', facecolor='none', zorder=5)
    im = ax.contourf(lon, lat, relativeVorticity[t,:,:], levels, cmap=cmap, extend='max', transform=pc)
    cl = ax.contour(lon, lat, HGTsubset.sel(time=timeList[t])/10., HGTlines, colors='blue', linewidths=0.75,
                    transform=pc)
    ax.clabel(cl, cl.levels[::2], inline=True, fmt='%d', fontsize=8)
    ax.barbs(lon[::8], lat[::8], Uvalues[t,::8,::8], Vvalues[t,::8,::8], pivot='tip', zorder=50, color='k', length=5,
             sizes=barbSize, lw=0.5, transform=pc)
    gl = ax.gridlines(crs=pc, draw_labels=True, x_inline=False, linewidth=0.25)
    gl.top_labels, gl.right_labels, gl.rotate_labels = [False] * 3
    gl.xlocator = mticker.FixedLocator(np.arange(-180., 180., 10.))
    gl.ylocator = mticker.FixedLocator(np.arange(-90., 90., 5.))
    gl.xlabel_style = {'size': 11}
    gl.ylabel_style = {'size': 11}
    ax.set_title('forecast hour %03d    [init: %s]\nvalid at %s'
                 % (forecastHour,InitTime.strftime('%HZ %d %b %Y'),dt.strftime('%HZ %d %b %Y')), loc='left')
    ax.set_title('850-hPa cyclonic relative vorticity [$\mathregular{10^{-5}}$ $\mathregular{s^{-1}}$]\n' +
                 '500-hPa height [dam], and 200-hPa wind [kt]', loc='right', weight='bold')
    ax.spines['geo'].set_zorder(100)
    plt.tight_layout()
    addcolorbar(fig, ax, im, levels)
    # plt.show()  # uncomment to display the figure
    plt.savefig(PNGname, dpi=120)
    plt.close()

ds.close()
