"""
Script to plot pressure-weighted-mean relative humidity (RH) and winds along with mean sea level pressure.
Credit goes to Dr. Levi Cowan (Tropical Tidbits) for the original map concept.
This version relies on Unidata's Science Gateway THREDDS server.

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
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import os
import sys
import xarray as xr


def addcolorbar(figure, axis, imagedata, ticks, level1, level2):
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
    cbar_label = '%d-%d-hPa' % (level1,level2) + ' relative humidity [%]'
    cbar.set_label(cbar_label, size=11, weight='bold')


if len(sys.argv) < 2:
    exit('    Provide the GFS initialization time in YYYYMMDDHH format when executing this script.')

#### DEFAULT SETTINGS
InitTime = datetime.strptime(sys.argv[1], '%Y%m%d%H')
pLevel1 = 700.  # lower boundary of layer in hPa
pLevel2 = 300.  # upper boundary of layer in hPa
timeList = [np.datetime64(InitTime+timedelta(hours=h)) for h in range(0,121,6)]  # every 6 hours out to 120 hours
domain = [-105., -50., 5., 40.]  # western North Atlantic [min longitude, max longitude, min latitude, max latitude]

#### Load the remote dataset via Unidata's Science Gateway THREDDS server
URL = 'https://tds.scigw.unidata.ucar.edu/thredds/dodsC/grib/NCEP/GFS/Global_0p25deg/GFS_Global_0p25deg_' + \
      InitTime.strftime('%Y%m%d_%H00.grib2')
try:
    ds = xr.open_dataset(URL)
except OSError:
    exit('    URL for desired GFS initialization time not available.')

#### Pull the desired fields from the remote dataset (nothing is downloaded yet!)
U = ds['u-component_of_wind_isobaric']  # zonal wind on pressure levels
V = ds['v-component_of_wind_isobaric']  # meridional wind on pressure levels
RH = ds['Relative_humidity_isobaric']  # relative humidity on pressure levels
MSLP = ds['MSLP_Eta_model_reduction_msl']  # mean sea level pressure using the Eta model reduction

#### Check the data dimensions to then extract the desired domain because GFS coordinate names can vary!
Ucoords = list(U.coords)  # lat, lon, reftime, time, levels
Udims = {Ucoords[0]: slice(domain[3],domain[2]), Ucoords[1]: slice(domain[0]+360.,domain[1]+360.),
         Ucoords[3]: timeList, Ucoords[4]: slice(pLevel2*100.,pLevel1*100.)}
Usubset = U.sel(**Udims)
Vcoords = list(V.coords)
Vdims = {Vcoords[0]: slice(domain[3],domain[2]), Vcoords[1]: slice(domain[0]+360.,domain[1]+360.),
         Vcoords[3]: timeList, Vcoords[4]: slice(pLevel2*100.,pLevel1*100.)}
Vsubset = V.sel(**Vdims)
RHcoords = list(RH.coords)
RHdims = {RHcoords[0]: slice(domain[3],domain[2]), RHcoords[1]: slice(domain[0]+360.,domain[1]+360.),
          RHcoords[3]: timeList, RHcoords[4]: slice(pLevel2*100.,pLevel1*100.)}
RHsubset = RH.sel(**RHdims)
MSLPcoords = list(MSLP.coords)
MSLPdims = {MSLPcoords[0]: slice(domain[3],domain[2]), MSLPcoords[1]: slice(domain[0]+360.,domain[1]+360.),
            MSLPcoords[3]: timeList}
MSLPsubset = MSLP.sel(**MSLPdims)

#### Compute pressure-weighted layer averages
#    NOTE: levels for 700-300 hPa happen to be evenly-spaced every 50 hPa, but that's not always the case for
#    pressure-level data!
print(datetime.now().strftime('    %H:%M:%S -- pulling zonal winds'))
Uweighted = np.zeros(Usubset.shape, dtype=float)
Ulevels = Usubset[Ucoords[4]].values
for l in range(Ulevels.size):
    coords = {Ucoords[4]: Ulevels[l]}
    Uweighted[:,l,:,:] = Usubset.sel(**coords).values * 5000.
print(datetime.now().strftime('    %H:%M:%S -- pulling meridional winds'))
Vweighted = np.zeros(Vsubset.shape, dtype=float)
Vlevels = Vsubset[Vcoords[4]].values
for l in range(Vlevels.size):
    coords = {Vcoords[4]: Vlevels[l]}
    Vweighted[:,l,:,:] = Vsubset.sel(**coords).values * 5000.
print(datetime.now().strftime('    %H:%M:%S -- pulling relative humidity'))
RHweighted = np.zeros(RHsubset.shape, dtype=float)
RHlevels = RHsubset[RHcoords[4]].values
for l in range(RHlevels.size):
    coords = {RHcoords[4]: RHlevels[l]}
    RHweighted[:,l,:,:] = RHsubset.sel(**coords).values * 5000.
Umean = (np.sum(Uweighted,axis=1) / (5000. * float(Ulevels.size))) * 1.943844  # m/s --> kt
Vmean = (np.sum(Vweighted,axis=1) / (5000. * float(Vlevels.size))) * 1.943844  # m/s --> kt
RHmean = np.sum(RHweighted,axis=1) / (5000. * float(RHlevels.size))

#### Create maps for these time steps
levels = np.arange(5.,96.,5.)  # levels for RH shading
RHcmap = plt.get_cmap('BrBG')  # colormap for RH shading (browns = drier, greens = more moist)
MSLPlines = np.arange(900.,1060.,2.)  # values for MSLP contours
barbSize = {'emptybarb': 0., 'width': 0.2, 'height': 0.3}
pc = ccrs.PlateCarree()
merc = ccrs.Mercator()
coasts = cfeature.COASTLINE.with_scale('50m')
countries = cfeature.BORDERS.with_scale('50m')
states = cfeature.STATES.with_scale('50m')
lat = Usubset[Ucoords[0]].values
lon = Usubset[Ucoords[1]].values
lon[lon>180] -= 360.  # convert western longitudes to negative values
for t in range(len(timeList)):
    dt = datetime.strptime(str(timeList[t])[0:13],'%Y-%m-%dT%H')
    diff = dt - InitTime
    forecastHour = int(diff.days*24 + diff.seconds/(60*60))
    PNGname = 'GFS_%d%dRHwind_MSLP_%s_f%03d.png' % (pLevel1,pLevel2,sys.argv[1],forecastHour)  # output map name
    if os.path.exists(PNGname) is True:
        continue  # skip this time if the output map PNG file already exists!
    print(PNGname)
    fig, ax = plt.subplots(figsize=(12,7.5), subplot_kw={'projection': merc})
    ax.set_extent(domain, pc)
    ax.add_feature(coasts, linewidth=1., edgecolor='k', facecolor='none', zorder=5)
    ax.add_feature(countries, linewidth=1., edgecolor='k', facecolor='none', zorder=5)
    ax.add_feature(states, linewidth=1., edgecolor='k', facecolor='none', zorder=5)
    im = ax.contourf(lon, lat, RHmean[t,:,:], levels, cmap=RHcmap, extend='both', transform=pc)
    coords = {MSLPcoords[3]: timeList[t]}
    cl = ax.contour(lon, lat, MSLPsubset.sel(**coords)/100., MSLPlines, colors='k', linewidths=1, transform=pc)
    ax.clabel(cl, cl.levels[::2], inline=True, fmt='%d', fontsize=8)
    ax.barbs(lon[::8], lat[::8], Umean[t,::8,::8], Vmean[t,::8,::8], pivot='tip', zorder=50, color='k', sizes=barbSize,
             lw=0.5, length=5, transform=pc)
    gl = ax.gridlines(crs=pc, draw_labels=True, x_inline=False, linewidth=0.25)
    gl.top_labels, gl.right_labels, gl.rotate_labels = [False] * 3
    gl.xlocator = mticker.FixedLocator(np.arange(-180., 180., 10.))
    gl.ylocator = mticker.FixedLocator(np.arange(-90., 90., 5.))
    gl.xlabel_style = {'size': 11}
    gl.ylabel_style = {'size': 11}
    ax.set_title('forecast hour %03d    [init: %s]    valid at %s'
                 % (forecastHour,InitTime.strftime('%HZ %d %b %Y'),dt.strftime('%HZ %d %b %Y')), loc='left')
    ax.set_title('%d-%d-hPa wind [kt]\nMSLP [hPa]' % (pLevel1,pLevel2), loc='right', weight='bold')
    ax.spines['geo'].set_zorder(100)
    plt.tight_layout()
    addcolorbar(fig, ax, im, levels, pLevel1, pLevel2)
    # plt.show()  # uncomment to display the figure
    plt.savefig(PNGname, dpi=120)
    plt.close()

ds.close()
