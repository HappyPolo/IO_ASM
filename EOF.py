#!/usr/bin/env python
# coding: utf-8
# %%
# // tag: 计算MAM印度洋海温的EOF
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker
import numpy as np
import xarray as xr
# import iris
from eofs.xarray import Eof
from xMCA import xMCA
import matplotlib.pyplot as plt
import gc
import scipy.stats as stats
from scipy.signal import detrend
import time
# %%


def standardize(x):
    return (x - x.mean()) / x.std()


def SAM(v):
    #V850 - V200
    lon = v.lon
    lat = v.lat
    lon_range = lon[(lon >= 70.) & (lon <= 110.)]
    lat_range = lat[(lat >= 10.) & (lat <= 30.)]
    v850 = v.sel(level=850, lon=lon_range,
                 lat=lat_range).mean(dim=['lat', 'lon']).drop('level')
    v200 = v.sel(level=200, lon=lon_range,
                 lat=lat_range).mean(dim=['lat', 'lon']).drop('level')
    sam = standardize(v850-v200)
    return sam


def WY(u):
    # U850 - U200
    lon = u.lon
    lat = u.lat
    lon_range = lon[(lon >= 40.) & (lon <= 110.)]
    lat_range = lat[(lat >= 5.) & (lat <= 20.)]
    u850 = u.sel(level=850, lon=lon_range,
                 lat=lat_range).mean(dim=['lat', 'lon']).drop('level')
    u200 = u.sel(level=200, lon=lon_range,
                 lat=lat_range).mean(dim=['lat', 'lon']).drop('level')
    wy = standardize(u850-u200)
    return wy


def SEAM(u):
    lon = u.lon
    lat = u.lat
    lon_range1 = lon[(lon >= 90.) & (lon <= 130.)]
    lat_range1 = lat[(lat >= 5.) & (lat <= 15.)]
    lon_range2 = lon[(lon >= 110.) & (lon <= 140.)]
    lat_range2 = lat[(lat >= 22.) & (lat <= 33.)]
    u1 = u.sel(level=850, lon=lon_range1,
               lat=lat_range1).mean(dim=['lat', 'lon']).drop('level')
    u2 = u.sel(level=850, lon=lon_range2,
               lat=lat_range2).mean(dim=['lat', 'lon']).drop('level')
    seam = standardize(u1-u2)
    return seam


def EAM(u):
    lon = u.lon
    lat = u.lat
    lon_range1 = lon[(lon >= 110.) & (lon <= 150.)]
    lat_range1 = lat[(lat >= 25.) & (lat <= 35.)]
    lon_range2 = lon[(lon >= 110.) & (lon <= 150.)]
    lat_range2 = lat[(lat >= 40.) & (lat <= 50.)]
    u1 = u.sel(level=200, lon=lon_range1,
               lat=lat_range1).mean(dim=['lat', 'lon']).drop('level')
    u2 = u.sel(level=200, lon=lon_range2,
               lat=lat_range2).mean(dim=['lat', 'lon']).drop('level')
    eam = standardize(u1-u2)
    return eam


def month_to_season(xMon, season):
    """ This function takes an xarray dataset containing monthly data spanning years and
        returns a dataset with one sample per year, for a specified three-month season.

        Time stamps are centered on the season, e.g. seasons='DJF' returns January timestamps.

        If a calculated season's timestamp falls outside the original range of monthly values, then the calculated mean
        is dropped.  For example, if the monthly data's time range is [Jan-2000, Dec-2003] and the season is "DJF", the
        seasonal mean computed from the single month of Dec-2003 is dropped.
    """
    startDate = xMon.time[0]
    endDate = xMon.time[-1]
    seasons_pd = {
        'DJF': ('QS-DEC', 1),
        'JFM': ('QS-JAN', 2),
        'FMA': ('QS-FEB', 3),
        'MAM': ('QS-MAR', 4),
        'AMJ': ('QS-APR', 5),
        'MJJ': ('QS-MAY', 6),
        'JJA': ('QS-JUN', 7),
        'JAS': ('QS-JUL', 8),
        'ASO': ('QS-AUG', 9),
        'SON': ('QS-SEP', 10),
        'OND': ('QS-OCT', 11),
        'NDJ': ('QS-NOV', 12)
    }
    try:
        (season_pd, season_sel) = seasons_pd[season]
    except KeyError:
        raise ValueError("contributed: month_to_season: bad season: SEASON = " +
                         season)

    # Compute the three-month means, moving time labels ahead to the middle month.
    month_offset = 'MS'
    xSeasons = xMon.resample(time=season_pd, loffset=month_offset).mean()

    # Filter just the desired season, and trim to the desired time range.
    xSea = xSeasons.sel(time=xSeasons.time.dt.month == season_sel)
    xSea = xSea.sel(time=slice(startDate, endDate))
    return xSea


def anomalies(ds):
    '''
    # Compute anomalies by removing the time-mean.
    '''
    da = ds - ds.mean(dim='time')
    da.attrs = ds.attrs
    return da


def print_debug(message):
    if debug:
        print(message)


def filplonlat(ds):
    # To facilitate data subsetting
    # print(da.attrs)
    '''
    print_debug(
        f'\n\nBefore flip, lon range is [{ds["lon"].min().data}, {ds["lon"].max().data}].'
    )
    ds["lon"] = ((ds["lon"] + 180) % 360) - 180
    # Sort lons, so that subset operations end up being simpler.
    ds = ds.sortby("lon")
    '''
    # print_debug(
    #     f'\n\nAfter flip, lon range is [{ds["lon"].min().data}, {ds["lon"].max().data}].'
    # )
    # To facilitate data subsetting

    ds = ds.sortby("lat", ascending=True)
    # print(ds.attrs)
    print_debug('\n\nAfter sorting lat values, ds["lat"] is:')
    print_debug(ds["lat"])
    return ds


def weightslat(ds):
    deg2rad = np.pi/180.
    clat = ds['lat'].astype(np.float64)
    clat = np.sqrt(np.cos(deg2rad * clat))
    print_debug('\n\nclat:\n')
    print_debug(clat)
    # Xarray will apply lat-based weights to all lons and timesteps automatically.
    # This is called "broadcasting".
    wds = ds
    wds.attrs = ds.attrs
    ds = clat * ds
    wds.attrs['long_name'] = 'Wgt: '+wds.attrs['long_name']
    return wds


def readData(var, fdir):
    ds = xr.open_dataset(fdir)[var]
    dsa = anomalies(ds)
    wdsa = filplonlat(dsa)
    # wdsa = weightslat(dsa)
    return wdsa


def EOF(da, N):
    coslat = np.cos(np.deg2rad(da.coords['lat'].values)).clip(0., 1.)
    wgts = np.sqrt(coslat)[..., np.newaxis]
    solver = Eof(da, weights=wgts)
    eof = solver.eofsAsCovariance(neofs=N, pcscaling=1)
    pc = solver.pcs(npcs=N, pcscaling=1)
    pc = pc.transpose('mode', 'time')
    return eof, pc


def test_pcs(pc, pa):
    fig, (ax1, ax2) = plt.subplots(2, 2, figsize=(8, 5))
    pc[0].plot(ax=ax1[0])
    pc[1].plot(ax=ax1[1])

    pa[0].plot(ax=ax2[0])
    pa[1].plot(ax=ax2[1])


def selYear(da, startYear, endYear):
    startDate = da.sel(time=da.time.dt.year == startYear).time[0]
    endDate = da.sel(time=da.time.dt.year == endYear).time[-1]
    da = da.sel(time=slice(startDate, endDate))
    return da


def selMon(da, Mon):
    return da.sel(time=da.time.dt.month == Mon)


# %%
sdir = './data/ERA5/ERA5.sst.mon.1x1.nc'
pdir = './data/ERA5/ERA5.prc.mon.1x1.nc'
udir = './data/ERA5/ERA5.u.mon.1x1.nc'
vdir = './data/ERA5/ERA5.v.mon.1x1.nc'

N = 3
LlatS = [0., -15., -45., -45.,  0.,   0.]
LlatN = [30.,  15.,   0.,  30., 30.,  30.]
LlonL = [50.,  50.,  50.,  50., 50.,  78.]
LlonR = [110., 110., 110., 110., 78., 100.]
Llatlon = [LlatS, LlatN, LlonL, LlonR]
RlatS, RlatN, RlonL, RlonR = -30., 60., 30., 180.
Rlatlon = [RlatS, RlatN, RlonL, RlonR]
region = ['NIO', 'SIO', 'TIO', 'BIO']

month = ['MAM', 'APR', 'MAY', 'JUN']

debug = False
# %%
start = time.time()
SST = readData('sst', sdir)
PRC = readData('tp', pdir)
U = readData('u', udir).sel(level=850)
V = readData('v', vdir).sel(level=850)
end = time.time()
print(end-start)
# %%
SST = selYear(SST, 1982, 2020)
PRC = selYear(PRC, 1982, 2020)
U = selYear(U, 1982, 2020)
V = selYear(V, 1982, 2020)

# %%
# note:BIO,MAY

SST = selMon(SST, 5)
SST = weightslat(SST)
sst = SST.sel(lat=slice(-45, 30), lon=slice(50, 110))
# %%
sst
# %%
eof, pc = EOF(sst, N)
# %%

# %%
