#!/usr/bin/env python
# coding: utf-8
# %%
import xarray as xr
from eofs.xarray import Eof
from xMCA import xMCA
import numpy as np
import scipy.stats as stats
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cmaps
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.ticker as mticker
import time
# %%


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


def SVD(L, R, N):
    #   R = r.sel(lat=slice(nr[0]-1.,nr[1]+1.),lon=slice(nr[2]-1.,nr[3]+1.))
    #   L = l.sel(lat=slice(nl[n][0]-1.,nl[n][1]+1.),lon=slice(nl[n][3]-1.,nl[n][3]+1.))
    sst_ts = xMCA(L, R)
    sst_ts.solver()
    lp, rp = sst_ts.patterns(n=N)
    lt, rt = sst_ts.expansionCoefs(n=N)
    le, re, lphet, rphet = sst_ts.heterogeneousPatterns(
        n=N, statistical_test=True)
    frac = sst_ts.covFracs(n=N)
    Frac = frac * 100.
    Corr = xr.corr(lt, rt, dim='time')
    return le, re, lt, rt, Frac, lphet, rphet, Corr


def readData(var, fdir):
    ds = xr.open_dataset(fdir)[var]
    dsa = anomalies(ds)
    dsa = filplonlat(dsa)
    wdsa = weightslat(dsa)
    return wdsa


def test_pn(i, le, ule, vle):
    fig, ax1 = plt.subplots(3, figsize=(3, 5))
    le[i].plot(ax=ax1[0], cmap='bwr')
    ule[i].plot(ax=ax1[1], cmap='bwr')
    vle[i].plot(ax=ax1[2], cmap='bwr')


def selYear(da, startYear, endYear):
    startDate = da.sel(time=da.time.dt.year == startYear).time[0]
    endDate = da.sel(time=da.time.dt.year == endYear).time[-1]
    da = da.sel(time=slice(startDate, endDate))
    return da


def selMon(da, Mon):
    return da.sel(time=da.time.dt.month == Mon)


# %%
# bug:NCEP的数据，这里直接用python对时间切片做SVD会出现时间坐标不一致的情况，暂时还没解决
# sdir = './data/sst.mon.mean.nc'
# pdir = './data/precip.mon.mean.1x1.nc'
# udir = './data/uwnd.mon.mean.1x1.nc'
# vdir = './data/vwnd.mon.mean.1x1.nc'
# 暂时先使用ERA5的数据
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
# SST = readData('sst', sdir)
# PRC = readData('precip', pdir)
# U = readData('uwnd', udir)
# V = readData('vwnd', vdir)
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


# %%
# note:BIO,MAY

sst = selMon(SST, 5)
prc = selMon(PRC, 5)
u = selMon(U, 5)
v = selMon(V, 5)
# %%
Rp = prc.sel(lat=slice(RlatS, RlatN), lon=slice(RlonL, RlonR))
Ru = u.sel(lat=slice(RlatS, RlatN), lon=slice(RlonL, RlonR))
Rv = v.sel(lat=slice(RlatS, RlatN), lon=slice(RlonL, RlonR))
x = Rp.time


I = 3
L = sst.sel(lat=slice(LlatS[I], LlatN[I]), lon=slice(LlonL[I], LlonR[I]))
start = time.time()
le, re, lt, rt, Frac, lphet, rphet, Corr = SVD(L, Rp, N)
end = time.time()
print(end-start)
ule, ure, ult, urt, ulphet, Fracu, rphetu, Corru = SVD(L, Ru, N)
end = time.time()
print(end-start)
vle, vre, vlt, vrt, vlphet, Fracv, rphetv, Corrv = SVD(L, Rv, N)
end = time.time()
print(end-start)
ure = ure.where(rphetu <= 0.1)
vre = vre.where(rphetv <= 0.1)
# %%
test_pn(0, le, ule, vle)
test_pn(1, le, ule, vle)
test_pn(2, le, ule, vle)
# %%
le[0], re[0], lt[0], rt[0] = -le[0], -re[0], -lt[0], -rt[0]
ule[0], ure[0], ult[0], urt[0] = -ule[0], -ure[0], -ult[0], -urt[0]
vle[0], vre[0], vlt[0], vrt[0] = -vle[0], -vre[0], -vlt[0], -vrt[0]
# le[1], re[1], lt[1], rt[1] = -le[1], -re[1], -lt[1], -rt[1]
ule[1], ure[1], ult[1], urt[1] = -ule[1], -ure[1], -ult[1], -urt[1]
vle[1], vre[1], vlt[1], vrt[1] = -vle[1], -vre[1], -vlt[1], -vrt[1]
# le[2], re[2], lt[2], rt[2] = -le[2], -re[2], -lt[2], -rt[2]
ule[2], ure[2], ult[2], urt[2] = -ule[2], -ure[2], -ult[2], -urt[2]
vle[2], vre[2], vlt[2], vrt[2] = -vle[2], -vre[2], -vlt[2], -vrt[2]
test_pn(0, le, ule, vle)
test_pn(1, le, ule, vle)
test_pn(2, le, ule, vle)
# %%
re = re.where(rphet <= 0.1)
le = le.where(lphet <= 0.1)


# %%
def draw_ax1(ax1, fig, font, i, proj, le, Frac, Corr, MON):
    lon = le.lon
    lat = le.lat
    im = ax1.contourf(
        lon, lat, le[i], transform=proj, extend='neither', cmap='RdBu_r', alpha=0.8, levels=np.arange(-1., 1.1, 0.2), zorder=0
    )
    cbposition = fig.add_axes([0.45, 0.12, 0.015, 0.55])
    cb1 = fig.colorbar(im, cax=cbposition, orientation='vertical',
                       spacing='proportional', format='%.1f', extend='both')
    cb1.set_label('', fontdict=font)
    cb1.ax.tick_params(labelsize=6)
    cb1.ax.tick_params(which='major', direction='out',
                       labelsize=6, length=2)  # 主刻度设置

    ax1.add_feature(cfeature.LAND, edgecolor='black')
    ax1.add_feature(cfeature.LAKES, edgecolor='black')
    ax1.coastlines(color='k', linewidth=0.6)

    ax1.set_extent([50, 110, -45, 30], crs=proj)
    xticks = np.arange(50, 111, 20)
    yticks = np.arange(-45, 31, 15)

    ax1.set_xticks(xticks, crs=proj)
    ax1.set_yticks(yticks, crs=proj)
    ax1.xaxis.set_major_formatter(
        LongitudeFormatter(zero_direction_label=True))
    ax1.yaxis.set_major_formatter(LatitudeFormatter())
    ax1.tick_params(axis='both', which='major', labelsize=7, direction='out',
                    length=5, width=0.3, pad=0.2, top=True, right=True)
    ax1.minorticks_on()
    ax1.tick_params(axis='both', which='minor', direction='out',
                    width=0.3, top=True, right=True)
    ax1.set_title('(a) '+MON+' SST', fontdict=font, loc='left')


def draw_ax2(ax3, fig, font, i, proj, re, ure, vre, MON):
    xticks = np.arange(30, 181, 50)
    yticks = np.arange(-30, 61, 30)

    ax2.set_xticks(xticks, crs=proj)
    ax2.set_yticks(yticks, crs=proj)
    ax2.xaxis.set_major_formatter(
        LongitudeFormatter(zero_direction_label=True))
    ax2.yaxis.set_major_formatter(LatitudeFormatter())
    ax2.tick_params(axis='both', which='major', labelsize=7, direction='out',
                    length=5, width=0.3, pad=0.2, top=True, right=True)
    ax2.minorticks_on()
    ax2.tick_params(axis='both', which='minor', direction='out',
                    width=0.3, top=True, right=True)
    ax2.spines['geo'].set_linewidth(0.8)
    ax2.set_extent([30, 180, -30, 60], crs=proj)

    ax2.coastlines(color='k', linewidth=0.6)
    # ax2.gridlines(color='lightgrey', linestyle='-')
    ax2.add_feature(cfeature.LAND, edgecolor='black')
    ax2.add_feature(cfeature.LAKES, edgecolor='black')

    rm = ax2.contourf(re.lon, re.lat, re[i], transform=proj, extend='neither',
                      cmap='BrBG', alpha=0.8, levels=np.arange(-1., 1.1, 0.2), zorder=0)
    cbposition = fig.add_axes([0.85, 0.38, 0.015, 0.25])
    cb2 = fig.colorbar(rm, cax=cbposition, orientation='vertical',
                       spacing='proportional', format='%.1f', extend='both')
    cb2.ax.tick_params(labelsize=6)

    ax2.set_title('(b) '+MON+' BIO PRC', fontdict=font, loc='left')
    Rlon = re.lon
    Rlat = re.lat
    # Q = ax2.quiver(Rlon[::4], Rlat[::4], ure[i, ::4, ::4], vre[i, ::4, ::4], transform=proj, zorder=2,
    #                units='width', width=0.0035, scale=0.1, scale_units='xy')
    # ax1.quiverkey(Q, 0.9, -0.17, 1, r'$1 m/s$',
    #               labelpos='E', fontproperties=font)


def draw_ax3(ax3, lt, rt, i, font, fig, x):
    ax3.plot(x, lt[i], ls='-', c='b', label='sst')
    ax3.plot(x, rt[i], ls='-', c='r', label='prc')
    # ax3.plot(x,urt[i],ls='-',c='c',label='uwnd')
    # ax3.plot(x,vrt[i], ls='-', c='m', label='vwnd')
    ax3.set_title('(c)', fontdict=font, loc='left')
    ax3.set_ylim(-3., 3.)
    ax3.legend(loc='upper left', frameon=False, fontsize=5, ncol=4)
    ax3.tick_params(labelsize=7)
    labels = ax3.get_xticklabels() + ax3.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]


# %%
font = {'family': 'Times New Roman',
        'weight': 'normal',
        'size': 6,
        }
proj = ccrs.PlateCarree()
MON = 'MAY'
for i in range(N):
    fig = plt.figure()
    gs = fig.add_gridspec(14, 13, wspace=2, hspace=.1)
    ax1 = fig.add_subplot(gs[4:14, 0:6], projection=proj)
    ax2 = fig.add_subplot(gs[0:14, 7:12], projection=proj)
    ax3 = fig.add_subplot(gs[11:14, 7:13])
    draw_ax1(ax1, fig, font, i, proj, le, Frac, Corr, MON)
    draw_ax2(ax3, fig, font, i, proj, re, ure, vre, MON)
    draw_ax3(ax3, lt, rt, i, font, fig, x)
    fig.savefig('./plt/MCA/BIO_MCA_MAY_'+str(i) +
                '_.png', dpi=600, facecolor='w', bbox_inches='tight', pad_inches=0.0, format='png')
# %%

# %%
# note:BIO,APR
mon = 4
sst = selMon(SST, mon)
prc = selMon(PRC, mon)
u = selMon(U, mon)
v = selMon(V, mon)

Rp = prc.sel(lat=slice(RlatS, RlatN), lon=slice(RlonL, RlonR))
Ru = u.sel(lat=slice(RlatS, RlatN), lon=slice(RlonL, RlonR))
Rv = v.sel(lat=slice(RlatS, RlatN), lon=slice(RlonL, RlonR))
x = Rp.time


I = 3
L = sst.sel(lat=slice(LlatS[I], LlatN[I]), lon=slice(LlonL[I], LlonR[I]))
le, re, lt, rt, Frac, lphet, rphet, Corr = SVD(L, Rp, N)
ule, ure, ult, urt, ulphet, Fracu, rphetu, Corru = SVD(L, Ru, N)
vle, vre, vlt, vrt, vlphet, Fracv, rphetv, Corrv = SVD(L, Rv, N)
ure = ure.where(rphetu <= 0.1)
vre = vre.where(rphetv <= 0.1)
# %%
test_pn(0, le, ule, vle)
test_pn(1, le, ule, vle)
test_pn(2, le, ule, vle)
# %%
le[0], re[0], lt[0], rt[0] = -le[0], -re[0], -lt[0], -rt[0]
# ule[0], ure[0], ult[0], urt[0] = -ule[0], -ure[0], -ult[0], -urt[0]
vle[0], vre[0], vlt[0], vrt[0] = -vle[0], -vre[0], -vlt[0], -vrt[0]
# le[1], re[1], lt[1], rt[1] = -le[1], -re[1], -lt[1], -rt[1]
ule[1], ure[1], ult[1], urt[1] = -ule[1], -ure[1], -ult[1], -urt[1]
vle[1], vre[1], vlt[1], vrt[1] = -vle[1], -vre[1], -vlt[1], -vrt[1]
le[2], re[2], lt[2], rt[2] = -le[2], -re[2], -lt[2], -rt[2]
# ule[2], ure[2], ult[2], urt[2] = -ule[2], -ure[2], -ult[2], -urt[2]
# vle[2], vre[2], vlt[2], vrt[2] = -vle[2], -vre[2], -vlt[2], -vrt[2]
test_pn(0, le, ule, vle)
test_pn(1, le, ule, vle)
test_pn(2, le, ule, vle)
# %%
re = re.where(rphet <= 0.1)
le = le.where(lphet <= 0.1)
font = {'family': 'Times New Roman',
        'weight': 'normal',
        'size': 6,
        }
proj = ccrs.PlateCarree()
MON = 'APR'
for i in range(N):
    fig = plt.figure()
    gs = fig.add_gridspec(14, 13, wspace=2, hspace=.1)
    ax1 = fig.add_subplot(gs[4:14, 0:6], projection=proj)
    ax2 = fig.add_subplot(gs[0:14, 7:12], projection=proj)
    ax3 = fig.add_subplot(gs[11:14, 7:13])
    draw_ax1(ax1, fig, font, i, proj, le, Frac, Corr, MON)
    draw_ax2(ax3, fig, font, i, proj, re, ure, vre, MON)
    draw_ax3(ax3, lt, rt, i, font, fig, x)
    fig.savefig('./plt/MCA/BIO_MCA_APR_'+str(i) +
                '_.png', dpi=600, facecolor='w', bbox_inches='tight', pad_inches=0.0, format='png')
# %%
# %%
# note:BIO,MAR
mon = 3
sst = selMon(SST, mon)
prc = selMon(PRC, mon)
u = selMon(U, mon)
v = selMon(V, mon)

Rp = prc.sel(lat=slice(RlatS, RlatN), lon=slice(RlonL, RlonR))
Ru = u.sel(lat=slice(RlatS, RlatN), lon=slice(RlonL, RlonR))
Rv = v.sel(lat=slice(RlatS, RlatN), lon=slice(RlonL, RlonR))
x = Rp.time


I = 3
L = sst.sel(lat=slice(LlatS[I], LlatN[I]), lon=slice(LlonL[I], LlonR[I]))
le, re, lt, rt, Frac, lphet, rphet, Corr = SVD(L, Rp, N)
ule, ure, ult, urt, ulphet, Fracu, rphetu, Corru = SVD(L, Ru, N)
vle, vre, vlt, vrt, vlphet, Fracv, rphetv, Corrv = SVD(L, Rv, N)
ure = ure.where(rphetu <= 0.1)
vre = vre.where(rphetv <= 0.1)
# %%
test_pn(0, le, ule, vle)
test_pn(1, le, ule, vle)
test_pn(2, le, ule, vle)
# %%
# le[0], re[0], lt[0], rt[0] = -le[0], -re[0], -lt[0], -rt[0]
# # ule[0], ure[0], ult[0], urt[0] = -ule[0], -ure[0], -ult[0], -urt[0]
# vle[0], vre[0], vlt[0], vrt[0] = -vle[0], -vre[0], -vlt[0], -vrt[0]
# # le[1], re[1], lt[1], rt[1] = -le[1], -re[1], -lt[1], -rt[1]
# ule[1], ure[1], ult[1], urt[1] = -ule[1], -ure[1], -ult[1], -urt[1]
# vle[1], vre[1], vlt[1], vrt[1] = -vle[1], -vre[1], -vlt[1], -vrt[1]
le[2], re[2], lt[2], rt[2] = -le[2], -re[2], -lt[2], -rt[2]
# ule[2], ure[2], ult[2], urt[2] = -ule[2], -ure[2], -ult[2], -urt[2]
# vle[2], vre[2], vlt[2], vrt[2] = -vle[2], -vre[2], -vlt[2], -vrt[2]

# %%
re = re.where(rphet <= 0.1)
le = le.where(lphet <= 0.1)
font = {'family': 'Times New Roman',
        'weight': 'normal',
        'size': 6,
        }
proj = ccrs.PlateCarree()
MON = 'MAR'
for i in range(N):
    fig = plt.figure()
    gs = fig.add_gridspec(14, 13, wspace=2, hspace=.1)
    ax1 = fig.add_subplot(gs[4:14, 0:6], projection=proj)
    ax2 = fig.add_subplot(gs[0:14, 7:12], projection=proj)
    ax3 = fig.add_subplot(gs[11:14, 7:13])
    draw_ax1(ax1, fig, font, i, proj, le, Frac, Corr, MON)
    draw_ax2(ax3, fig, font, i, proj, re, ure, vre, MON)
    draw_ax3(ax3, lt, rt, i, font, fig, x)
    fig.savefig('./plt/MCA/BIO_MCA_MAR_'+str(i) +
                '_.png', dpi=600, facecolor='w', bbox_inches='tight', pad_inches=0.0, format='png')
# %%
# %%
# note:BIO,MAR
mon = 6
sst = selMon(SST, mon)
prc = selMon(PRC, mon)
u = selMon(U, mon)
v = selMon(V, mon)

Rp = prc.sel(lat=slice(RlatS, RlatN), lon=slice(RlonL, RlonR))
Ru = u.sel(lat=slice(RlatS, RlatN), lon=slice(RlonL, RlonR))
Rv = v.sel(lat=slice(RlatS, RlatN), lon=slice(RlonL, RlonR))
x = Rp.time


I = 3
L = sst.sel(lat=slice(LlatS[I], LlatN[I]), lon=slice(LlonL[I], LlonR[I]))
le, re, lt, rt, Frac, lphet, rphet, Corr = SVD(L, Rp, N)
ule, ure, ult, urt, ulphet, Fracu, rphetu, Corru = SVD(L, Ru, N)
vle, vre, vlt, vrt, vlphet, Fracv, rphetv, Corrv = SVD(L, Rv, N)
ure = ure.where(rphetu <= 0.1)
vre = vre.where(rphetv <= 0.1)
# %%
test_pn(0, le, ule, vle)
test_pn(1, le, ule, vle)
test_pn(2, le, ule, vle)
# %%
le[0], re[0], lt[0], rt[0] = -le[0], -re[0], -lt[0], -rt[0]
ule[0], ure[0], ult[0], urt[0] = -ule[0], -ure[0], -ult[0], -urt[0]
vle[0], vre[0], vlt[0], vrt[0] = -vle[0], -vre[0], -vlt[0], -vrt[0]
# # le[1], re[1], lt[1], rt[1] = -le[1], -re[1], -lt[1], -rt[1]
# ule[1], ure[1], ult[1], urt[1] = -ule[1], -ure[1], -ult[1], -urt[1]
# vle[1], vre[1], vlt[1], vrt[1] = -vle[1], -vre[1], -vlt[1], -vrt[1]
le[2], re[2], lt[2], rt[2] = -le[2], -re[2], -lt[2], -rt[2]
# ule[2], ure[2], ult[2], urt[2] = -ule[2], -ure[2], -ult[2], -urt[2]
# vle[2], vre[2], vlt[2], vrt[2] = -vle[2], -vre[2], -vlt[2], -vrt[2]

# %%
re = re.where((rphet) <= 0.1)
le = le.where((lphet) <= 0.1)
font = {'family': 'Times New Roman',
        'weight': 'normal',
        'size': 6,
        }
proj = ccrs.PlateCarree()
MON = 'JUN'
for i in range(N):
    fig = plt.figure()
    gs = fig.add_gridspec(14, 13, wspace=2, hspace=.1)
    ax1 = fig.add_subplot(gs[4:14, 0:6], projection=proj)
    ax2 = fig.add_subplot(gs[0:14, 7:12], projection=proj)
    ax3 = fig.add_subplot(gs[11:14, 7:13])
    draw_ax1(ax1, fig, font, i, proj, le, Frac, Corr, MON)
    draw_ax2(ax3, fig, font, i, proj, re, ure, vre, MON)
    draw_ax3(ax3, lt, rt, i, font, fig, x)
    fig.savefig('./plt/MCA/BIO_MCA_JUN_'+str(i) +
                '_.png', dpi=600, facecolor='w', bbox_inches='tight', pad_inches=0.0, format='png')
# %%
