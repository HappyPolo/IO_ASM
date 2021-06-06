#!/usr/bin/env python
# coding: utf-8
# note:
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
from svd_defs import *
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


def draw_ax2(ax3, fig, font, i, proj, re, MON):
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
sdir = '../../data/OISST/sst.mon.mean.nc'
pdir = '../../data/NCEP/precip.mon.mean.1x1.nc'
udir = '../../data/NCEP/uwnd.mon.mean.1x1.nc'
vdir = '../../data/NCEP/vwnd.mon.mean.1x1.nc'
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
SST = readData('sst', sdir)
PRC = readData('precip', pdir)
U = readData('uwnd', udir)
V = readData('vwnd', vdir)

SST = selYear(SST, 1982, 2020)
PRC = selYear(PRC, 1982, 2020)
U = selYear(U, 1982, 2020)
V = selYear(V, 1982, 2020)
# %%
# note:BIO,MAY

sst = selMon(SST, 5)
prc = selMon(PRC, 5)
u = selMon(U, 5)
v = selMon(V, 5)
# %%
prc
# %%
Rp = prc.sel(lat=slice(RlatS, RlatN), lon=slice(RlonL, RlonR))
x = Rp.time
I = 3
L = sst.sel(lat=slice(LlatS[I], LlatN[I]), lon=slice(LlonL[I], LlonR[I]))
L['time'] = Rp['time']
start = time.time()
le, re, lt, rt, Frac, lphet, rphet, Corr = SVD(L, Rp, N)
end = time.time()
print(end-start)

# %%
re = re.where(rphet <= 0.1)
le = le.where(lphet <= 0.1)
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
    draw_ax2(ax3, fig, font, i, proj, re, MON)
    draw_ax3(ax3, lt, rt, i, font, fig, x)
    fig.savefig('./plt/MCA/BIO_MCA_MAY_'+str(i) +
                '_.png', dpi=600, facecolor='w', bbox_inches='tight', pad_inches=0.0, format='png')

# %%
# note:BIO,APR
sst = selMon(SST, 4)
prc = selMon(PRC, 5)
# u = selMon(U, 5)
# v = selMon(V, 5)
Rp = prc.sel(lat=slice(RlatS, RlatN), lon=slice(RlonL, RlonR))
x = Rp.time
I = 3
L = sst.sel(lat=slice(LlatS[I], LlatN[I]), lon=slice(LlonL[I], LlonR[I]))
L['time'] = Rp['time']
start = time.time()
le, re, lt, rt, Frac, lphet, rphet, Corr = SVD(L, Rp, N)
end = time.time()
print(end-start)
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
    draw_ax2(ax3, fig, font, i, proj, re, MON)
    draw_ax3(ax3, lt, rt, i, font, fig, x)
    fig.savefig('./plt/MCA/BIO_MCA_APR_'+str(i) +
                '_.png', dpi=600, facecolor='w', bbox_inches='tight', pad_inches=0.0, format='png')
# %%
# %%
# note:BIO,MAR
sst = selMon(SST, 4)
prc = selMon(PRC, 5)
# u = selMon(U, 5)
# v = selMon(V, 5)
Rp = prc.sel(lat=slice(RlatS, RlatN), lon=slice(RlonL, RlonR))
x = Rp.time
I = 3
L = sst.sel(lat=slice(LlatS[I], LlatN[I]), lon=slice(LlonL[I], LlonR[I]))
L['time'] = Rp['time']
start = time.time()
le, re, lt, rt, Frac, lphet, rphet, Corr = SVD(L, Rp, N)
end = time.time()
print(end-start)
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
    draw_ax2(ax3, fig, font, i, proj, re, MON)
    draw_ax3(ax3, lt, rt, i, font, fig, x)
    fig.savefig('./plt/MCA/BIO_MCA_MAR_'+str(i) +
                '_.png', dpi=600, facecolor='w', bbox_inches='tight', pad_inches=0.0, format='png')
# %%
