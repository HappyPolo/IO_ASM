def anomalies(ds):
    '''
    # Compute anomalies by removing the time-mean.
    '''
    da = ds - ds.mean(dim='time')
    da.attrs = ds.attrs
    return da


# def print(message):
#     if debug:
#         print(message)


def filplonlat(ds):
    # To facilitate data subsetting
    # print(da.attrs)
    '''
    print(
        f'\n\nBefore flip, lon range is [{ds["lon"].min().data}, {ds["lon"].max().data}].'
    )
    ds["lon"] = ((ds["lon"] + 180) % 360) - 180
    # Sort lons, so that subset operations end up being simpler.
    ds = ds.sortby("lon")
    '''
    # print(
    #     f'\n\nAfter flip, lon range is [{ds["lon"].min().data}, {ds["lon"].max().data}].'
    # )
    # To facilitate data subsetting

    ds = ds.sortby("lat", ascending=True)
    # print(ds.attrs)
    print('\n\nAfter sorting lat values, ds["lat"] is:')
    print(ds["lat"])
    return ds


def weightslat(ds):
    deg2rad = np.pi/180.
    clat = ds['lat'].astype(np.float64)
    clat = np.sqrt(np.cos(deg2rad * clat))
    print('\n\nclat:\n')
    print(clat)
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
