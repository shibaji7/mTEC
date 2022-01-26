"""analyzeTEC.py: Module is dedicated for data analysis"""

__author__ = "Chakraborty, S."
__copyright__ = ""
__credits__ = []
__license__ = "MIT"
__version__ = "1.0."
__maintainer__ = "Chakraborty, S."
__email__ = "shibaji7@vt.edu"
__status__ = "Research"

import os
import datetime as dt
import json
from loguru import logger
import numpy as np
import aacgmv2
import pandas as pd
from multiprocessing import Pool
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter

import sys
sys.path.extend(["py/"])

from plotMap import MapPlot
from downloadTEC import ReadTEC
import plotTEC

def sg_filter(d, window_length, polyorder=None):
    polyorder = 2 if polyorder is None else polyorder
    ds = savgol_filter(d, window_length, polyorder)
    return ds

def smooth(x, window_len=5, window="hanning"):
    if x.ndim != 1: raise ValueError("smooth only accepts 1 dimension arrays.")
    if x.size < window_len: raise ValueError("Input vector needs to be bigger than window size.")
    if window_len<3: return x
    if not window in ["flat", "hanning", "hamming", "bartlett", "blackman"]: 
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")
    s = np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    if window == "flat": w = numpy.ones(window_len,"d")
    else: w = eval("np."+window+"(window_len)")
    y = np.convolve(w/w.sum(),s,mode="valid")
    d = window_len - 1
    y = y[int(d/2):-int(d/2)]
    return y

def g2m(row):
    mlat, mlon, mlt = aacgmv2.get_aacgm_coord(row["GDLAT"], row["GLON"], 350, row["DATE"])
    row["MLAT"], row["MLON"], row["MLT"] = np.round(mlat, 0), np.round(mlon, 0), np.round(mlt, 2)
    row["MLAT"], row["MLON"] = 2.*np.round(row["MLAT"]/2, 0), 2.*np.round(row["MLON"]/2, 0)
    return row

def sudo_run_DT_filter(po):
    dx, L, W, time_windows, dur, lat, lon = po["dx"], po["L"], po["W"],\
                po["time_windows"], po["dur"], po["lat"], po["lon"]
    dE = pd.DataFrame()
    L = L if np.mod(L,2) == 1 else L-1
    for tw in time_windows:
        ox = pd.DataFrame()
        ST = tw[0].hour*3600 + tw[0].minute*60 + tw[0].second
        xnew = [ST + (i*dur*60) for i in range(W)]
        ox["DATE"] = [tw[0] + dt.timedelta(minutes=i*dur) for i in range(W)]
        ox["GDLAT"], ox["GLON"], ox["GDALT"], ox["iTEC"], ox["TEC"] = lat, lon, 350., np.nan, np.nan
        ox["bTEC"], ox["dTEC"] = np.nan, np.nan
        o = dx[(dx.DATE>=tw[0]) & (dx.DATE<tw[1])]
        dates = o.DATE.apply(lambda x: x-dt.timedelta(minutes=dur/2)).tolist()
        ox.TEC[ox.DATE.isin(dates)] = o.TEC.tolist()
        if len(o)>=0.5*W: 
            x, y = np.array(o.DATE.apply(lambda t: t.hour*3600 + t.minute*60 + t.second)),\
                    np.array(o.TEC)
            x, y = x[~np.isnan(y)], y[~np.isnan(y)]
            ox["iTEC"] = interp1d(x, y, kind="linear", bounds_error=False, fill_value="extrapolate")(xnew)
        dE = pd.concat([dE, ox])
    if len(dE) > 0:
        dE.bTEC = sg_filter(dE.iTEC, L)
        dE.dTEC = smooth(np.array(dE.iTEC - dE.bTEC), 9)
    return dE

class AnalyzeTEC(object):
    """
    Filter and analyze data for median-filtering
    """
    
    def __init__(self, dates, frame):
        self.frame = frame
        with open("config/params.json", "r") as f: o = json.load(f)
        for k in o.keys():
            setattr(self, k, o[k])
        self.dates = [dates[0]-dt.timedelta(hours=self.end_hour_padding), 
                      dates[1]+dt.timedelta(hours=self.end_hour_padding)]
        self.mp = MapPlot(self.hemi, self.coords, self.dur)
        self.mfilter = MedianFilter(thresh=self.thresh)
        self.filter_grid()
        return
    
    def filter_grid(self):
        logger.info(f" Start gridding...")
        self.frame = self.frame[(self.frame.DATE>=self.dates[0]) & 
                                (self.frame.DATE<=self.dates[1])]
        self.frame = self.frame[(self.frame.GDLAT>=self.bottom_bound_lat) & 
                                (self.frame.GDLAT<=self.top_bound_lat)]
        self.frame = self.frame[(self.frame.GLON>=self.left_bound_lon) & 
                                (self.frame.GLON<=self.right_bound_lon)]
        logger.info(f" n.Unique Time(%d)"%(len(np.unique(self.frame.DATE))))
        logger.info(f" End gridding.")
        if "aacgmv2" in self.coords: 
            logger.info(f" Convert GEO to MAG...")
            self.frame = self.frame.apply(g2m, axis=1)
        self.detreand_by_coordinate()
        return
    
    
    def detreand_by_coordinate(self):
        logger.info(f" Detreand by: {self.coords} Coords.")
        lats, lons = np.unique(self.frame.GDLAT), np.unique(self.frame.GLON)
        # Set basic params
        L = int(self.ma_time_window/self.dur)
        W = int(self.hour_window*3600/(self.dur*60))
        logger.info(f" Parameter set: W({W}), L({L})")
        self.H = pd.DataFrame()
        logger.info(f" Total loops: N={len(lats)*len(lons)}")
        self.time_length = np.rint((self.dates[1]-self.dates[0]).total_seconds()/(self.hour_window*3600.))
        self.time_windows = [(self.dates[0]+dt.timedelta(hours=h*self.hour_window),
                              self.dates[0]+dt.timedelta(hours=(h+1)*self.hour_window)) 
                             for h in range(int(self.time_length-1))]
        logger.info(f" Total time window length {self.time_length}")
        proc_objs = []
        for lat in lats:
            for lon in lons:
                dx = self.frame[(self.frame.GDLAT == lat) & (self.frame.GLON == lon)]
                proc_objs.append({"dx": dx, "L": L, "W": W, "dur": self.dur, 
                                  "lat": lat, "lon": lon, "time_windows": self.time_windows})
        logger.info(f" Run parallel processes for {len(proc_objs)}")
        with Pool(16) as p:
            for o in p.map(sudo_run_DT_filter, proc_objs):
                #logger.info(f" Return length {len(o)}")
                self.H = pd.concat([self.H, o])
        self.plot_timeseries()
        return
    
    def doFilter(self):
        logger.info(f" Start m.filtering...")
        self.proc_data = {"S1": {}, "S2": {}}
        raw_data = {}
        nLen = len(self.frame.DATE.unique())
        scan_list = []
        for i in range(1, nLen-1):
            scans = []
            dn = dt.datetime.utcfromtimestamp(self.frame.DATE.unique().tolist()[i]/1e9)
            logger.info(f" m.Filter on {dn}")
            for j in range(i-1,i+2):
                dx = dt.datetime.utcfromtimestamp(self.frame.DATE.unique().tolist()[i]/1e9)
                if (dx in raw_data.keys()): z = raw_data[dx]
                else:
                    dat = self.frame[self.frame.DATE==dx]
                    _, _, z = self.to_grid(dat)
                    raw_data[dx] = z
                scans.append(z)
            scan_list.append(np.array(scans))
            if i==1: logger.info(f" Shape of input scans {np.array(scans).shape}")
        with Pool(8) as p: 
            for Z in p.map(self.mfilter.dofilter, scan_list):
                logger.info(f" Shape of output scan {Z.shape}")
                self.proc_data["S1"][dn] = Z
        self.proc_data["S0"] = raw_data
        logger.info(f" End m.filtering.")
        return
    
    def plot_timeseries(self, lat=65., lon=-150.):
        logger.info(f" Plot TS for ({lat}, {lon})")
        txt = r"$\Lambda, \Phi=%d,%d$"%(lat,lon)
        o = self.H[(self.H.GDLAT==lat) & (self.H.GLON==lon)]
        T = o.DATE.tolist()
        Ys = {
            "left":{
                "ylabel": r"TEC [TECu]",
                "ylim": [0,30],
                "entry": {
                    "TEC": {
                        "v": o.TEC.tolist(),
                        "label": r"$TEC^{CW}$",
                        "color": "r",
                        "mk": "o",
                        "ms": 0.5,
                        "alpha": 0.9
                    },
                    "iTEC": {
                        "v": o.iTEC.tolist(),
                        "label": r"$TEC^i$",
                        "color": "b",
                        "mk": "o",
                        "ms": 0.5,
                        "alpha": 0.5
                    },
                    "bTEC": {
                        "v": o.bTEC.tolist(),
                        "label": r"$TEC^b$",
                        "color": "g",
                        "mk": "o",
                        "ms": 0.5,
                        "alpha": 0.5
                    }
                }
            },
            "right": {
                "ylabel": r"$\Delta$TEC [TECu]",
                "ylim": [-1,1],
                "entry": {
                    "dTEC": {
                        "v": o.dTEC.tolist(),
                        "label": r"$TEC^d$",
                        "color": "k",
                        "mk": "o",
                        "ms": 0.5,
                        "alpha": 1.
                    }
                }
            }
        }
        plotTEC.plot_timeseries(T, Ys, txt=txt, fname="tmp/figures/out.png")
        return
    
    def generate_grid_map(self, date_index, key="iTEC"):
        xparam, yparam = "GDLAT", "GLON"
        if "aacgmv2" in self.coords: xparam, yparam = "MLAT", "MLON"
        os.makedirs("tmp/figures/", exist_ok=True)
        dns, Z = [], []
        for di in range(date_index-1,date_index+2):
            dn = dt.datetime.utcfromtimestamp(self.H.DATE.unique().tolist()[di]/1e9)
            o = self.H[self.H.DATE==dn]
            X, Y, z = self.to_grid(o, zparam=key)
            Z.append(z)
        dn = dt.datetime.utcfromtimestamp(self.H.DATE.unique().tolist()[date_index]/1e9)
        dn = dn + dt.timedelta(minutes=self.dur/2)
        self.mp.ini_figure(dn)
        Z = self.mfilter.dofilter(np.array(Z))
        lim = True
        if key == "dTEC": lim=False
        self.mp.set_frame(X, Y, Z, lim=lim)
        dn = dn - dt.timedelta(minutes=self.dur/2)
        map_fig_file = self.map_fig_file.format(Y=dn.year, m=dn.strftime("%b").lower(), 
                                                d="%02d"%dn.day, H="%02d"%dn.hour, 
                                                M="%02d"%dn.minute, T=key)
        self.mp.ax.text(0.02, 0.8, r"$\tau=%.1f$"%self.thresh, ha="left", va="center",
                        transform=self.mp.ax.transAxes, fontsize="small",)
        self.mp.save(map_fig_file)
        return
    
    def to_grid(self, q, xparam="GDLAT", yparam="GLON", zparam="TEC"):
        """
        Method converts frame to "GDLAT" and "GLON" or gate
        """
        plotParamDF = q[ [xparam, yparam, zparam] ]
        plotParamDF[xparam] = plotParamDF[xparam].tolist()
        plotParamDF[yparam] = plotParamDF[yparam].tolist()
        plotParamDF = plotParamDF.groupby( [xparam, yparam] ).mean().reset_index()
        plotParamDF = plotParamDF[ [xparam, yparam, zparam] ].pivot( xparam, yparam )
        x = plotParamDF.index.values
        y = plotParamDF.columns.levels[1].values
        X, Y  = np.meshgrid( x, y )
        # Mask the nan values! pcolormesh can't handle them well!
        Z = np.ma.masked_where(
                np.isnan(plotParamDF[zparam].values),
                plotParamDF[zparam].values)
        return X,Y,Z
    
class MedianFilter(object):
    """
    Filter median-filtering fo each time instance
    """
    
    def __init__(self, thresh=0.1, kernel=None, time_ins=3):
        if kernel is None: kernel = np.array([[[1,2,1],[2,3,2],[1,2,1]],
                                              [[2,3,2],[3,5,3],[2,3,2]],
                                              [[1,2,1],[1,2,1],[1,2,1]]])
        self.thresh = thresh
        self.kernel = kernel
        self.time_ins = time_ins
        self.mIndex = int(time_ins/2)
        return
    
    def dofilter(self, scans):
        mscan = np.zeros((scans.shape[1], scans.shape[2]))*np.nan
        for _i in range(self.mIndex,mscan.shape[0]-self.mIndex):
            for _j in range(self.mIndex,mscan.shape[1]-self.mIndex):
                box = scans[:, _i-self.mIndex:_i+self.mIndex+1, _j-self.mIndex:_j+self.mIndex+1]
                box = np.ma.masked_where(np.isnan(box), box)
                box_kernel = np.ma.masked_where(np.isnan(box), np.ones_like(box)) * self.kernel
                if np.ma.count(box) == 0: th = 0.
                else: th = np.ma.sum(box_kernel) / np.sum(self.kernel)
                if th >= self.thresh: mscan[_i, _j] = np.ma.average(box, weights=self.kernel)
        return mscan

def call_run_plot(dates, frame):
    atec = AnalyzeTEC(dates, frame)
    #atec.doFilter()
    # Plot for testing figures
    for i in range(10,80):
        atec.generate_grid_map(i)
        atec.generate_grid_map(i, "bTEC")
        atec.generate_grid_map(i, "dTEC")
        #break
    return