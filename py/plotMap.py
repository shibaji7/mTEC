#!/usr/bin/env python

"""plotMap.py: module to plot map plots."""

__author__ = "Chakraborty, S."
__copyright__ = ""
__credits__ = []
__license__ = "MIT"
__version__ = "1.0."
__maintainer__ = "Chakraborty, S."
__email__ = "shibaji7@vt.edu"
__status__ = "Research"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.style.use(["science", "ieee"])
import matplotlib as mpl
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import sys
sys.path.extend(["py/"])

import os
import datetime as dt
import argparse
from dateutil import parser as prs
import numpy as np
import pandas as pd
import aacgmv2

import cartopy.crs as ccrs
import cartopy
import sdcarto

import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

def convert_to_map_lat_lon(xs, ys, _from, _to):
    lat, lon = [], []
    for x, y in zip(xs, ys):
        _lon, _lat = _to.transform_point(x, y, _from)
        lat.append(_lat)
        lon.append(_lon)
    return lat, lon

class MapPlot(object):
    """
    Plot data from map(ex) files
    """
    
    def __init__(self, hemi="north", coords="aacgmv2_mlt", dur=5):
        self.hemi = hemi
        self.coords = coords
        self.dur = 5
        return
    
    def ini_figure(self, date):
        """
        Instatitate figure and axes labels
        """
        proj = cartopy.crs.NorthPolarStereo() if self.hemi == "north" else cartopy.crs.SouthPolarStereo()
        
        self.fig = plt.figure(dpi=300, figsize=(4,4))
        self.ax = self.fig.add_subplot(111, projection="sdcarto", map_projection = proj,
                                  coords=self.coords, plot_date=date)
        self.ax.overaly_coast_lakes(lw=0.4, alpha=0.9)
        self.ax.set_extent([-180, 180, 30, 90], crs=cartopy.crs.PlateCarree())
        plt_lons = np.arange( 0, 361, 15 )
        mark_lons = np.arange( 0, 360, 15 )
        plt_lats = np.arange(20,90,10)
        gl = self.ax.gridlines(crs=cartopy.crs.Geodetic(), linewidth=0.5)
        gl.xlocator = mticker.FixedLocator(plt_lons)
        gl.ylocator = mticker.FixedLocator(plt_lats)
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        gl.n_steps = 90
        self.ax.mark_latitudes(plt_lats, fontsize="small", color="k")
        self.ax.mark_longitudes(plt_lons, fontsize="small", color="k")
        self.ax.text(0.01, 1.05, self.date_string(date), ha="left", va="center", 
                     transform=self.ax.transAxes, fontsize="small")
        self.proj = proj
        self.geo = ccrs.Geodetic()
        self.ax.text(-0.02, 0.97, "Coord: MAG", ha="center", va="top", 
                     transform=self.ax.transAxes, fontsize="x-small", rotation=90)
        return
    
    def date_string(self, date, label_style="web"):
        # Set the date and time formats
        dfmt = "%d/%b/%Y" if label_style == "web" else "%d %b %Y,"
        tfmt = "%H:%M"
        stime, etime = date - dt.timedelta(minutes=self.dur/2),\
                date + dt.timedelta(minutes=self.dur/2)
        date_str = "{:{dd} {tt}} -- ".format(stime, dd=dfmt, tt=tfmt)
        if etime.date() == stime.date(): date_str = "{:s}{:{tt}} UT".format(date_str, etime, tt=tfmt)
        else: date_str = "{:s}{:{dd} {tt}} UT".format(date_str, etime, dd=dfmt, tt=tfmt)
        return date_str
    
    def set_frame(self, mlat, mlon, tecu, lim=True):
        if lim: vmin, vmax = 0, 20
        else: vmin, vmax = -.2, .2
        XYZ = self.proj.transform_points(self.geo, mlon, mlat)
        pc = self.ax.pcolor(XYZ[:,:,0], XYZ[:,:,1], tecu.T, cmap=cm.jet, vmin=vmin, vmax=vmax)
        ax = inset_axes(self.ax, width="3%", height="15%", loc="upper left")
        cbar = mpl.pyplot.colorbar(pc, cax=ax, orientation="vertical")
        cbar.ax.tick_params(labelsize="xx-small") 
        vlabel = r"TEC [$\times 10^{16} m^{-2}$]"
        cbar.set_label(vlabel, size="xx-small")
        return
    
    def save(self, fname):
        self.fig.savefig(fname, bbox_inches="tight")
        plt.close()
        return