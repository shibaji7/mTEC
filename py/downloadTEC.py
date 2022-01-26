"""downloadTEC.py: Module is dedicated of data fetching using madrigal dwonlaod module"""

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
import madrigalWeb.madrigalWeb
from loguru import logger
import pandas as pd

class ReadTEC(object):
    """
    Read / Download data from Madrigal website
    """
    
    def __init__(self, dates):
        self.dates = dates
        with open("config/props.json", "r") as f: o = json.load(f)
        for k in o.keys():
            setattr(self, k, o[k])
        self.dn_file_names = []
        self.stored_files = []
        self.generate_filenames()
        return
    
    def generate_filenames(self):
        dn = self.dates[0].replace(hour=0,minute=0,second=0)
        while dn <= self.dates[1]:
            fname = self.file_name.format(Y=dn.year, y=str(dn.year)[2:], d="%02d"%dn.day,
                                          m="%02d"%dn.month, mmm=dn.strftime("%b").lower())
            if not os.path.exists("tmp/%s.gz"%fname): self.dn_file_names.append(fname)
            dn += dt.timedelta(1)
            self.stored_files.append("tmp/%s.gz"%fname)
        if len(self.dn_file_names) > 0: self.list_filenames()
        else: logger.info(f" Files are in local.")
        return
    
    def link(self):
        self.dlink = madrigalWeb.madrigalWeb.MadrigalData(self.madrigalUrl)
        return
    
    def list_listruments(self):
        instList = self.dlink.getAllInstruments()
        for inst in instList:
            if "GNSS" in inst.name: logger.info(" Inst. Name-", str(inst.name))
        return instList
    
    def list_experments(self, inst_id=8000):
        start, end = self.dates[0], self.dates[1]
        expList = self.dlink.getExperiments(inst_id, start.year, start.month, start.day, 
                                            start.hour, start.minute, start.second, 
                                            end.year, end.month, end.day, end.hour, 
                                            end.minute, end.second)
        logger.info(f" Number of experiments, {len(expList)}")
        return expList
    
    def list_filenames(self, inst_id=8000):
        if not hasattr(self, "dlink"): self.link()
        expList = self.list_experments(inst_id)
        for exp in expList:
            fileList  = self.dlink.getExperimentFiles(exp.id)
            for f in fileList:
                fn = "/".join(f.name.split("/")[4:]).replace("hdf5", "txt")
                if fn in self.dn_file_names: self.download_file(f, fn)
        return
    
    def download_file(self, f, fn):
        logger.info(f" E.File {fn}")
        file = "tmp/" + fn
        path = "/".join(file.split("/")[:-1])
        os.makedirs(path, exist_ok=True)
        result = self.dlink.downloadFile(f.name, file.replace("hdf5","txt"), self.user_name, self.user_email, 
                                         self.user_affiliation, "simple")
        fn = file.replace("hdf5","txt")
        os.system("gzip -d %s.gz"%fn)
        with open(fn, "r") as tf: lines = tf.readlines()
        ox = []
        for ln in lines[1:]:
            ln = list(filter(None, ln.replace("\n", "").split(" ")))
            date = dt.datetime(int(ln[0]),int(ln[1]),int(ln[2]),
                               int(ln[3]),int(ln[4]),int(ln[5]))
            ox.append({"DATE": date, "GDLAT": float(ln[11]), 
                       "GLON": float(ln[12]), "TEC": float(ln[13]), 
                       "GDALT": float(ln[15])})
        ox = pd.DataFrame.from_records(ox)
        ox.to_csv(fn, index=False, header=True)
        os.system("gzip %s"%fn)
        return
    
    def read_stored_files(self):
        logger.info(" Loading local files ...")
        dx = pd.DataFrame()
        for f in self.stored_files:
            logger.info(f" Load file {f}")
            os.system("gzip -d %s"%f)
            f = f.replace(".gz", "")
            dx = pd.concat([dx, pd.read_csv(f, parse_dates=["DATE"])])
            os.system("gzip %s"%f)
        logger.info(" Data loaded.")
        return dx
    
if __name__ == "__main__":
    dates = [dt.datetime(2013,1,8,0,1), dt.datetime(2013,1,8,23,59)]
    ReadTEC(dates).read_stored_files()