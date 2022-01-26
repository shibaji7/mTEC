"""run.py: Module is the __main__ module"""

__author__ = "Chakraborty, S."
__copyright__ = ""
__credits__ = []
__license__ = "MIT"
__version__ = "1.0."
__maintainer__ = "Chakraborty, S."
__email__ = "shibaji7@vt.edu"
__status__ = "Research"

import sys
sys.path.append("py/")
import argparse
from dateutil import parser as prs
from loguru import logger
import datetime as dt

from downloadTEC import ReadTEC
from analyzeTEC import *

def run_analysis(args):
    logger.info(" Run analysis...")
    dates = [args.start-dt.timedelta(hours=1), args.end+dt.timedelta(hours=1)]
    frame = ReadTEC(dates).read_stored_files()
    call_run_plot(dates, frame)
    return

# Script run can also be done via main program
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--start", default=dt.datetime(2013,1,8,19), help="Start date", type=prs.parse)
    parser.add_argument("-e", "--end", default=dt.datetime(2013,1,8,22), help="End date", type=prs.parse)
    parser.add_argument("-v", "--verbose", action="store_false", help="Increase output verbosity (default True)")
    logger.info(f" Simulation run using run.__main__")
    args = parser.parse_args()
    if args.verbose:
        logger.info(" Parameter list for run ")
        for k in vars(args).keys():
            print("     ", k, "->", vars(args)[k])
    run_analysis(args)
    pass