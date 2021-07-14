import numpy as np
import config

def convert_polar(xdata):
    lst = []
    for entry in xdata:
        ax, ay, az, bx, by, bz, lx, ly, lz = entry
        at = np.arccos(az)
        ap = np.arctan2(ay, ax)
        bt = np.arccos(bz)
        bp = np.arctan2(by, bx)
        lt = np.arccos(lz)
        lp = np.arctan2(ly, lx)
        lst.append([at, ap, bt, bp, lt, lp])
    return np.array(lst)
