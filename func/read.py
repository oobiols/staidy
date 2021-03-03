#import pymesh
import numpy as np
import glob
import re

def atoi(text):
 return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def coordinates(case):

 coords = np.loadtxt('./coordinates/'+case+'/*.obj')

 return coords


def addrs(data, benchmark, case, grid):

 x_addrs = []
 y_addr = []

 train_x_path   = "./" + data + "/" + benchmark + "/" + case + "/input/*"
 train_x_addrs  = sorted(glob.glob(train_x_path))
 train_x_addrs  = list(train_x_addrs)
 train_x_addrs.sort(key=natural_keys)
 train_x_addrs  = train_x_addrs[0:700]
 x_addrs.append(train_x_addrs)

 train_y_path   = "./" + data + "/" + benchmark + "/" + case + "/output/*"
 train_y_addr  = sorted(glob.glob(train_y_path))
 train_y_addr  = list(train_y_addr)
 train_y_addr.sort(key=natural_keys)
 y_addr.append(train_y_addr)

 x_addrs = np.asarray(x_addrs)
 y_addr  = np.asarray(y_addr)

 return x_addrs, y_addr
