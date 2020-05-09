import numpy as np
import glob
import os
import pandas as pd
import sys
import h5py
from enum import Enum
import itertools
import math
import targets
import argparse
#from closed_worlds import CW_900


class Feature(Enum):
    timestamp = 0
    length = 1
    direction = 2
    wang = 3


TS = "timestamp"
DIR = "direction"
LEN = "length"
NUM = "num_cells"
ACK = "ack"
WANG = "wang"


def path_txt(filepath,txtpath):
    doc=open(txtpath,'w')
    for root ,dirs, files in os.walk(filepath):
        for file_single in files:
            test = filepath+file_single
            doc.write(test+'\n')
    doc.close()
 

def expand2(dir, num_cells):
    expanded = []
    for _ in range(0, num_cells):
        expanded.append(dir)
    return expanded


def expand(num_cells):
    dir = int(math.copysign(1, num_cells))
    num_cells = abs(num_cells)
    expanded = []
    for _ in range(0, num_cells):
        expanded.append(dir)
    return expanded


def to_wang(trace, maxlen):
    
    # pad to maxlen
    trace = np.pad(trace, (0, maxlen - trace.size), 'constant')
    #默认填充为0
    trace.astype(np.int8)
    #转换为8bit的整数
    return trace


def parse_target(filename):
    return filename.split("_")[1:-1][0]
    # return filename.split("_")[0]
    # return filename.split("_")[0]


def smart_shuffle(lists, list=False):
    if list:
        return [x for x in itertools.chain.from_iterable(itertools.zip_longest(*lists)) if x is not None and x.any()]
    else:
        return [x for x in itertools.chain.from_iterable(itertools.zip_longest(*lists)) if x]


#def save_data(dirpath, savepath, towang=True, h5=False):
    #fnames = glob.glob(os.path.join(dirpath, "*/*/*.csv"))
def save_data(txtname, savepath, maxlen, traces,towang=True, h5=False): #, type)
    """Saves the Tor dataset.
    # Arguments
        txtname: path to txt file with list of files with traffic traces
        savepath: path to file where to save the data
        compress: compress the data
        h5: save in h5 format as well
    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """
    features = ["direction", "num_cells"]
    #if towang:
    #    nb_features = 1
    #else:
    #   nb_features = len(features)

    fnames = [x.strip() for x in open(txtname, "r").readlines()]
    nb_traces = len(fnames)

    print("Saving dataset with {} traces".format(nb_traces))

    labels = np.empty([nb_traces], dtype=np.dtype(object))
    #if type == "lstm":
    #    data = np.empty([nb_traces, maxlen, nb_features], dtype=np.int16)  
    data = np.empty([nb_traces, maxlen], dtype=np.int16)
    
    num_labels = {}
    i = 0
    
    for f in fnames:
        label = parse_target(os.path.basename(f)) 
        if traces:
            if label not in num_labels:
                num_labels[label] = 1
            else:
                count = num_labels[label]
                if count == traces:
                    continue
                else:
                    num_labels[label] = count + 1
        
        #try:    
        # df = pd.read_csv(f, nrows=10000, sep=";", dtype=None, usecols=[0,1,2,3,4,5], header=0)
        df = pd.read_csv(f, nrows=10000,dtype=None, usecols=[0,1,2,3,4,5], header=0)
        #except:
        #    continue
        # if df.empty:
        #     print("empty", f)
        #     continue
        # if towang:
            #values = to_wang(df["direction"] * df["num_cells"], maxlen)
        
        #df.columns=['packet_index','timestamp','length','direction']#头
        # df.columns=['No.','Time','Source','Destination','Protocol','Length']
        #df.eval('new1=length*direction',inplace=True)
        # merge Length and direction
        #predata=df[df.columns[4]]/60
        # predata=(i for i in predata if i!=0)
        # predata=df[predata]
        #values =to_wang(df[df.columns[5]], maxlen)
        values =to_wang(df[df.columns[5]]/60, maxlen)  
        # else:
        #     print("WRONG")
        #values = df.values
      
        data[i] = values
        labels[i] = label
        i += 1

    labels = np.array(labels)
    nb_classes = len(set(labels))
    data = np.array(data)
   
    # Save in npz
    savepath = savepath + '.npz'
    np.savez_compressed(savepath, data=data, labels=labels) 
    print('Saved a dataset with {} traces for {} websites to {}'.format(data.shape[0], nb_classes, savepath))
    
    if h5:
        # Save hdf5
        h5f = h5py.File('{}.h5'.format(savepath), 'w')
        print(data)
        h5f.create_dataset('data', data=data)
        h5f.create_dataset('labels', data=labels)
        h5f.close()


def show_npz(datapath,txtpath,xname,yname):
    np.set_printoptions(threshold=np.inf)
    npzfile=np.load(datapath)
    x=npzfile[xname]
    y=npzfile[yname]
    npzfile.close()
    doc=open(txtpath,'w')
    for i in range(0, len(y)): 
        print(y[i],file=doc)
    #for i in x:
    #     print(i,file=doc)
    doc.close()

path_txt(<file_path>,<path_txt>); ##input your file_path and path_txt
save_data('ilepath.txt','data', 5000,100)

#show_npz("npz_path","show_path","data","labels")

