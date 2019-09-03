import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import re
import os
import sys
import gc
import pickle
import gzip
from multiprocessing import Pool
import multiprocessing


def get_parallel_arg_list(n_jobs, arg_list, uid_list=[]):

    parallel_list = []
    length = len(arg_list)

    for num in range(n_jobs):

        if num == n_jobs - 1:
            if len(uid_list):
                tmp_key = uid_list[num * int(length/(n_jobs)) :  ]
            tmp_args = arg_list[num * int(length/(n_jobs)) :  ]
        else:
            if len(uid_list):
                tmp_key = uid_list[num * int(length/(n_jobs)) : (num+1) * int(length/(n_jobs)) ]
            tmp_args = arg_list[num * int(length/(n_jobs)) : (num+1) * int(length/(n_jobs)) ]

        if len(uid_list):
            parallel_list.append([tmp_key, tmp_args])
        else:
            parallel_list.append(tmp_args)

    return parallel_list


def load_file(path):
    if path.count('.csv'):
        return pd.read_csv(path)
    elif path.count('.gz'):
        delimiter='gz'
    elif path.count('.npy'):
        delimiter='npy'
    filename = get_filename(path=path, delimiter=delimiter)

    if filename[:5]=='train':
        filename = filename[6:]
    elif filename[:4]=='test':
        filename = filename[5:]

    if path[-3:]=='npy':
        tmp = pd.Series(np.load(path), name=filename)
    elif path[-2:]=='fp':
        with gzip.open(path, mode='rb') as fp:
            data = fp.read()
            tmp = pd.Series(pickle.loads(data), name=filename)
    elif path[-2:]=='gz':
        with gzip.open(path, mode='rb') as gz:
            data = gz.read()
            tmp = pd.Series(pickle.loads(data), name=filename)
    return tmp

def get_filename(path, delimiter='gz'):
    filename = re.search(rf'/([^/.]*).{delimiter}', path).group(1)
    return filename

def parallel_load_data(path_list, n_jobs=multiprocessing.cpu_count()):
    p = Pool(n_jobs)
    p_list = p.map(load_file, path_list)
    p.close
    return p_list

def load_file_wrapper(args):
    return load_file(*args)


def parallel_process(func, arg_list, n_jobs=multiprocessing.cpu_count()):
    process = Pool(n_jobs)
    #  p = Pool(len(arg_list))
    callback = process.map(func, arg_list)
    process.close()
    process.terminate()
    return callback

