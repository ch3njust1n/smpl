'''
	Justin Chen

	6.27.17
'''

import os, pickle, datetime, ujson, codecs, glob, torch, math
import numpy as np
from subprocess import PIPE, Popen
from time import gmtime, strftime
from sys import getsizeof
from random import choice


def get_date():
    strftime("%Y-%m-%d-%H-%M-%S", gmtime())


def get_publicIP():
    ip, err = Popen(["ipconfig", "getifaddr", "en0"], stdout=PIPE).communicate()
    return ip.strip()


def save_pickle(save_dir, data):
    filename = gen_filename(save_dir, 'p')
    pickle.dump(data, open(filename, 'wb'))
    return filename


def save_ujson(save_dir, data):
    filename = gen_filename(save_dir, 'ujson')
    ujson.dump(data, codecs.open(filename, 'w', encoding='utf-8'))
    return filename


def load_ujson(file_path):
    with open(file_path, 'rb') as file:
        return ujson.load(file)


def save_pt(data, path):
    save_dir = path.split('/')[:-1]
    if os.path.exists('/'.join(save_dir)) and path.endswith('.pt'):
        torch.save(data, path)
    else:
        raise Exception('Invalid save directory')


def is_pt(path):
    return os.path.isfile(path) and (path.endswith('.pt') or path.endswith('.pth'))


def load_pt(path):
    if is_pt(path):
        return torch.load(path)
    else:
        return []


def gen_filename(file_dir, ext):
    ext = ext.split('.')[1]
    return os.path.join(file_dir, datetime.datetime.now().isoformat().replace(':', '-').replace('.', '') + '.' + ext)


def latest_model(save_dir):
    return max(glob.iglob(os.path.join(save_dir, '*.pth')), key=os.path.getctime)

'''
Get the memory size of a Python construct.
Useful for optimizting memory usage.

Input:  obj   (object) Python object or primative
        units (str)    String abbreviation for memory units
Output: size  (int)    Memory size of object  
'''
def get_mem(obj, units='b'):
    scale = 1

    if units.lower() == 'kb':
        scale = 1e-3
    elif units.lower() == 'mb':
        scale = 1e-6
    elif units.lower() == 'gb':
        scale = 1e-9

    return getsizeof(a)*scale

