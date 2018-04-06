'''
	Justin Chen

	6.27.17

	Boston University 
	Hariri Institute for Computing and 
    Computational Sciences & Engineering
'''

import os, pickle, datetime, ujson, codecs, glob, torch, math
import numpy as np
from subprocess import PIPE, Popen

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
