'''
	Justin Chen

	6.27.17
'''

import os, pickle, datetime, ujson, codecs, glob, torch, math, logging
import socket, fcntl, struct
import numpy as np
from subprocess import PIPE, Popen
from time import gmtime, strftime
from sys import getsizeof


def log(directory, filename):
    # Setup logging
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger()
    path = os.path.join(directory, '{}.log'.format(filename))
    handler = logging.FileHandler(filename=path)
    formatter = logging.Formatter('[%(filename)s:%(lineno)s - %(funcName)20s()] %(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    handler.createLock()
    logger.addHandler(handler)
    return logger, path


def get_date():
    return strftime("%Y-%m-%d-%H-%M-%S", gmtime())


def get_publicIP():
    ip, err = Popen(["ipconfig", "getifaddr", "en0"], stdout=PIPE).communicate()
    return ip.strip()


def save_pickle(save_dir, data):
    filename = gen_filename(save_dir, 'p')
    pickle.dump(data, open(filename, 'wb'))
    return filename


def save_json(save_dir, data):
    filename = gen_filename(save_dir, 'json')
    ujson.dump(data, codecs.open(filename, 'w', encoding='utf-8'))
    return filename


'''
Load ujson object from a saved file

Input   file_path (string) Path to ujson file
Output: (list) Loaded ujson
'''
def load_json(file_path):
    with open(file_path, 'rb') as file:
        return ujson.load(file)


'''
Get the IP address of the current machine
Source:
https://raspberrypi.stackexchange.com/questions/6714/how-to-get-the-raspberry-pis-ip-address-for-ssh
'''
def get_ip_address(ifname):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    return socket.inet_ntoa(fcntl.ioctl(
        s.fileno(),
        0x8915,  # SIOCGIFADDR
        struct.pack('256s', ifname[:15])
    )[20:24])


'''
Determine which configuration information corresponds to this machine.
All nodes on the MOC use device ens3 for network communication.
This will raise an exception if the party.ujson configuration file is missing information
on this machine. Removes this machine from the list.

Input:  config (list) List from party.ujson
        eth (string, optional) Ethernet interface
Output: dictionary containing identity of this machine
'''
def get_me(config, eth='ens3'):
    ip = get_ip_address(eth)
    me = {}
    for i, conf in enumerate(config):
        if conf['host'] == ip:
            return dict(config.pop(i))
    raise Exception('Error: party.json is missing this host\'s information')


'''
Check if the party configuration file is correct

Input: roster (list)
'''
def check_party(roster):
    info = {"alias": [], "addr": []}

    for p in roster:
        if p['alias'] in info['alias']:
            raise Exception('Aliases must be unique')
        else:
            info['alias'].append(p['alias'])

        addr = '{}:{}'.format(p['host'], p['port'])
        
        if addr in info['addr']:
            raise Exception('Address (host, port) must be unique')
        else:
            info['addr'].append(addr)


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
