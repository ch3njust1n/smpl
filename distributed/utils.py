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


'''
Get a Logger object

Input:  directory (string)           Directory containing log files
        filename  (string)           Log filename
        mode      (string, optional) Logging mode
        level     (int, optional)    Logging level
Output: log       (logging.Logger)   Logger object
        path      (string)           Absolute path to log file
'''
def log(directory, filename, mode='a', level=logging.DEBUG):
    if filename.endswith('.log'):
        filename = filename.split('.')[0]

    path = os.path.join(directory, '{}.log'.format(filename))
    
    formatter = logging.Formatter('[%(filename)s:%(lineno)s - %(funcName)20s()] %(asctime)s - %(levelname)s - %(message)s')
    fileHandler = logging.FileHandler(filename=path, mode=mode)
    fileHandler.setFormatter(formatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)

    log = logging.getLogger(filename)
    log.setLevel(level)
    log.addHandler(fileHandler)
    log.addHandler(streamHandler)

    return log, path


'''
Close log file. Once closed, can longer write or append to log.

Input:  logger (logging.Logger) Logger object
'''
def close_log(logger):
    for i, l in enumerate(logger.handlers):
        l.close()
        logger.removeHandler(l)


'''
Get current date and time

Output: (string) Formatted current date and time
'''
def get_date():
    return strftime("%Y-%m-%d-%H-%M-%S", gmtime())


'''
Get public IP address of this machine

Output: (string) IP address
'''
def get_publicIP():
    ip, err = Popen(["ipconfig", "getifaddr", "en0"], stdout=PIPE).communicate()
    return ip.strip()


'''
Pickel data structure

Input:  save_dir (string) Directory to save object to
Output: data     (object) Data structure
'''
def save_pickle(save_dir, data):
    filename = gen_filename(save_dir, 'p')
    pickle.dump(data, open(filename, 'wb'))
    return filename


'''
Save data as a JSON

Input:  save_dir (string)
        data     (object)
Output: (string) Path to saved file
'''
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


'''
Save pytorch data

Input:  data (dict)   PyTorch state data
        path (string) Path to save to
'''
def save_pt(data, path):
    save_dir = path.split('/')[:-1]
    if os.path.exists('/'.join(save_dir)) and path.endswith('.pt'):
        torch.save(data, path)
    else:
        raise Exception('Invalid save directory')


'''
Determine if file is a PyTorch file

Input:  path (string) Absolute path to file
Output: (bool) True if file is a PyTorch file
'''
def is_pt(path):
    return os.path.isfile(path) and (path.endswith('.pt') or path.endswith('.pth'))


'''
Load PyTorch file

Input:  path (string) Absolute path to PyTorch saved data
Output: Either state dict or empty list
'''
def load_pt(path):
    if is_pt(path):
        return torch.load(path)
    else:
        return []


'''
Generate filename

Input:  file_dir (string) Absolute directory to file
        ext      (string) File extension
Output: (string) Absolute path to file
'''
def gen_filename(file_dir, ext):
    ext = ext.split('.')[1]
    return os.path.join(file_dir, datetime.datetime.now().isoformat().replace(':', '-').replace('.', '') + '.' + ext)


'''
Retrieve latest model

Input:  save_dir (string) Absolute path to directory containing PyTorch save data
Output: object
'''
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
