import pickle
from random import randint
from os import path


def init():
    if not path.isfile('filter_enc'):
        initId = []
        initParams = []
        f = open('filter_enc', 'wb')
        f.write(pickle.dumps({'id': initId, 'params': initParams}))  # to open file in write mode
        f.close()  # to close file
    return


def get(filter):
    init()
    data = pickle.loads(open('filter_enc', 'rb').read())
    index = 0
    for id in data['id']:
        if id == filter:
            break
        index += 1
    if index == len(data['id']):
        data['params'].append([.5, .5, .5])
        data['id'].append(filter)
    return (data, index)


def set(filter, params):
    (data, index) = get(filter)
    data['params'][index] = tuple(params)
    f = open('filter_enc', 'wb')
    f.write(pickle.dumps(data))  # to open file in write mode
    f.close()  # to close file
    return


def train(filter):
    (data, index) = get(filter)
    off = .25
    dec = 1000000.
    params = list(data['params'][index])
    for (i, param) in enumerate(params):
        bounds = (int(-min(param, off)*dec), int(min(1.-param, off)*dec))
        params[i] += float(randint(*bounds)) / dec
    set(filter, params)
    return
