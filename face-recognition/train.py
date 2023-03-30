import pickle
from random import randint
from os import path

from filters import filters


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
        vals = [.5, .5, .5]
        if filter == filters.CUSTOM_BLUR:
            vals = [-1, -1, -1]
        data['params'].append(vals)
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
        if filter == filters.CUSTOM_BLUR:
            params[i] += 1
            while   params[i] != 0*121 + 2*11 + 9\
                and params[i] != 0*121 + 3*11 + 8\
                and params[i] != 0*121 + 4*11 + 7\
                and params[i] != 1*121 + 4*11 +10\
                and params[i] != 2*121 +10*11 + 4\
                and params[i] != 3*121 + 0*11 +10\
                and params[i] != 4*121 + 1*11 + 9\
                and params[i] != 4*121 +10*11 + 0\
                and params[i] != 5*121 + 5*11 + 8\
                and params[i] != 5*121 + 9*11 + 4\
                and params[i] != 6*121 + 8*11 + 5\
                and params[i] != 6*121 +10*11 + 3\
                and params[i] != 8*121 + 0*11 + 9\
                and params[i] != 8*121 + 8*11 + 1\
                and params[i] != 9*121 + 8*11 + 4\
                and params[i] !=10*121 + 8*11 + 4\
                and params[i] !=10*121 + 9*11 + 3:
                params[i] += 1
            continue
        params[i] += float(randint(*bounds)) / dec
    set(filter, params)
    return
