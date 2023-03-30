import cv2 as cv
import math
from math import ceil
from pickle import loads
from numpy import array

from images import Images


def filters(image, filter, locs=None):
    def greyscale():
        #total = params[0] + params[1] + params[2]
        #params[0] /= total
        #params[1] /= total
        #params[2] /= total
        for (y, X, Y, x) in locs:
            for j in range(x, X):
                for i in range(y, Y):
                    #grey = params[0]*image.rgb[i, j, 0] + params[1]*image.rgb[i, j, 1] + params[2]*image.rgb[i, j, 2] / 3
                    grey = image.rgb[i, j, 0] + image.rgb[i, j, 1] + image.rgb[
                        i, j, 2] / 3
                    image.rgb[i, j, 0] = grey
                    image.rgb[i, j, 1] = grey
                    image.rgb[i, j, 2] = grey
        return image

    def normalise():
        for (y, X, Y, x) in locs:
            box = image.rgb[y:Y, x:X]
            #image.rgb[y:Y, x:X] = cv.normalize(box, box, params[0]*255, params[1]*255, cv.NORM_MINMAX)
            image.rgb[y:Y, x:X] = cv.normalize(box, box, 0, 255, cv.NORM_MINMAX)
        return image

    def weightedGreyscale():
        for (y, X, Y, x) in locs:
            box = cv.cvtColor(cv.cvtColor(image.rgb[y:Y, x:X], cv.COLOR_RGB2GRAY), cv.COLOR_GRAY2RGB)
            image.rgb[y:Y, x:X] = box
        return image

    def negate(img):
        for (y, X, Y, x) in locs:
            for j in range(x, X):
                for i in range(y, Y):
                    img[i, j, 0] = 255 - img[i, j, 0]
                    img[i, j, 1] = 255 - img[i, j, 1]
                    img[i, j, 2] = 255 - img[i, j, 2]
        return img

    def rgbNegative():
        image.rgb = negate(image.rgb)
        return image

    def negative():
        image.rgb = cv.cvtColor(negate(cv.cvtColor(image.rgb, cv.COLOR_RGB2HSV)), cv.COLOR_HSV2RGB)
        return image

    def canny():
        for (y, X, Y, x) in locs:
            #box = cv.cvtColor(cv.Canny(image.rgb[y:Y, x:X], params[0]*200, params[1]*200), cv.COLOR_GRAY2RGB)
            #image.rgb[y:Y, x:X] = cv.normalize(image.rgb[y:Y, x:X], image.rgb[y:Y, x:X], 0, 255, cv.NORM_MINMAX)
            #mean = 0.
            #for i in range(X-x):
             #   for j in range(Y-y):
              #      mean += float(image.rgb[y+j, x+i, 0])/(X-x)/(Y-y)
            #box = cv.cvtColor(cv.Canny(image.rgb[y:Y, x:X], 255/3., 255), cv.COLOR_GRAY2RGB)
            box = cv.cvtColor(cv.Canny(image.rgb[y:Y, x:X], 100, 200), cv.COLOR_GRAY2RGB)
            for i in range(len(box)):
                for j in range(len(box[0])):
                    for k in range(3):
                        box[i,j,k] = max(128, box[i,j,k])
            #box = cv.cvtColor(cv.Canny(image.rgb[y:Y, x:X], 0.66*mean, 1.33*mean), cv.COLOR_GRAY2RGB)
            image.rgb[y:Y, x:X] = box
        return image

    def blur():
        for (y, X, Y, x) in locs:
            #dim = int(params[0]*10+1)
            #dim = 28#ceil(pow((X-x) * (Y-y) / dim/dim, .5))
            #dim = 17
            image = weightedGreyscale()
            dim = 50
            image.rgb[y:Y, x:X] = cv.blur(image.rgb[y:Y, x:X], (dim, dim))
        return image

    def pixel():
        for (y, X, Y, x) in locs:
            #dim = int(params[0]*20+1)
            dim = 10
            dim = ceil(pow((X-x) * (Y-y) / dim/dim, .5))
            for j in range(ceil((X-x)/dim)):
                for i in range(ceil((Y-y)/dim)):
                    box = image.rgb[y + i*dim: min(y + (i+1)*dim, Y), x + j*dim: min(x + (j+1)*dim, X)]
                    box = cv.blur(box, (2*dim, 2*dim))
                    image.rgb[y + i*dim: min(y + (i+1)*dim, Y), x + j*dim: min(x + (j+1)*dim, X)] = box
        return image

    def downscale():
        for (y, X, Y, x) in locs:
            #dim = int(params[0]*20+1)
            dim = 10
            box = cv.resize(image.rgb[y:Y, x:X], None, fx= 1/dim, fy= 1/dim, interpolation = cv.INTER_AREA)
            image.rgb[y:Y, x:X] = cv.resize(box, (X-x, Y-y), interpolation = cv.INTER_CUBIC)
        return image

    def replace():
        sea = cv.cvtColor(cv.imread('sea.jpg'), cv.COLOR_BGR2RGB)
        for (y, X, Y, x) in locs:
            image.rgb[y:Y, x:X, :] = 128#sea[:Y-y, :X-x]
        return image

    def customBlur():
        params = list(data['params'][index])
        #cen = math.floor(params[0]/11/11)-5
        #edg = (math.floor(params[0]/11)%11)-5
        #cor = (params[0]%11)-5
        cen = params[0]
        edg = params[1]
        cor = params[2]
        #cen /= weight
        #edg /= weight
        #cor /= weight
        kernel = array([[cor, edg, cor],
                        [edg, cen, edg],
                        [cor, edg, cor]])
        image.rgb = image.rgb.astype('float64')
        for (y, X, Y, x) in locs:
            image.rgb[y:Y, x:X] = cv.filter2D(image.rgb[y:Y, x:X], -1, kernel)
            for i in range(y,Y):
                for j in range(x,X):
                    for k in range(3):
                        image.rgb[i,j,k] = min(255, max(0, image.rgb[i,j,k]))#*weight))
        image.rgb = image.rgb.astype('uint8')
        return image


    data = loads(open('filter_enc', 'rb').read())
    index = 0
    for id in data['id']:
        if id == filter:
            break
        index += 1
    params = 0#list(data['params'][index])

    kNames = loads(open('face_enc', 'rb').read())
    if locs is None:
        if image.subjectStr() == '':
            locs = [(0, len(image.rgb[0]), len(image.rgb), 0)]
        else:
            locs = kNames[image.subjectStr()][image.typeStr()]['locs']

    return {filters.GREYSCALE:          greyscale,
            filters.NORMALISE:          normalise,
            filters.WEIGHTED_GREYSCALE: weightedGreyscale,
            filters.RGB_NEGATIVE:       rgbNegative,
            filters.NEGATIVE:           negative,
            filters.CANNY:              canny,
            filters.BLUR:               blur,
            filters.PIXEL:              pixel,
            filters.DOWNSCALE:          downscale,
            filters.REPLACE:            replace,
            filters.CUSTOM_BLUR:        customBlur}.get(filter)()


filters.GREYSCALE, \
    filters.NORMALISE, \
    filters.WEIGHTED_GREYSCALE, \
    filters.RGB_NEGATIVE, \
    filters.NEGATIVE, \
    filters.CANNY, \
    filters.BLUR, \
    filters.PIXEL, \
    filters.DOWNSCALE, \
    filters.REPLACE, \
    filters.CUSTOM_BLUR \
    = tuple(range(11))
