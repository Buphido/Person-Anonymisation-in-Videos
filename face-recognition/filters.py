import cv2 as cv
from math import ceil
from pickle import loads

from images import Images


def filters(image, filter):
    def greyscale():
        total = params[0] + params[1] + params[2]
        params[0] /= total
        params[1] /= total
        params[2] /= total
        for (y, X, Y, x) in locs:
            for j in range(x, X):
                for i in range(y, Y):
                    grey = params[0]*image.rgb[i, j, 0] + params[1]*image.rgb[i, j, 1] + params[2]*image.rgb[i, j, 2] / 3
                    image.rgb[i, j, 0] = grey
                    image.rgb[i, j, 1] = grey
                    image.rgb[i, j, 2] = grey
        return image

    def normalise():
        for (y, X, Y, x) in locs:
            box = image.rgb[y:Y, x:X]
            image.rgb[y:Y, x:X] = cv.normalize(box, box, params[0]*255, params[1]*255, cv.NORM_MINMAX)
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
            image.rgb[y:Y, x:X] = cv.normalize(image.rgb[y:Y, x:X], image.rgb[y:Y, x:X], 0, 255, cv.NORM_MINMAX)
            mean = 0.
            for i in range(X-x):
                for j in range(Y-y):
                    mean += float(image.rgb[y+j, x+i, 0])/(X-x)/(Y-y)
            box = cv.cvtColor(cv.Canny(image.rgb[y:Y, x:X], 0.66*mean, 1.33*mean), cv.COLOR_GRAY2RGB)
            image.rgb[y:Y, x:X] = box
        return image

    def blur():
        for (y, X, Y, x) in locs:
            #dim = int(params[0]*10+1)
            dim = 28#ceil(pow((X-x) * (Y-y) / dim/dim, .5))
            image.rgb[y:Y, x:X] = cv.blur(image.rgb[y:Y, x:X], (dim, dim))
        return image

    def pixel():
        for (y, X, Y, x) in locs:
            dim = int(params[0]*20+1)
            dim = ceil(pow((X-x) * (Y-y) / dim/dim, .5))
            for j in range(ceil((X-x)/dim)):
                for i in range(ceil((Y-y)/dim)):
                    box = image.rgb[y + i*dim: min(y + (i+1)*dim, Y), x + j*dim: min(x + (j+1)*dim, X)]
                    box = cv.blur(box, (2*dim, 2*dim))
                    image.rgb[y + i*dim: min(y + (i+1)*dim, Y), x + j*dim: min(x + (j+1)*dim, X)] = box
        return image

    def downscale():
        for (y, X, Y, x) in locs:
            dim = int(params[0]*20+1)
            box = cv.resize(image.rgb[y:Y, x:X], None, fx= 1/dim, fy= 1/dim, interpolation = cv.INTER_AREA)
            image.rgb[y:Y, x:X] = cv.resize(box, (X-x, Y-y), interpolation = cv.INTER_CUBIC)
        return image

    def replace():
        sea = cv.cvtColor(cv.imread('sea.jpg'), cv.COLOR_BGR2RGB)
        for (y, X, Y, x) in locs:
            image.rgb[y:Y, x:X] = sea[:Y-y, :X-x]
        return image

    data = loads(open('filter_enc', 'rb').read())
    index = 0
    for id in data['id']:
        if id == filter:
            break
        index += 1
    params = list(data['params'][index])

    kNames = loads(open('face_enc', 'rb').read())
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
            filters.REPLACE:            replace}.get(filter)()


filters.GREYSCALE, \
    filters.NORMALISE, \
    filters.WEIGHTED_GREYSCALE, \
    filters.RGB_NEGATIVE, \
    filters.NEGATIVE, \
    filters.CANNY, \
    filters.BLUR, \
    filters.PIXEL, \
    filters.DOWNSCALE, \
    filters.REPLACE \
    = tuple(range(10))
