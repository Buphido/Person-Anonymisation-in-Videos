import cv2 as cv
from math import ceil
from pickle import loads


def filters(rgb, locs, filter):
    def greyscale():
        total = params[0] + params[1] + params[2]
        params[0] /= total
        params[1] /= total
        params[2] /= total
        for (y, X, Y, x) in locs:
            for j in range(x, X):
                for i in range(y, Y):
                    grey = params[0]*rgb[i, j, 0] + params[1]*rgb[i, j, 1] + params[2]*rgb[i, j, 2] / 3
                    rgb[i, j, 0] = grey
                    rgb[i, j, 1] = grey
                    rgb[i, j, 2] = grey
        return rgb

    def normalise():
        for (y, X, Y, x) in locs:
            box = rgb[y:Y, x:X]
            rgb[y:Y, x:X] = cv.normalize(box, box, params[0]*255, params[1]*255, cv.NORM_MINMAX)
        return rgb

    def weightedGreyscale():
        for (y, X, Y, x) in locs:
            box = cv.cvtColor(cv.cvtColor(rgb[y:Y, x:X], cv.COLOR_RGB2GRAY), cv.COLOR_GRAY2RGB)
            rgb[y:Y, x:X] = box
        return rgb

    def negate(img):
        for (y, X, Y, x) in locs:
            for j in range(x, X):
                for i in range(y, Y):
                    img[i, j, 0] = 255 - img[i, j, 0]
                    img[i, j, 1] = 255 - img[i, j, 1]
                    img[i, j, 2] = 255 - img[i, j, 2]
        return img

    def rgbNegative():
        return negate(rgb)

    def negative():
        return cv.cvtColor(negate(cv.cvtColor(rgb, cv.COLOR_RGB2HSV)), cv.COLOR_HSV2RGB)

    def canny():
        for (y, X, Y, x) in locs:
            box = cv.cvtColor(cv.Canny(rgb[y:Y, x:X], params[0]*200, params[1]*200), cv.COLOR_GRAY2RGB)
            rgb[y:Y, x:X] = box
        return rgb

    def blur():
        for (y, X, Y, x) in locs:
            dim = int(params[0]*10+1)
            dim = ceil(pow((X-x) * (Y-y) / dim/dim, .5))
            rgb[y:Y, x:X] = cv.blur(rgb[y:Y, x:X], (dim, dim))
        return rgb

    def pixel():
        for (y, X, Y, x) in locs:
            dim = int(params[0]*20+1)
            dim = ceil(pow((X-x) * (Y-y) / dim/dim, .5))
            for j in range(ceil((X-x)/dim)):
                for i in range(ceil((Y-y)/dim)):
                    box = rgb[y + i*dim: min(y + (i+1)*dim, Y), x + j*dim: min(x + (j+1)*dim, X)]
                    box = cv.blur(box, (2*dim, 2*dim))
                    rgb[y + i*dim: min(y + (i+1)*dim, Y), x + j*dim: min(x + (j+1)*dim, X)] = box
        return rgb

    def downscale():
        for (y, X, Y, x) in locs:
            dim = int(params[0]*20+1)
            box = cv.resize(rgb[y:Y, x:X], None, fx= 1/dim, fy= 1/dim, interpolation = cv.INTER_AREA)
            rgb[y:Y, x:X] = cv.resize(box, (X-x, Y-y), interpolation = cv.INTER_CUBIC)
        return rgb

    data = loads(open('filter_enc', 'rb').read())
    index = 0
    for id in data['id']:
        if id == filter:
            break
        index += 1
    params = list(data['params'][index])
    return {filters.GREYSCALE:          greyscale,
            filters.NORMALISE:          normalise,
            filters.WEIGHTED_GREYSCALE: weightedGreyscale,
            filters.RGB_NEGATIVE:       rgbNegative,
            filters.NEGATIVE:           negative,
            filters.CANNY:              canny,
            filters.BLUR:               blur,
            filters.PIXEL:              pixel,
            filters.DOWNSCALE:          downscale}.get(filter)()


filters.GREYSCALE, \
    filters.NORMALISE, \
    filters.WEIGHTED_GREYSCALE, \
    filters.RGB_NEGATIVE, \
    filters.NEGATIVE, \
    filters.CANNY, \
    filters.BLUR, \
    filters.PIXEL, \
    filters.DOWNSCALE \
    = tuple(range(9))
