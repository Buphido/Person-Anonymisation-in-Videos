import cv2 as cv
from math import ceil


def filters(rgb, locs, m):
    def greyscale():
        for (y, X, Y, x) in locs:
            for j in range(x, X):
                for i in range(y, Y):
                    grey = rgb[i, j, 0] / 3. + rgb[i, j, 1] / 3. + rgb[i, j, 2] / 3
                    rgb[i, j, 0] = grey
                    rgb[i, j, 1] = grey
                    rgb[i, j, 2] = grey
        return rgb

    def spreadGreyscale():
        mn = 255
        mx = 0
        for (y, X, Y, x) in locs:
            for j in range(x, X):
                for i in range(y, Y):
                    grey = rgb[i, j, 0] / 3. + rgb[i, j, 1] / 3. + rgb[i, j, 2] / 3
                    mx = max(grey, mx)
                    mn = min(grey, mn)
                    rgb[i, j, 0] = grey
                    rgb[i, j, 1] = grey
                    rgb[i, j, 2] = grey
        for (y, X, Y, x) in locs:
            for j in range(x, X):
                for i in range(y, Y):
                    spread = (rgb[i, j, 0] - mn) / (mx - mn) * 255
                    rgb[i, j, 0] = spread
                    rgb[i, j, 1] = spread
                    rgb[i, j, 2] = spread
        return rgb

    def weightedGreyscale():
        for (y, X, Y, x) in locs:
            box = cv.cvtColor(cv.cvtColor(rgb[y:Y, x:X], cv.COLOR_RGB2GRAY), cv.COLOR_GRAY2RGB)
            rgb[y:Y, x.X] = box
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
            box = cv.cvtColor(cv.Canny(rgb[y:Y, x:X], 25, 150), cv.COLOR_GRAY2RGB)
            rgb[y:Y, x:X] = box
        return rgb

    def blur():
        for (y, X, Y, x) in locs:
            dim = 3
            dim = ceil(pow((X-x) * (Y-y) / dim/dim, .5))
            rgb[y:Y, x:X] = cv.blur(rgb[y:Y, x:X], (dim, dim))
        return rgb

    def pixel():
        for (y, X, Y, x) in locs:
            dim = 10
            dim = ceil(pow((X-x) * (Y-y) / dim/dim, .5))
            for j in range(ceil((X-x)/dim)):
                for i in range(ceil((Y-y)/dim)):
                    box = rgb[y + i*dim: min(y + (i+1)*dim, Y), x + j*dim: min(x + (j+1)*dim, X)]
                    box = cv.blur(box, (2*dim, 2*dim))
                    rgb[y + i*dim: min(y + (i+1)*dim, Y), x + j*dim: min(x + (j+1)*dim, X)] = box
        return rgb

    def downscale():
        for (y, X, Y, x) in locs:
            dim = 12
            box = cv.resize(rgb[y:Y, x:X], None, fx= 1/dim, fy= 1/dim, interpolation = cv.INTER_AREA)
            rgb[y:Y, x:X] = cv.resize(box, (X-x, Y-y), interpolation = cv.INTER_CUBIC)
        return rgb

    return {filters.greyscale: greyscale,
            filters.spreadGreyscale: spreadGreyscale,
            filters.weightedGreyscale: weightedGreyscale,
            filters.rgbNegative: rgbNegative,
            filters.negative: negative,
            filters.canny: canny,
            filters.blur: blur,
            filters.pixel: pixel,
            filters.downscale: downscale}.get(m)()


filters.greyscale, \
    filters.spreadGreyscale, \
    filters.weightedGreyscale, \
    filters.rgbNegative, \
    filters.negative, \
    filters.canny, \
    filters.blur, \
    filters.pixel, \
    filters.downscale \
    = tuple(range(9))
