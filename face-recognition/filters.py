import cv2 as cv
import math


def filters(rgb, locs, m):
    def greyscale():
        for (x, Y, X, y) in locs:
            for i in range(x, X):
                for j in range(y, Y):
                    grey = rgb[i, j, 0] / 3. + rgb[i, j, 1] / 3. + rgb[i, j, 2] / 3
                    rgb[i, j, 0] = grey
                    rgb[i, j, 1] = grey
                    rgb[i, j, 2] = grey
        return rgb

    def spreadGreyscale():
        mn = 255
        mx = 0
        for (x, Y, X, y) in locs:
            for i in range(x, X):
                for j in range(y, Y):
                    grey = rgb[i, j, 0] / 3. + rgb[i, j, 1] / 3. + rgb[i, j, 2] / 3
                    mx = max(grey, mx)
                    mn = min(grey, mn)
                    rgb[i, j, 0] = grey
                    rgb[i, j, 1] = grey
                    rgb[i, j, 2] = grey
        for (x, Y, X, y) in locs:
            for i in range(x, X):
                for j in range(y, Y):
                    spread = (rgb[i, j, 0] - mn) / (mx - mn) * 255
                    rgb[i, j, 0] = spread
                    rgb[i, j, 1] = spread
                    rgb[i, j, 2] = spread
        return rgb

    def weightedGreyscale():
        for (x, Y, X, y) in locs:
            box = cv.cvtColor(cv.cvtColor(rgb[x:X, y:Y], cv.COLOR_RGB2GRAY), cv.COLOR_GRAY2RGB)
            rgb[x:X, y:Y] = box
        return rgb

    def negate(img):
        for (x, Y, X, y) in locs:
            for i in range(x, X):
                for j in range(y, Y):
                    img[i, j, 0] = 255 - img[i, j, 0]
                    img[i, j, 1] = 255 - img[i, j, 1]
                    img[i, j, 2] = 255 - img[i, j, 2]
        return img

    def rgbNegative():
        return negate(rgb)

    def negative():
        return cv.cvtColor(negate(cv.cvtColor(rgb, cv.COLOR_RGB2HSV)), cv.COLOR_HSV2RGB)

    def canny():
        for (x, Y, X, y) in locs:
            box = cv.cvtColor(cv.Canny(rgb[x:X, y:Y], 0, 100), cv.COLOR_GRAY2RGB)
            rgb[x:X, y:Y] = box
        return rgb

    def blur():
        for (x, Y, X, y) in locs:
            dim = 3
            dim = math.ceil(pow((X-x) * (Y-y) / dim/dim, .5))
            rgb[x:X, y:Y] = cv.blur(rgb[x:X, y:Y], (dim, dim))
        return rgb

    def pixel():
        for (x, Y, X, y) in locs:
            dim = 14
            dim = math.ceil(pow((X-x) * (Y-y) / dim/dim, .5))
            i = 0
            while x + i*dim < X:
                j = 0
                while y + j*dim < Y:
                    box = rgb[x + i*dim: min(x + (i+1)*dim, X-1), y + j*dim: min(y + (j+1)*dim, Y-1)]
                    (width, height, _) = box.shape
                    if width <= 0 or height <= 0 :
                        break;
                    box = cv.blur(box, (2*dim, 2*dim))
                    rgb[x + i*dim: min(x + (i+1)*dim, X-1), y + j*dim: min(y + (j+1)*dim, Y-1)] = box
                    #print("completed", i*math.ceil((Y-y)/dim)+j+1, "/", math.ceil((X-x)/dim)*math.ceil((Y-y)/dim))
                    j += 1
                i += 1
        return rgb

    return {filters.greyscale: greyscale,
            filters.spreadGreyscale: spreadGreyscale,
            filters.weightedGreyscale: weightedGreyscale,
            filters.rgbNegative: rgbNegative,
            filters.negative: negative,
            filters.canny: canny,
            filters.blur: blur,
            filters.pixel: pixel}.get(m)()


filters.greyscale, \
    filters.spreadGreyscale, \
    filters.weightedGreyscale, \
    filters.rgbNegative, \
    filters.negative, \
    filters.canny, \
    filters.blur, \
    filters.pixel \
    = tuple(range(8))
