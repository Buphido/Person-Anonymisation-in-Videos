import cv2 as cv
import numpy as np
from sys import exit
from face_recognition import face_locations
from random import randint
from pickle import loads

from extract import extract
from recognise import recognise
from filters import filters
from train import train
from images import Images
from progress import Progress


def effectiveness(opt, out):
    comp = Criteria(opt.detectable is None or opt.detectable == out.detectable, opt.recognisable == out.recognisable)
    #return (int(comp.detectable) + int(comp.recognisable))/2
    return float(int(float(100000*passed)/float(passed + failed)))/100000.


def mark():
    black = name == 'Undetectable'
    colour = (255*int(out.recognisable and not black), 0, 255*int(not out.recognisable and not black))
    border = (255*int(not out.recognisable or black), 255, 255*int(out.recognisable or black))
    # draw the corresponding rectangle
    cv.rectangle(result, (x, y), (X, Y), colour, 2)
    if out.detectable != black:
        # write the predicted face name on the image
        cv.putText(result, name, (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.75, border, 4)
        cv.putText(result, name, (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.75, colour, 2)
        # write the filter efficiency on the image
        eff = unrecognisable.effectiveness(out)
        cv.putText(result, 'Rec.: ' + str(eff), (x, Y + 17), cv.FONT_HERSHEY_SIMPLEX, 0.75,
                   border, 4)
        cv.putText(result, 'Rec.: ' + str(eff), (x, Y + 17), cv.FONT_HERSHEY_SIMPLEX, 0.75,
                   colour, 2)
        eff = detectable.effectiveness(out)
        cv.putText(result, 'Det.: ' + str(eff), (x, Y + 40), cv.FONT_HERSHEY_SIMPLEX, 0.75,
                   border, 4)
        cv.putText(result, 'Det.: ' + str(eff), (x, Y + 40), cv.FONT_HERSHEY_SIMPLEX, 0.75,
                   colour, 2)
    return


def problem():
    test = (subject-3.5)*(subject-4.5)*(subject-5.5)*(subject-8.5)*(subject-13.5)*(subject-14.5)
    if test < 0:
        return True
    return False


class Criteria:
    passed = 0
    total = 0
    def __init__(self, detectable, recognisable):
        self.detectable = detectable
        self.recognisable = recognisable

    def effectiveness(self, out):
        if self.total < 110:
            comp = Criteria(self.detectable is None or self.detectable == out.detectable, self.recognisable == out.recognisable)
            ret = int(comp.detectable and comp.recognisable)
            self.total += 1
            self.passed += ret
        return float(int(float(100000 * self.passed) / float(self.total))) / 100000.

if __name__ == '__main__':
    source = 'yalefaces'
    # extract facial data from data set by relative path, if none given previously trained model is taken
    extract(None)
    kNames = loads(open('face_enc', 'rb').read())
    results = []
    for filt in range(10):
        if filt != filters.WEIGHTED_GREYSCALE and filt != filters.CANNY and filt != filters.BLUR and filt != filters.REPLACE:
            continue
        detectable = Criteria(True, False)
        unrecognisable = Criteria(None, False)
        prog = Progress('Calculating effectiveness ' + str(filt) + ':', 110, 100)
        prog.initialise(None)
        for subject in range(1,16):
            if problem():
                continue
            for type in range(11):
                # set subject by number and type of image by predefined constant in Images
                image = Images(subject, type)
                #ret, image = img.cap.read()  # cv.imread('./archive/subject01/biden-trump.jpg')
                #if not ret:
                 #   cap = cv.VideoCapture(path)
                  #  continue
                #rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)


                # find locations of faces before filtering
                locs = kNames[image.subjectStr()][image.typeStr()]['locs']
                # train/mutate selected filter
                #train(filt)
                # apply selected filter
                image = filters(image, filt)
                # preserve locs in markers for later use
                markers = list(zip(locs, ['Undetectable' for loc in locs]))

                # find locations of faces after filtering
                locs = face_locations(image.rgb, model='cnn')
                # attempt to recognise filtered faces
                recognised = recognise(image, locs)
                # record fulfillment of the effectiveness criteria
                out = Criteria(len(locs) == 1, len(recognised) == 1 and recognised[0][1] == image.subjectStr())

                # prepare output image
                result = cv.cvtColor(image.rgb, cv.COLOR_RGB2BGR)
                # prepare list of all markers to be added to the image
                markers += recognised
                # iterate over the markers and apply them on result
                for ((y, X, Y, x), name) in markers:
                    mark()

                collage = np.copy(result)
                if len(results) == 1:
                    collage = np.concatenate((results[0], result), axis=1)
                if len(results) == 2:
                    collage.fill(255)
                    collage = np.concatenate((np.concatenate((results[0], results[1]), axis=1),
                                              np.concatenate((result,     collage),    axis=1)), axis=0)
                if len(results) == 3:
                    collage = np.concatenate((np.concatenate((results[0], results[1]), axis=1),
                                              np.concatenate((results[2], result),     axis=1)), axis=0)
                cv.imshow('Frame', collage)
                cv.waitKey(1)
                prog.update()
        results.append(result)
        prog.quit()

    key = -1
    while key != 27:
        key = cv.waitKey(1)
    cv.destroyAllWindows()
    exit(0)
