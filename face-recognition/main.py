import sys
import face_recognition
import pickle
import cv2 as cv
import os
import math

from extract import extract
from recognise import recognise
from filters import filters


def f(m):
    k = (m + 6 * i / n) % 6
    return 255 * (1 - max(0, min(k, 4 - k, 1)))


# def f(i, c):
# return 127.5 * math.sin(2 * math.pi * (i / n + 0.25 + (c - 1) / 3)) + 127.5


def colour():
    out = (f(5), f(3), f(1))
    return out


def effectiveness(opt, out):
    comp = Criteria(opt.detectable == out.detectable, opt.recognisable == out.recognisable)
    return (int(comp.detectable) + int(comp.recognisable))/2


class Criteria:
    def __init__(self, detectable, recognisable):
        self.detectable = detectable
        self.recognisable = recognisable

if __name__ == '__main__':
    source = 'yalefaces'
    # extract facial data from data set by relative path, if none given previously trained model is taken
    extract(source)
    # to find path of xml file containing haarCascade file
    cfp = os.path.dirname(cv.__file__) + "/data/haarcascade_frontalface_alt2.xml"
    # load the harcaascade in the cascade classifier
    fc = cv.CascadeClassifier(cfp)
    # load the known faces and embeddings saved in last file
    data = pickle.loads(open('face_enc', "rb").read())

    # Find path to the image/video and pass it here
    subject = 'subject05'
    path = source + '/' + subject + '/centerlight'
    cap = cv.VideoCapture(path)

    detectable = Criteria(True, False)

    key = -1
    while key != 27:
        ret, image = cap.read()  # cv.imread('./archive/subject01/biden-trump.jpg')
        if not ret:
            cap = cv.VideoCapture(path)
            continue
        rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        # the facial embeddings for face in input
        blackLocs = face_recognition.face_locations(rgb, model="cnn", number_of_times_to_upsample=2)
        rgb = filters(rgb, blackLocs, filters.downscale)
        locs = face_recognition.face_locations(rgb, model="cnn", number_of_times_to_upsample=0)
        result = cv.cvtColor(rgb, cv.COLOR_RGB2BGR)
        recognised = recognise(rgb, locs)
        #test = list(recognised)
        out = Criteria(len(locs) == 1, len(recognised) == 1 and recognised[0][1] == subject)
        print("\b\b\b" + str(effectiveness(detectable, out)), end = '')
        for (y, X, Y, x) in blackLocs:
            cv.rectangle(result, (x, y), (X, Y), (0, 0, 0), 2)
        for (y, X, Y, x) in locs:
            cv.rectangle(result, (x, y), (X, Y), (0, 0, 255), 2)
        for ((y, X, Y, x), name) in recognised:
            colour = (255*int(out.recognisable), 0, 255*int(not out.recognisable))
            # rescale the face coordinates
            # draw the predicted face name on the image
            cv.rectangle(result, (x, y), (X, Y), colour, 2)
            # print(name, colour())
            cv.putText(result, name, (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.75, colour, 2)

        cv.imshow("Frame", result)
        key = cv.waitKey(1000)

    cap.release()
    cv.destroyAllWindows()
    sys.exit(0)
