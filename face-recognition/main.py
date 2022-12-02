import cv2 as cv
from sys import exit
from face_recognition import face_locations

from extract import extract
from recognise import recognise
from filters import filters
from train import train


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
    extract(None)

    # Find path to the image/video and pass it here
    subject = 'subject05'
    path = source + '/' + subject + '/centerlight'
    cap = cv.VideoCapture(path)

    detectable = Criteria(True, False)

    filter = filters.DOWNSCALE
    key = -1
    while key != 27:
        ret, image = cap.read()  # cv.imread('./archive/subject01/biden-trump.jpg')
        if not ret:
            cap = cv.VideoCapture(path)
            continue
        rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        # the facial embeddings for face in input
        blackLocs = face_locations(rgb, model='cnn', number_of_times_to_upsample=0)
        train(filter)
        rgb = filters(rgb, blackLocs, filter)
        locs = face_locations(rgb, model='cnn', number_of_times_to_upsample=0)
        result = cv.cvtColor(rgb, cv.COLOR_RGB2BGR)
        recognised = recognise(rgb, locs)
        #test = list(recognised)
        out = Criteria(len(locs) == 1, len(recognised) == 1 and recognised[0][1] == subject)
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
            cv.putText(result, 'Eff.: ' + str(effectiveness(detectable, out)), (x, Y+17), cv.FONT_HERSHEY_SIMPLEX, 0.75, colour, 2)

        cv.imshow('Frame', result)
        key = cv.waitKey(1000)

    cap.release()
    cv.destroyAllWindows()
    exit(0)
