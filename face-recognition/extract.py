import face_recognition
import cv2 as cv
from imutils import paths  # imutils includes opencv functions
from pickle import dumps
from os import walk

from progress import Progress


def extract(source):
    if source is None:
        return
    # get paths of each file in folder named Images
    # Images here that contains data(folders of various people)
    imagePath = list(paths.list_images(source))
    kNames = {}
    ndir = 0
    prog = Progress('Extraction:')
    # loop over the image paths
    for (root, dirs, files) in walk('yalefaces'):
        if ndir == 0:
            ndir = len(dirs) - 1
        if 'subject' not in root:
            continue
        # extract the person name from the image path
        name = root.split('/')[-1]
        prog.initialise(n=ndir*(len(files) - 1))
        kTypes = {}
        for file in files:
            if '.DS_Store' == file:
                continue
            _, image = cv.VideoCapture(root + '/' + file).read()
            rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            locs = face_recognition.face_locations(rgb, model='cnn')
            # compute the facial embedding for any face
            encodings = face_recognition.face_encodings(rgb, locs)
            # loop over the encodings
            kEncodings = []
            kLocs = []
            for i in range(len(encodings)):
                kLocs.append(locs[i])
                kEncodings.append(encodings[i])
            kTypes[file] = {'locs': kLocs, 'encodings': kEncodings}
            prog.update()
        kNames[name] = kTypes

    # save emcodings along with their names in dictionary data
    #data = {'encodings': kEncodings, 'subjects': kNames, 'types': kTypes}
    # use pickle to save data into a file for later use
    f = open('face_enc', 'wb')
    f.write(dumps(kNames))  # to open file in write mode
    f.close()  # to close file
    prog.quit()
    return


if __name__ == '__main__':
    extract('yalefaces')
