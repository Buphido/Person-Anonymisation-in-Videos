from imutils import paths  # imutils includes opencv functions
import face_recognition
import pickle
import cv2 as cv
import os

from progress import progress


def extract(source):
    if source is None:
        return
    # get paths of each file in folder named Images
    # Images here that contains data(folders of various people)
    imagePath = list(paths.list_images(source))
    kEncodings = []
    kNames = []

    i = n = 0
    # loop over the image paths
    for (root, dirs, files) in os.walk('yalefaces'):
        if n == 0:
            n = len(dirs) - 1
        if "subject" not in root:
            continue
        # extract the person name from the image path
        name = root.split(os.path.sep)[-1]
        j = 0
        m = len(files) - 1
        for file in files:
            if ".DS_Store" == file:
                continue
            cap = cv.VideoCapture(root + os.path.sep + file)
            while True:
                # load the input image and convert it from BGR
                ret, image = cap.read()
                if not ret:
                    break
                rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)

                boxes = face_recognition.face_locations(rgb, model='hog')
                # compute the facial embedding for the any face
                encodings = face_recognition.face_encodings(rgb, boxes)
                # loop over the encodings
                for encoding in encodings:
                    kEncodings.append(encoding)
                    kNames.append(name)

                    # save emcodings along with their names in dictionary data
                    data = {"encodings": kEncodings, "names": kNames}
                    # use pickle to save data into a file for later use
                    f = open("face_enc", "wb")
                    f.write(pickle.dumps(data))  # to open file in write mode
                    f.close()  # to close file
                j += 1
                progress("Extraction:", i*m + j, n*m, 100)
        i += 1
    print()
    return
