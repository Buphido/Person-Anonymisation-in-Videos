import sys

import face_recognition
import pickle
import cv2
import os
import math


def f(m):
    k = (m + 6*i/n) % 6
    return 255*(1 - max(0, min(k, 4 - k, 1)))


#def f(i, c):
    #return 127.5 * math.sin(2 * math.pi * (i / n + 0.25 + (c - 1) / 3)) + 127.5


def colour():
    out = (f(5), f(3), f(1))
    return out


if __name__ == '__main__':
    # to find path of xml file containing haarCascade file
    cfp = os.path.dirname(cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"
    # load the harcaascade in the cascade classifier
    fc = cv2.CascadeClassifier(cfp)
    # load the known faces and embeddings saved in last file
    data = pickle.loads(open('face_enc', "rb").read())

    # Find path to the image you want to detect face and pass it here
    image = cv2.imread('family-glasses.JPG')
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Overwrite image with its negative
    height, width, _ = rgb.shape
    # for i in range(height):
    #   for j in range(width):
    #      for k in range(0, 3):
    #         if(rgb[i, j, k] < 240):
    #            if(k>1):
    #               rgb[i, j, k] = 180
    #              continue
    #         if(k>0):
    #            rgb[i, j, k] = 206
    #           continue
    #      rgb[i, j, k] = 255

    # the facial embeddings for face in input
    locs = face_recognition.face_locations(rgb, model="cnn", number_of_times_to_upsample=2)

    for (y, X, Y, x) in locs :
        for j in range(x, X):
            for i in range(y, Y):
                grey = rgb[i, j, 0]/3. + rgb[i, j, 1]/3. + rgb[i, j, 2]/3
                rgb[i, j, 0] = grey
                rgb[i, j, 1] = grey
                rgb[i, j, 2] = grey
    result = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    encodings = face_recognition.face_encodings(rgb, known_face_locations=locs)
    names = []
    # loop over the facial embeddings incase
    # we have multiple embeddings for multiple faces
    for encoding in encodings:
        # Compare encodings with encodings in data["encodings"]
        # Matches contain array with boolean values True and False
        matches = face_recognition.compare_faces(data["encodings"], encoding)
        # set name =unknown if no encoding matches
        name = "Unknown"
        # check to see if we have found a match
        if True in matches:
            # Find positions at which we get True and store them
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            count = {}
            # loop over the matched indexes and maintain a count for
            # each recognized face face
            for i in matchedIdxs:
                # Check the names at respective indexes we stored in matchedIdxs
                name = data["names"][i]
                # increase count for the name we got
                count[name] = count.get(name, 0) + 1
            # set name which has highest count
            name = max(count, key=count.get)
            # will update the list of names
            names.append(name)

    # do loop over the recognized faces
    n = min(len(locs), len(names))
    for (i, (y, X, Y, x), name) in zip(range(n), locs, names):
        # rescale the face coordinates
        # draw the predicted face name on the image
        cv2.rectangle(result, (x, y), (X, Y), colour(), 2)
        print(name, colour())
        cv2.putText(result, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, colour(), 2)
    cv2.imshow("Frame", result)
    key = 0
    while key != 27:
        key = cv2.waitKey(20)
    cv2.destroyAllWindows()
    sys.exit(0)
