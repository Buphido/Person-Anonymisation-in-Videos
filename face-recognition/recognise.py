import face_recognition
from pickle import loads

from images import Images


def recognise(image, locs):
    # load the known faces and embeddings saved in last file
    kNames = loads(open('face_enc', 'rb').read())

    encodings = face_recognition.face_encodings(image.rgb, known_face_locations=locs)
    names = []
    # loop over the facial embeddings incase
    # we have multiple embeddings for multiple faces
    for encoding in encodings:
        # Compare encodings with encodings in data["encodings"]
        # Matches contain array with boolean values True and False
        matches = []
        fp = []
        for kName, kTypes in kNames.items():
            for kType, data in kTypes.items():
                matches += face_recognition.compare_faces(data['encodings'], encoding)
                fp.append({'name': kName, 'type': kType})
        # set name = Unrecognisable if no encoding matches
        name = 'Unrecognisable'
        # check to see if we have found a match
        if True in matches:
            # Find positions at which we get True and store them
            matchedIdxs = [id for (id, b) in enumerate(matches) if b]
            count = {}
            # loop over the matched indexes and maintain a count for
            # each recognized face
            for id in matchedIdxs:
                if fp[id]['name'] == image.subjectStr() and fp[id]['type'] == image.typeStr():
                    continue
                # Check the names at respective indexes we stored in matchedIdxs
                name = fp[id]['name']
                # increase count for the name we got
                count[name] = count.get(name, 0) + 1
            # set name which has highest count
            if len(count) > 0:
                name = max(count, key=count.get)
        # will update the list of names
        names.append(name)

    # do loop over the recognized faces
    return tuple(zip(locs, names))

