import cv2 as cv
import math
import os
import numpy as np
from sys import exit
from face_recognition import face_locations
from random import randint
from pickle import loads

import train
from extract import extract
from recognise import recognise
from filters import filters
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
        cv.putText(result, '(' + str(cen) + ',', (x, y-69), cv.FONT_HERSHEY_SIMPLEX, 0.75, border, 4)
        cv.putText(result, '(' + str(cen) + ',', (x, y-69), cv.FONT_HERSHEY_SIMPLEX, 0.75, colour, 2)
        cv.putText(result,       str(edg) + ',', (x, y-46), cv.FONT_HERSHEY_SIMPLEX, 0.75, border, 4)
        cv.putText(result,       str(edg) + ',', (x, y-46), cv.FONT_HERSHEY_SIMPLEX, 0.75, colour, 2)
        cv.putText(result,       str(cor) + ')', (x, y-23), cv.FONT_HERSHEY_SIMPLEX, 0.75, border, 4)
        cv.putText(result,       str(cor) + ')', (x, y-23), cv.FONT_HERSHEY_SIMPLEX, 0.75, colour, 2)
        #cv.putText(result, '(' + str(2*cen) + ',', (x, y - 69), cv.FONT_HERSHEY_SIMPLEX, 0.75, border, 4)
        #cv.putText(result, '(' + str(2*cen) + ',', (x, y - 69), cv.FONT_HERSHEY_SIMPLEX, 0.75, colour, 2)
        #cv.putText(result,       str(2*edg) + ',', (x, y - 46), cv.FONT_HERSHEY_SIMPLEX, 0.75, border, 4)
        #cv.putText(result,       str(2*edg) + ',', (x, y - 46), cv.FONT_HERSHEY_SIMPLEX, 0.75, colour, 2)
        #cv.putText(result,       str(2*cor) + ')', (x, y - 23), cv.FONT_HERSHEY_SIMPLEX, 0.75, border, 4)
        #cv.putText(result,       str(2*cor) + ')', (x, y - 23), cv.FONT_HERSHEY_SIMPLEX, 0.75, colour, 2)
        # write the filter efficiency on the image
        eff = unrecognisable.effectiveness(out)
        cv.putText(result, 'Unreco: ' + str(eff), (x, Y + 17), cv.FONT_HERSHEY_SIMPLEX, 0.75,
                   border, 4)
        cv.putText(result, 'Unreco: ' + str(eff), (x, Y + 17), cv.FONT_HERSHEY_SIMPLEX, 0.75,
                   colour, 2)
        eff = detectable.effectiveness(out)
        cv.putText(result, 'DetUnr: ' + str(eff), (x, Y + 40), cv.FONT_HERSHEY_SIMPLEX, 0.75,
                   border, 4)
        cv.putText(result, 'DetUnr: ' + str(eff), (x, Y + 40), cv.FONT_HERSHEY_SIMPLEX, 0.75,
                   colour, 2)
    return


def problem(subject):
    test = (subject-3.5)*(subject-4.5)*(subject-5.5)*(subject-8.5)*(subject-13.5)*(subject-14.5)
    if test < 0:
        return True
    return False

def common(subject):
    test = (subject-0.5)*(subject-1.5)*(subject-2.5)*(subject-3.5)*(subject-8.5)*(subject-9.5)*(subject-14.5)*(subject-15.5)
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
        return round(self.passed/self.total,5)

if __name__ == '__main__':
    source = 'yalefaces'
    # extract facial data from data set by relative path, if none given previously trained model is taken
    extract(None)
    kNames = loads(open('face_enc', 'rb').read())
    results = []
    dim = 0
    end = 0
    #train.set(filters.CUSTOM_BLUR, [-5,-3,4])
    dec = 10000000000000000000
    u1 = u2 = u3 = 0
    while abs(u1 + 4*(u2+u3)) > 3 or max(u1, max(u2, u3)) <= 0:
        u1 = float(randint(-10*dec, 10*dec)) / dec
        u2 = float(randint(-10*dec, 10*dec)) / dec
        u3 = float(randint(-10*dec, 10*dec)) / dec
    exists = os.path.exists("results.txt")
    file = open("results.txt", "a+")
    train.set(filters.CUSTOM_BLUR, [3.456302388796793, -8.851734782864286, 7.864385222319630])
    if not exists:
        file.write("Step    Eff_PrevBest F1 F2 F3    Eff_Curr f1 f2 f3\n")
        #train.set(filters.CUSTOM_BLUR, [-5.003124178597295, -2.8855210246746683, 4.02985958381214])
        #train.set(filters.CUSTOM_BLUR, [u1, u2, u3])
        train.set(filters.CUSTOM_BLUR, [9.93903629818035, 2.5065005576477706, -5.157512297774112])
    step = 1
    faces = 10
    #best = 0.00606060606060606061
    #best = 0.01515151515151515152
    best = -1
    check = 0
    rocords = np.zeros((99, 110), dtype=bool)
    for cen in range(-10,11):
        for edg in range(-10,11):
            for cor in range(-10,11):
                if cen+4*edg+4*cor < -5 or max(cen,max(edg,cor)) <= 0 or (min(cen, min(edg, cor)) >= 0 and check == 0) or not max(abs(cen), max(abs(cor), abs(edg))) > 5 or cen%2 != 0 or cor%2 != 0 or edg%2 != 0:
                    continue
                #if abs(2*cen) < 6 and abs(2*edg) < 6 and abs(2*cor) < 6:
                    #continue
                #params = (cen+5)*121 + (edg+5)*11 + cor+5
                #if  params != (5-5) * 121 + (5-3) * 11 + (5+4) \
                #and params != (5-5) * 121 + 3 * 11 + 8 \
                #and params != (5-5) * 121 + 4 * 11 + 7 \
                #and params != (5-4) * 121 + 4 * 11 + 10 \
                #and params != (5-3) * 121 + 10 * 11 + 4 \ gradient reconstruction
                #and params != (5-2) * 121 + 0 * 11 + 10 \
                #and params != (5-1) * 121 + 1 * 11 + 9 \
                #and params != (5-1) * 121 + 10 * 11 + 0 \
                #and params != (5+0) * 121 + 5 * 11 + 8 \
                #and params != (5+0) * 121 + 9 * 11 + 4 \
                #and params != (5+1) * 121 + 8 * 11 + 5 \
                #and params != (5+1) * 121 + 10 * 11 + 3 \
                #and params != (5+3) * 121 + 0 * 11 + 9 \
                #and params != (5+3) * 121 + 8 * 11 + 1 \
                #and params != 9 * 121 + 8 * 11 + 4 \
                #and params != 10 * 121 + 8 * 11 + 4 \
                #and params != 10 * 121 + 9 * 11 + 3:
                 #   continue
                filt = filters.CUSTOM_BLUR
                (data, index) = train.get(filt)
                params = list(data['params'][index])
                #cen = params[0]
                #edg = params[1]
                #cor = params[2]
                oldcen = cen
                oldedg = edg
                oldcor = cor
                params = (cen + 5) * 121 + (edg + 5) * 11 + cor + 5

                dec = 10000000000000000000
                #dev = 0.3989
                dev = 0.1
                u1 = float(randint(1,dec))/dec
                u2 = float(randint(1,dec))/dec
                value = dev * math.sin(math.pi*u1) * math.sqrt(-2*math.log(u2))
                #value = math.copysign(dev * math.sqrt(-2*math.log(dev*math.sqrt(2*math.pi)*math.fabs(x))),x)
                z = float(randint(-dec,dec))/dec
                theta = float(randint(0,dec))/dec*2*math.pi
                #cen += value*math.sqrt(1-z*z) * math.cos(theta)
                #edg += value*math.sqrt(1-z*z) * math.sin(theta)
                #cor += value*z
                train.set(filt, [cen, edg, cor])
                #train.set(filt, [2*cen,2*edg,2*cor])

                #train.train(filt)
                #if filt != filters.WEIGHTED_GREYSCALE and filt != filters.CANNY and filt != filters.BLUR and filt != filters.REPLACE:
                    #continue
                detectable = Criteria(True, False)
                unrecognisable = Criteria(None, False)
                prog = Progress('Calculating effectiveness (' + str(cen) + ',' + str(edg) + ',' + str(cor) + '):', n=11*10)
                #prog = Progress('Calculating effectiveness (' + str(2*cen) + ',' + str(2*edg) + ',' + str(2*cor) + '):', 110, 100)
                prog.initialise()
                tpos = [0]*15
                pos = [0]*15
                fpos = [0]*15
                results.append(Images(1,0).rgb)
                oldend = end
                olddim = dim
                dim = math.ceil(math.sqrt(len(results)))
                end = math.ceil(len(results) / dim)
                start = end-1
                if end > oldend:
                    start = max(0,oldend-1)
                if dim > olddim:
                    collage = np.array([])
                    start = 0
                for i in range(start, end):
                    linef = np.array([])
                    lineb = np.array([])
                    blanc = np.copy(results[i * dim])
                    blanc.fill(255)
                    for j in range(dim-1):
                        if i * dim + j < len(results)-1:
                            linef = np.concatenate((linef, results[i * dim + j]), axis=1) if linef.size else np.copy(results[i * dim + j])
                            continue
                        lineb = np.concatenate((lineb, blanc), axis=1) if lineb.size else np.copy(blanc)
                    if i < end-1:
                        line = np.concatenate((linef, results[min(len(results)-1, (i+1)*dim-1)]), axis=1) if linef.size else np.copy(results[min(len(results)-1, (i+1)*dim-1)])
                        line = np.concatenate((line, lineb), axis=1) if lineb.size else line
                        collage = np.concatenate((collage, line), axis=0) if collage.size else np.copy(line)
                for subject in range(1,16):
                    if problem(subject):# or not common(subject):
                        continue
                    for type in range(11):
                        # set subject by number and type of image by predefined constant in Images
                        image = Images(subject=subject, type=type)
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
                        if len(recognised) == 1 and recognised[0][1][:7] == 'subject':
                                i = int(recognised[0][1][7:])-1
                                pos[i] += 1
                                if out.recognisable:
                                    tpos[i] += 1
                                else:
                                    fpos[i] += 1

                        # prepare output image
                        result = cv.cvtColor(image.rgb, cv.COLOR_RGB2BGR)
                        # prepare list of all markers to be added to the image
                        markers += recognised
                        # iterate over the markers and apply them on result
                        for ((y, X, Y, x), name) in markers:
                            mark()

                        #collage = np.copy(result)
                        results[len(results)-1] = result
                        #dim = math.ceil(math.sqrt(len(results)))
                        #collage = np.array([])
                        #for i in range(math.ceil(len(results)/dim)):
                         #   line = np.copy(results[i*dim])
                          #  blanc = np.copy(line)
                           # blanc.fill(255)
                            #for j in range(1,dim):
                             #   if i*dim + j < len(results):
                              #      line = np.concatenate((line, results[i*dim + j]), axis=1)
                               #     continue
                                #line = np.concatenate((line, blanc), axis=1)
                            #collage = np.concatenate((collage, line), axis=0) if collage.size else line
                        line = np.concatenate((linef, result), axis=1) if linef.size else np.copy(result)
                        line = np.concatenate((line, lineb), axis=1) if lineb.size else line
                        cv.imshow('Frame', np.concatenate((collage, line), axis=0) if collage.size else np.copy(line))
                        cv.waitKey(1)
                        prog.update()
                prog.quit()
                avgx = 0
                avgy = 0
                for i in range(15):
                    if problem(i+1): #or not common(i+1):
                        continue
                    rel = 0.0
                    if pos[i] != 0:
                        rel = tpos[i]/pos[i]
                    avgx += fpos[i]/11/(faces-1)/faces
                    avgy += tpos[i]/11/faces
                    print(str(i+1) + ":(" + str(rel) + "," + str(tpos[i]/11) + "," + str(fpos[i]/11/(faces-1)) + "), ", end='')
                dis = avgx-avgy
                check = dis
                rocords[int(avgx*11*(faces-1)*faces), int(avgy*11*faces)] = True
                print("\b\b\nAvg.:(" + str(avgx) + "," + str(avgy) + "," + str(dis) + "), Sum: " + str(cen + 4*edg + 4*cor))
                #if dis > 0:
                    #print("Beat Random")
                file.write(str(step) + "    " + str(best) + " " + str(oldcen) + " " + str(oldedg) + " " + str(
                    oldcor) + "    " + str(dis) + " " + str(cen) + " " + str(edg) + " " + str(cor) + "\n")
                file.flush()
                step += 1
                if dis <= best:
                    train.set(filt, [oldcen, oldedg, oldcor])
                else:
                    best = dis
                    print("New Best!")
                #6.108410006583712 1.7584848962646678 -0.4735478928822906

    for (x, y), yes in np.ndenumerate(rocords):
        if yes:
            print(str(math.floor((x/11/(faces-1)/faces)*100000)/100000) + " " + str(math.floor((y/11/faces)*100000)/100000))
    key = -1
    while key != 27:
        key = cv.waitKey(1)
    cv.destroyAllWindows()
    exit(0)
