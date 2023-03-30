import cv2 as cv

class Images:
    CENTERLIGHT, \
        GLASSES, \
        HAPPY, \
        LEFTLIGHT, \
        NOGLASSES, \
        NORMAL, \
        RIGHTLIGHT, \
        SAD, \
        SLEEPY, \
        SURPRISED, \
        WINK = range(11)

    def __init__(self, subject=None, type=None, rgb=None):
        self.subject = subject
        self.type = type
        if rgb is None:
            self.rgb = cv.cvtColor(cv.VideoCapture(self.fp()).read()[1], cv.COLOR_BGR2RGB)
        else:
            self.rgb = rgb

    def subjectStr(self):
        if self.subject is None or self.subject < 1 or self.subject > 15:
            return ''
        return 'subject' + ('0' if (self.subject < 10) else '') + str(self.subject)

    def typeStr(self):
        return {self.CENTERLIGHT:   'centerlight',
                self.GLASSES:       'glasses',
                self.HAPPY:         'happy',
                self.LEFTLIGHT:     'leftlight',
                self.NOGLASSES:     'noglasses',
                self.NORMAL:        'normal',
                self.RIGHTLIGHT:    'rightlight',
                self.SAD:           'sad',
                self.SLEEPY:        'sleepy',
                self.SURPRISED:     'surprised',
                self.WINK:          'wink'}.get(self.type)

    def fp(self):
        return 'yalefaces/' + self.subjectStr() + '/' + self.typeStr()
