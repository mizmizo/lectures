#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cv2
import sys

def encode():
    argv = sys.argv
    argc = len(argv)

    if (argc != 2):
        print "put movie file name!"
        quit()

    cap = cv2.VideoCapture(argv[1])

    ret, frame = cap.read()
    if (len(frame.shape) == 2):
        print "input BGR movie!"
        quit()

    while(cap.isOpened()):
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    encode()
