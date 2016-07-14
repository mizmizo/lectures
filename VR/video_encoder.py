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
    frame = cv2.resize(frame, (160, 90))
    cv2.imshow('frame', frame)

    out_data = frame.reshape(1, 1, frame.shape[0] * frame.shape[1] * frame.shape[2])[0]
    cnt = 0
    init_flag = False
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (160, 90))
            cv2.imshow('frame', frame)
            if init_flag:
                out_data = frame.reshape(1, 1, frame.shape[0] * frame.shape[1] * frame.shape[2])[0]
                init_flag = False
            else:
                out_data = np.concatenate((out_data,frame.reshape(1, 1, frame.shape[0] * frame.shape[1] * frame.shape[2])[0]),axis=0)
        else:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if (len(out_data) > 1800):
            print "save data %d: " % cnt + str(out_data.shape)
            out_name = argv[1][argv[1].rfind("/") + 1 : argv[1].rfind(".")] + "_" + str(cnt) + ".npy"
            np.save(out_name, out_data)
            cnt += 1
            init_flag = True

    print "save data %d: " % cnt + str(out_data.shape)
    out_name = argv[1][argv[1].rfind("/") + 1 : argv[1].rfind(".")] + "_" + str(cnt) + ".npy"
    np.save(out_name, out_data)
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    encode()
