#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cv2
import sys

def make():
    print "input File name"
    name = raw_input()
    print "input flame length"
    length = input()
    label = np.zeros(length)
    cont = True

    while cont:
        print "input blood flame Number"
        print "Start:"
        start = input()
        print "End:"
        end = input()
        for i in xrange(end + 1 - start):
            label[i + start] = 1
        print "continue? input True or False"
        while 1:
            try:
                tmp = input()
                if(tmp == True or tmp == False):
                    cont = tmp
                    break
                else:
                    print "input True or False!"
            except:
                print "input True or False!"
    print "output array:" + str(label)
    np.save(name, label)


if __name__ == '__main__':
    make()
