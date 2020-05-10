'''
Eric Chen
3/4/2020

Script to experiment with thresholding, morph. operators, contours, etc. on a
directed graph created by GraphViz.
'''

import numpy as np
import cv2

def main():
    '''
    # read in image as grayscale
    source = cv2.imread('testgraphs/testgraph1.png', cv2.IMREAD_GRAYSCALE)
    source = 255-source # invert this particular input to make contours white


    #source = cv2.imread('contour_test.png', cv2.IMREAD_GRAYSCALE)
    cv2.imshow('original image inverted', source)
    while cv2.waitKey(15) < 0: pass

    thresh = 50
    _, img_thresholded = cv2.threshold(source, thresh, 255, cv2.THRESH_BINARY)
    cv2.imshow('image thresholded', img_thresholded)
    while cv2.waitKey(15) < 0: pass

    # findContours actually outputs like this despite what documentation says
    contours, hierarchy = cv2.findContours(img_thresholded, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    # redraw contours to make sure we have it right
    for i in range(len(contours)):
        bg = np.zeros_like(img_thresholded)
        cv2.drawContours(bg, contours, i, (255, 255, 255), thickness=2)
        #cv2.imshow('drawing contours', bg)
        #while cv2.waitKey(15) < 0: pass
    '''


    has_circles = cv2.imread('contour_test.png', cv2.IMREAD_GRAYSCALE)
    out = cv2.cvtColor(has_circles, cv2.COLOR_GRAY2RGB)
    cv2.imshow('this img has fucking circles', has_circles)
    while cv2.waitKey(15) < 0: pass

    #just_circle = bg = np.zeros_like(img_thresholded)
    circles = cv2.HoughCircles(has_circles, cv2.HOUGH_GRADIENT, 1, 20, param1=30, param2=15)
    circles_rounded = np.round(circles).astype('int')
    circles_flattened = circles_rounded[0, :] # get rid of the outer brackets -> 2D

    for (x, y, r) in circles_flattened:
        cv2.circle(out, (x, y), r, (0, 0, 255), 2)

    cv2.imshow('img with circles outlined', out)
    while cv2.waitKey(15) < 0: pass


    '''
    approx = cv2.approxPolyDP(contours[1], 0.01*cv2.arcLength(contours[1], True), True)
    print(approx)
    bg = np.zeros_like(img_thresholded)
    cv2.drawContours(bg, [approx], -1, (255, 255, 255), thickness=2)
    cv2.imshow('approximated contour 5', bg)
    while cv2.waitKey(15) < 0: pass
    print(len(approx))
    '''


    # intermediate goal: recognize basic circles. The approx of the big shapes
    # will be TOO big.

    # the approach of counting size of the approx only works assuming there are
    # basic shapes in the thing. Hough circle transform to detect circles?

    # put contours on hold for now
    # https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_contours/py_table_of_contents_contours/py_table_of_contents_contours.html
    # get familiar with all contour stuff, then start figuring out how to do shape recognition



if __name__ == ('__main__'):
    main()
