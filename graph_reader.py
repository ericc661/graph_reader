'''
Eric Chen
5/9/20

GraphReader class: uses CV techniques to identify nodes and edges in an image
of a graph.

TODO: morph after node removal, then work on identifying edges?
TODO: maybe morph for state detection but use thresholded, not morphed version for
        detecting the numbers? bc morph can tend to erase numbers. The morph may
        be very important for the line detection because of the edge noise. And
        implement functions for all of these.
TODO: enforce thresholding for every image - maybe after morph operators
TODO: figure out how to get circles fully surround the node
TODO: input validation for if nodes are labeled the same thing
idea: way to detect self-loops: use less strict circle detection, if we have two
        intersecting circles then we probably have that the smaller circle is a
        self-loop? if one circle is completely inside another, then it might just
        be a number
TODO: automate the min size for a circle - want to exclude numbers/labels but detect
        self-loops as well as states
TODO: try on hand-drawn graphs
TODO: when erasing nodes: account for different line thicknesses of the nodes


notes:
circles are more centered on thresholded images!



process:
-read in grayscale image
-perform inversion if needed so we can have a black backround
-threshold the image
-detect states/nodes on thresholded image
    - this involves the labeling part
-remove nodes and perform morph operators to leave just edges on the graph
-with purely edges: detect self-loops as well as regular transitions - straight
    as well as curved arrows
-at some point (potentially before), identify labels by looking in the nodes,
   finding the contour in there, and taking the bounding rect of that contour ->
   pass into mnist
-with all this information, construct/store full graph info: nodes and edges
'''

import numpy as np
import sys
import cv2

class GraphReader(object):

    '''
    summary: returns 2d list where each element is (x, y, r) of circle in image
    requires: image_gray to be the image to detect circles in
    effects: outputs 2d list, each element is an [x, y, radius] list.
    '''
    def find_circles(self, image_gray):
        assert len(image_gray.shape)==2

        # lower line params: img res/accum res, min dist b/t centers, then
        #   two parameters for canny edge detection
        circles = cv2.HoughCircles(image_gray, cv2.HOUGH_GRADIENT, \
                                    1, 50, param1=80, param2=40)

        #TODO: increase radius size to be safe?
        return np.round(circles[0, :]).astype('int')

    '''
    summary: draws colored circles on original image
    requires: orig to be the image to duplicate then draw circles on
    effects: returns a BGR image of the original but with circles drawn in red
    '''
    def draw_circles(self, orig):
        circles = self.find_circles(orig)
        out = cv2.cvtColor(orig, cv2.COLOR_GRAY2RGB)
        for (x, y, r) in circles:
            cv2.circle(out, (x, y), r, (0, 0, 255), 2)
        return out

    '''
    summary: blacks out nodes to just leave edges, extremely similar to draw_circles
    requires: orig to be the image to duplicate then erase nodes of
    effects: returns a
    '''
    def erase_nodes(self, orig):
        circles = self.find_circles(orig)
        out = orig.copy() # we want output to be single-channel, binary
        for (x, y, r) in circles:
            # NOTE: we must increase the radius to fully erase the edge
            cv2.circle(out, (x, y), r+10, 0, -1)
        return out

    '''
    summary: takes in input image and returns inverted copy if the image
      is majority-white. must be called first before passing image into other
      functions.
    '''
    def validate_input(self, img):
        image_validated = img.copy()
        if (np.average(img)) > 128:
            print('White background detected. Inverting image...')
            image_validated = 255-image_validated
        return image_validated

    '''
    shows image in new window
    '''
    def show(self, image, title):
        cv2.imshow(title, image)
        while cv2.waitKey(15) < 0: pass


class ContourUtility(object):

    # all these functions take in contours as defined by CV2

    # returns area of contour
    @staticmethod
    def get_area(contour):
        return cv2.contourArea(contour)

    # returns (x, y) coordinate of centroid of contour (i.e. (col, row))
    @staticmethod
    def get_cxy(contour):
        M = cv2.moments(contour)

        # return a center that won't be inside a state if m00 is 0
        if M['m00'] == 0:
            return (-1, -1) # to signal m00 was 0
        else:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            return (cx, cy)


def main():


    if len(sys.argv) != 2:
        print('usage: ' + sys.argv[0] + ' input_image')
        exit(1)

    gr = GraphReader()
    img = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
    img = gr.validate_input(img)

    # find circles on orig image
    gr.show(img, 'grayscale image')
    gr.show(gr.draw_circles(img), 'grayscale image with circles')

    # try with thresholding
    _, img_thresholded = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)
    gr.show(img_thresholded, 'thresholded image')
    gr.show(gr.draw_circles(img_thresholded), 'thresholded image w circles')

    # after nodes identified, try identifying the node labels with findContours
    #   inside the location

    node_info = gr.find_circles(img_thresholded)
    print(node_info)

    contours, hierarchy = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    # check if each contour is smaller than size of circle and contained in each circle:

    for i in range(len(contours)):
        area = ContourUtility.get_area(contours[i])
        cxy = ContourUtility.get_cxy(contours[i])
        print('This contour has area {} and is centered at {}'.format(area, cxy))

        bg = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        cv2.drawContours(bg, contours, i, (0, 0, 255), thickness=2)
        gr.show(bg, 'original w contours drawn')



    '''
    for i in range(len(contours)):
        # get contour's area and centroid
        cnt_area = ContourUtility.get_area(contours[i])
        cnt_cxy = np.array(ContourUtility.get_cxy(contours[i]))

        for node in circles:
            node_area = (np.pi)*(node[2] ** 2)
            node_cxy = node[0:2]

            # if the centroid of the contour is within the node/circle
            if np.linalg.norm(cnt_cxy-node_cxy) < node[2]:
                bg = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                cv2.drawContours(bg, contours, i, (0, 0, 255), thickness=2)
                gr.show(bg, "this contour's centroid is in the middle of a node")
    '''

    # we need to search thru the contours and find the contours that represent
    # node labels: just take the circle/node info, store it, then if the centroid of
    # a contour is within a radius's dis

    # AREA OF THE CONTOUR has to be less than some factor to differentiate the
    # outline of the whole thing or the node itself, from just the appropriate
    # label



    # now try removing nodes on thresholded img
    gr.show(gr.erase_nodes(img_thresholded), 'thresholded img w nodes erased')

    '''
    # try with thresholding AND morph operators:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    img_morphed = cv2.morphologyEx(img_thresholded, cv2.MORPH_OPEN, kernel)
    '''




if __name__ == '__main__':
    main()
