import numpy as np
import os
import cv2
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import math

image_list = os.listdir('/media/son/Repository2/V.DO/V_Caption/remember')
image_list.sort()

for image in image_list:

    img = cv2.imread(os.path.join('/media/son/Repository2/V.DO/V_Caption/remember/', image))
    img = cv2.resize(img, (1280, int(1280*img.shape[0]/img.shape[1])))

    # Prepocess
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (1, 1), 1000)
    flag, thresh = cv2.threshold(blur, 120, 255, cv2.THRESH_BINARY)
    # gray = cv2.bilateralFilter(gray, 11, 17, 17)
    # thresh = cv2.Canny(gray, 30, 200)

    # Find contours
    _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    # Select long perimeters only
    # perimeters = [cv2.arcLength(contours[i], True) for i in range(len(contours))]
    # listindex = [i for i in range(15) if perimeters[i] > perimeters[0] / 2]
    # numcards = len(listindex)

    mask = np.zeros_like(img)
    # Show image
    imgcont = img.copy()
    # [cv2.drawContours(imgcont, [contours[i]], 0, (0, 255, 0), 5) for i in listindex]
    # [cv2.drawContours(mask, [contours[i]], 0, (0, 255, 0), 5) for i in listindex]
    cv2.drawContours(mask, contours, 0, (0, 255, 0), 5)

    out = np.zeros_like(img)  # Extract out the object and place into output image
    out[mask == 255] = img[mask == 255]

    # Now crop
    (x, y, _) = np.where(mask == 255)
    (topx, topy) = (np.min(x), np.min(y))
    (bottomx, bottomy) = (np.max(x), np.max(y))
    out = img[topx:bottomx + 1, topy:bottomy + 1]
    out = cv2.resize(out, (1280, 720))

    cv2.imshow('result',out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # # now that we have our screen contour, we need to determine
    # # the top-left, top-right, bottom-right, and bottom-left
    # # points so that we can later warp the image -- we'll start
    # # by reshaping our contour to be our finals and initializing
    # # our output rectangle in top-left, top-right, bottom-right,
    # # and bottom-left order
    # pts = contours.reshape(4, 2)
    # rect = np.zeros((4, 2), dtype="float32")
    #
    # # the top-left point has the smallest sum whereas the
    # # bottom-right has the largest sum
    # s = pts.sum(axis=1)
    # rect[0] = pts[np.argmin(s)]
    # rect[2] = pts[np.argmax(s)]
    #
    # # compute the difference between the points -- the top-right
    # # will have the minumum difference and the bottom-left will
    # # have the maximum difference
    # diff = np.diff(pts, axis=1)
    # rect[1] = pts[np.argmin(diff)]
    # rect[3] = pts[np.argmax(diff)]
    #
    # # multiply the rectangle by the original ratio
    # rect *= ratio