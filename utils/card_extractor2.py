from skimage import exposure
import numpy as np
import argparse
import os
import cv2

image_list = os.listdir('/media/son/Repository2/V.DO/V_Caption/remember')

for image in image_list:

    img = cv2.imread(os.path.join('/media/son/Repository2/V.DO/V_Caption/remember/', image))
    img = cv2.resize(img, (1280, int(1280*img.shape[0]/img.shape[1])))
    orig = img.copy()
    ratio = img.shape[1] / 1280
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # gray = cv2.bilateralFilter(gray, 11, 17, 17)
    # edged = cv2.Canny(gray,d 30, 200)
    # cv2.imshow("Game Boy Screen", edged)
    # cv2.waitKey(0)
    # blur = cv2.GaussianBlur(gray, (1, 1), 1000)
    # flag, thresh = cv2.threshold(blur, 120, 255, cv2.THRESH_BINARY)

    gray = cv2.medianBlur(gray, 9)
    thresh = cv2.Canny(gray, 10, 25)

    (_, cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # (_, cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]
    screenCnt = None

    # loop over our contours
    for i, c in enumerate(cnts):
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        # if our approximated contour has four points, then
        # we can assume that we have found our screen
        if len(approx) == 4:
            screenCnt = approx
            break

    # now that we have our screen contour, we need to determine
    # the top-left, top-right, bottom-right, and bottom-left
    # points so that we can later warp the image -- we'll start
    # by reshaping our contour to be our finals and initializing
    # our output rectangle in top-left, top-right, bottom-right,
    # and bottom-left order
    pts = screenCnt.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")

    # the top-left point has the smallest sum whereas the
    # bottom-right has the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # compute the difference between the points -- the top-right
    # will have the minumum difference and the bottom-left will
    # have the maximum difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # multiply the rectangle by the original ratio
    rect *= ratio

    # now that we have our rectangle of points, let's compute
    # the width of our new image
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))

    # ...and now for the height of our new image
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))

    # take the maximum of the width and height values to reach
    # our final dimensions
    maxWidth = max(int(widthA), int(widthB))
    maxHeight = max(int(heightA), int(heightB))

    # construct our destination points which will be used to
    # map the screen to a top-down, "birds eye" view
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # calculate the perspective transform matrix and warp
    # the perspective to grab the screen
    M = cv2.getPerspectiveTransform(rect, dst)
    warp = cv2.warpPerspective(orig, M, (maxWidth, maxHeight))

    # cv2.drawContours(img, [screenCnt], -1, (0, 255, 0), 3)
    cv2.imshow("Game Boy Screen", img)
    cv2.imshow("Game Boy Screen_warped", warp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()