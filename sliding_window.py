# import the necessary packages
# from utils.datasets import pyramid
# from utils.datasets import sliding_window
import argparse
import time
import cv2
import imutils
import numpy as np
import matplotlib.pyplot as plt


# find the main ROI of image
def find_roi(image):
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	ret, thresh = cv2.threshold(gray,50,255,0)
	if imutils.is_cv2() or imutils.is_cv4():
		contours, _ = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	elif imutils.is_cv3():
		img, contours, _ = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

	approx = cv2.approxPolyDP(contours[0],50,True)
	mask = cv2.drawContours(image*0,approx, -1, (255), 15)
    
    # if our approximated contour has four points, then we
    # can assume that we have found our piece of paper
	if len(approx) == 4:
		return approx
	
	area=[]
	for cnt in contours:
		area.append(cv2.contourArea(cnt))
	contour_n=np.argmax(np.array(area))
	roi = cv2.boundingRect(contours[contour_n]) #x,y,w,h
	
	return roi

def sliding_window(image, stepSize, windowSize):
	# slide a window across the image
	for y in range(0, image.shape[0], stepSize):
		for x in range(0, image.shape[1], stepSize):
			# yield the current window
			yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

def pyramid(image, scale=1.5, minSize=(30, 30)):
	# yield the original image
	yield image

	# keep looping over the pyramid
	while True:
		# compute the new dimensions of the image and resize it
		w = int(image.shape[1] / scale)
		image = imutils.resize(image, width=w)

		# if the resized image does not meet the supplied minimum
		# size, then stop constructing the pyramid
		if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
			break

		# yield the next image in the pyramid
		yield image

if __name__ == '__main__':
	# construct the argument parser and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--image", default=r'.\data\images\remap_dummy_4p_l.png', help="Path to the image")
	args = vars(ap.parse_args())

	# load the image and define the window width and height
	image = cv2.imread(args["image"])
	(winW, winH) = (320, 320)

	# find the main ROI of image and check if it is valid
	roi = find_roi(image)
	if len(roi) == 4:
		if issubclass(type(roi), tuple):
			image = image[roi[1]:roi[3], roi[0]:roi[2]]
		else:
			image = image[roi[1,0,1]:roi[3,0,1], roi[1,0,0]:roi[3,0,0]]
	else:
		print('ROI not found')
		exit()

	fig = plt.figure()    
    
    # loop over the image pyramid
	for resized in pyramid(image, scale=1.5):
		# loop over the sliding window for each layer of the pyramid
		for (x, y, window) in sliding_window(resized, stepSize=int(winW*0.75), windowSize=(winW, winH)):
		# if the window does not meet our desired window size, ignore it
			if window.shape[0] != winH or window.shape[1] != winW:
				continue
			# THIS IS WHERE YOU WOULD PROCESS YOUR WINDOW, SUCH AS APPLYING A
			# MACHINE LEARNING CLASSIFIER TO CLASSIFY THE CONTENTS OF THE
			# WINDOW

			# since we do not have a classifier, we'll just draw the window
			clone = resized.copy()
			cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 20)
			# cv2.imwrite("./data/images/sliding_window_" + str(time.time()) + ".png", clone)
			# cv2.imshow("Window", clone)
			# cv2.waitKey(1)
			# time.sleep(0.025)

			
			plt.imshow(clone)
			plt.pause(0.1)
			fig.clear()
