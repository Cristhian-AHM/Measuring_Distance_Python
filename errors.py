from pyimagesearch.shapedetection import ShapeDetector
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2

# Punto medio
def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

# Medidas correctas (Top, Bottom, Mid)
steps = [6.3, 5.0, 5.4]
# Umbral
threshold = 0.5

# Constructor del Argparse
ap = argparse.ArgumentParser()
# Se obtiene la dirección de la imagen
ap.add_argument("-i", "--image", required=True,
	help="Direccion de la imagen")
args = vars(ap.parse_args()) 
# Se carga la imagen
image = cv2.imread(args["image"])
resized = imutils.resize(image, width=300)
ratio = image.shape[0] / float(resized.shape[0])

"""hsv_image = cv2.cvtColor(resized, cv2.COLOR_RGB2HSV)
h, s, v = cv2.split(hsv_image)
columns, rows, k = hsv_image.shape
print(columns)
print(rows)
cv2.imshow('Brightness', v)
#clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(16,16))
#v = clahe.apply(s)

for x in range(columns):
	for y in range(rows):
		if v[x,y] > 240 and v[x,y] < 242:
			v[x,y] = 0

hsv_image = cv2.merge([h, s, v])
hsv_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)
cv2.imshow("HSV Correction", hsv_image)
cv2.waitKey(0)"""
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (7, 7), 0)
thresh = cv2.adaptiveThreshold(blurred,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
           cv2.THRESH_BINARY,11,2)
# Se muestra la imagen
cv2.imshow("Imag", thresh)
# Se buscan los contornos
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

sd = ShapeDetector()
areas = []
# loop over the contours
for c in cnts:
	# compute the center of the contour, then detect the name of the
	# shape using only the contour
	M = cv2.moments(c)
	cX = int((M["m10"] / M["m00"]) * ratio)
	cY = int((M["m01"] / M["m00"]) * ratio)
	shape, color, area, status = sd.detect(c)

	if area > 200:
		# multiply the contour (x, y)-coordinates by the resize ratio,
		# then draw the contours and the name of the shape on the image
		if status:
			c = c.astype("float")
			c *= ratio
			c = c.astype("int")
			cv2.drawContours(image, [c], -1, color, 2)
			areas.append(area)
		else:
			"""
			box = cv2.minAreaRect(c)
			box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
			box = box.astype("float")
			box *= ratio
			box = box.astype("int")
			box = np.array(box, dtype="int")
			cv2.drawContours(image, [box], -1, color, 2)"""
			(x, y), (MA, ma), angle = cv2.fitEllipse(c)
			center = (x * ratio, y * ratio)
			axes = (MA * ratio, ma * ratio)
			ellipse = (center, axes, angle)
			#newEllipse = np.array(ellipse)
			#newEllipse *= ratio
			"""ellipse = c.astype("float")
			ellipse *= ratio
			ellipse = c.astype("int")"""
			print(ellipse)
			cv2.ellipse(image,ellipse,color,3)
		# show the output image
		cv2.imshow("Image", image)
		cv2.waitKey(0)

totalAreas = np.array(areas)
print(np.average(totalAreas))