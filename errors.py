from pyimagesearch.shapedetection import ShapeDetector
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2

def MapD(value, leftMin, leftMax, rightMin, rightMax):
    # Figure out how 'wide' each range is
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin

    # Convert the left range into a 0-1 range (float)
    valueScaled = float(value - leftMin) / float(leftSpan)

    # Convert the 0-1 range into a value in the right range.
    return rightMin + (valueScaled * rightSpan)
# Constructor del Argparse
ap = argparse.ArgumentParser()
# Se obtiene la dirección de la imagen
ap.add_argument("-i", "--image", required=True,
	help="Direccion de la imagen")
args = vars(ap.parse_args()) 
# Se carga la imagen
image = cv2.imread(args["image"])
print(image.shape)
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
cv2.waitKey(0)
# Se buscan los contornos
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

sd = ShapeDetector()
areas = []
# Se recorren todos los contornos encontrados.
for c in cnts:
	# Se calcula el momento de los contornos.
	M = cv2.moments(c)
	cX = int((M["m10"] / M["m00"]) * ratio)
	cY = int((M["m01"] / M["m00"]) * ratio)
	shape, color, area, status = sd.detect(c)
	if area > 200:
		# Dependiendo si el area esta dentro de cierto rango dibujara
		# el contorno de la figura o un elipse.
		if status:
			# Para dibujar el contorno se multiplican los valores de 
			# X y Y por el ratio en que se redujo la imagen.
			c = c.astype("float")
			c *= ratio
			c = c.astype("int")
			cv2.drawContours(image, [c], -1, color, 2)
			areas.append(area)
		else:
			# Para dibujar el elipse se hace de la misma manera salvo
			# que ahora se descompone el elipse en sus coordenadas y sus
			# acis para de esta manera multiplicarlos por el ratio.
			(x, y), (MA, ma), angle = cv2.fitEllipse(c)
			center = (x * ratio, y * ratio)
			print(center)
			print(MapD(center[0],0, 720, 0, 5))
			axes = (MA * ratio, ma * ratio)
			ellipse = (center, axes, angle)
			cv2.ellipse(image,ellipse,color,3)
		# Se muestra la imagen
		cv2.imshow("Image", image)
		cv2.waitKey(0)



totalAreas = np.array(areas)
print(np.average(totalAreas))