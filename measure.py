# import the necessary packages
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2
 
def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)
 
# Constructor del ArgumentParser
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to the input image")
ap.add_argument("-w", "--width", type=float, required=True,
	help="width of the left-most object in the image (in inches)")
args = vars(ap.parse_args())

# Carga la imagen
image = cv2.imread(args["image"])
# Selecciona un ROI
r = cv2.selectROI(image)
 
# Recorta la imagen
imCrop = image[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]

# Muestra la imagen recortada
cv2.imshow("Image", imCrop)

# Cambia de tamaño la imagen
resized = imutils.resize(imCrop, width=300)
# Se obtiene el ratio del cambio de tamaño
ratio = resized.shape[0] / float(imCrop.shape[0])

# Se cambia el color a blanco y negro y se distorsiona un poco
gray = cv2.cvtColor(imCrop, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (7, 7), 0)
 
# perform edge detection, then perform a dilation + erosion to
# close gaps in between object edges
# Realiza detección de bordes y realiza una dilatación y contracción
# para eliminar huecos entre los bordes del objeto
edged = cv2.Canny(gray, 50, 100)
edged = cv2.dilate(edged, None, iterations=1)
edged = cv2.erode(edged, None, iterations=1)

# Muestra la imagen con bordes
cv2.imshow("Edged", edged)

# Encuentra los bordes en el contorno de la imagenn
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
 
(cnts, _) = contours.sort_contours(cnts)

# Cantidad de pixeles (Para esta imagen) que hay en 1 in.
pixelsPerMetric = 52.9729

# Se recorren todos los contornos
for c in cnts:
	# Si el largo no es lo suficientemente largo se ignora.
	if cv2.contourArea(c) < 100:
		continue
 
	orig = image.copy()
	box = cv2.minAreaRect(c)
	box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
	box = np.array(box, dtype="int")

	box2 = cv2.minAreaRect(c)
	box2 = cv2.cv.BoxPoints(box2) if imutils.is_cv2() else cv2.boxPoints(box2)
	box2 = np.array(box2, dtype="int")

	box = perspective.order_points(box)
	box2 = perspective.order_points(box2)

	# Se multiplica el tamaño de la caja por el ratio
	box2 = box2.astype("float")
	box2 *= ratio
	cv2.drawContours(resized, [box2.astype("int")], -1, (0, 255, 0), 2)
 
	# Se recorren las coordenadas de la caja
	for (x, y) in box:
		cv2.circle(resized, (int(x * ratio), int(y * ratio)), 5, (0, 0, 255), -1)

		(tl, tr, br, bl) = box
		(tltrX, tltrY) = midpoint(tl, tr)
		(blbrX, blbrY) = midpoint(bl, br)
	 
		(tlblX, tlblY) = midpoint(tl, bl)
		(trbrX, trbrY) = midpoint(tr, br)
	 
		# Dibuja los puntos medios en la imagen
		cv2.circle(resized, (int(tltrX * ratio), int(tltrY * ratio)), 5, (255, 0, 0), -1)
		cv2.circle(resized, (int(blbrX * ratio), int(blbrY * ratio)), 5, (255, 0, 0), -1)
		cv2.circle(resized, (int(tlblX * ratio), int(tlblY * ratio)), 5, (255, 0, 0), -1)
		cv2.circle(resized, (int(trbrX * ratio), int(trbrY * ratio)), 5, (255, 0, 0), -1)
	 
		# Dibuja lineas entre los puntos medios
		cv2.line(resized, (int(tltrX * ratio), int(tltrY * ratio)), (int(blbrX * ratio), int(blbrY * ratio)),
			(255, 0, 255), 2)
		cv2.line(resized, (int(tlblX * ratio), int(tlblY * ratio)), (int(trbrX * ratio), int(trbrY * ratio)),
			(255, 0, 255), 2)

		# Calcula la distancia eucladiana entre los puntos medios
		dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
		dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
	 
		# Si aun no se tiene la cantidad de pixeles por metrica se calcula
		if pixelsPerMetric is None:
			pixelsPerMetric = dB / args["width"]
			print(pixelsPerMetric)

		# Se calcula el tamaño del objeto
		dimA = dA / pixelsPerMetric
		dimB = dB / pixelsPerMetric
	 
		# Se dibuja en la imagen
		cv2.putText(resized, "{:.1f}in".format(dimA),
			(int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
			0.65, (255, 255, 255), 1)
		cv2.putText(resized, "{:.1f}in".format(dimB),
			(int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
			0.65, (255, 255, 255), 1)
		print("Lenght: ", "{:.1f}in".format(dimA))
		print("Width: ", "{:.1f}in".format(dimB))
		# Se muestra la imagen final
		cv2.imshow("Image", resized)
		cv2.waitKey(0)