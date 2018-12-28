import cv2

class ShapeDetector:
	def __init__(self):
		pass

	# c es el contorno de la figura que intentamos reconocer.
	# Para intentar hacer detección de figuras se usara aproximación por contornos.
	def detect(self, c):	
		shape = "Sin identificar"
		peri = cv2.arcLength(c, True)
		area = cv2.contourArea(c)
		approx = cv2.approxPolyDP(c, 0.04 * peri, True)
				# if the shape is a triangle, it will have 3 vertices
		if len(approx) == 3:
			shape = "Triangulo"
		# if the shape has 4 vertices, it is either a square or
		# a rectangle
		elif len(approx) == 4:
			# compute the bounding box of the contour and use the
			# bounding box to compute the aspect ratio
			(x, y, w, h) = cv2.boundingRect(approx)
			ar = w / float(h)
 
			# a square will have an aspect ratio that is approximately
			# equal to one, otherwise, the shape is a rectangle
			shape = "Cuadrado" if ar >= 0.95 and ar <= 1.05 else "Rectangulo"
		# if the shape is a pentagon, it will have 5 vertices
		elif len(approx) == 5:
			shape = "Pentagono"
		# otherwise, we assume the shape is a circle
		else:
			shape = "Circulo"
		# return the name of the shape
		if area > 210 and area < 1000:
			status = True
			color = (0, 255, 0)
		else:
			status = False
			color = (0, 0, 255)
		return shape, color, area, status