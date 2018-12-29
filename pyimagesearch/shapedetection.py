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
		# Se calculan los vertices y en base a ello se obtiene que figura es.
		if len(approx) == 3:
			shape = "Triangulo"
		elif len(approx) == 4:
			(x, y, w, h) = cv2.boundingRect(approx)
			ar = w / float(h)
 
			shape = "Cuadrado" if ar >= 0.95 and ar <= 1.05 else "Rectangulo"
		elif len(approx) == 5:
			shape = "Pentagono"
		else:
			shape = "Circulo"
		if area > 210 and area < 1000:
			status = True
			color = (0, 255, 0)
		else:
			status = False
			color = (0, 0, 255)
		return shape, color, area, status