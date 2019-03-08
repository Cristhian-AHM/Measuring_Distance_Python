from Malla.pyimagesearch.shapedetection import ShapeDetector
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2
import os

def MapD(value, leftMin, leftMax, rightMin, rightMax):
    # Figure out how 'wide' each range is
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin

    # Convert the left range into a 0-1 range (float)
    valueScaled = float(value - leftMin) / float(leftSpan)

    # Convert the 0-1 range into a value in the right range.
    return rightMin + (valueScaled * rightSpan)


def damageDetection(camera, image, imageSize, background):
    #cap = cv2.VideoCapture(camera)
    # Constructor del Argparse
    #ap = argparse.ArgumentParser()
    # Se obtiene la dirección de la imagen
    #ap.add_argument("-i", "--image", required=True,
        #help="Direccion de la imagen")
    #args = vars(ap.parse_args()) 
    # Se carga la imagen
    #ret, frame = cap.read()
    frame = cv2.imread(image)
    # Se toma cada frame de la imagen
    cv2.imshow("Image", frame)
    resized = imutils.resize(frame, width=300)
    ratio = frame.shape[0] / float(resized.shape[0])

    lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
    cv2.imshow("LAB", lab)

    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)

    limg = cv2.merge((cl,a,b))
    cv2.imshow('limg', limg)

    final_image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    gray = cv2.cvtColor(final_image, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Gray", gray)
    #gray = cv2.equalizeHist(gray)
    #cv2.imshow("Gray Equalize", gray)
    #
    #cv2.imshow("Blurred", blurred)
    if background != "white":
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        ret,thresh = cv2.threshold(blurred,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    else:
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        thresh = cv2.adaptiveThreshold(blurred,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                cv2.THRESH_BINARY,11,2)
    #ret,thresh = cv2.threshold(blurred,200,255,cv2.THRESH_BINARY)
    #ret3,th3 = cv2.threshold(blurred,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #cv2.imshow("Blurred", blurred)
    cv2.imshow("Thresh", thresh)
    # Se buscan los contorno
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    sd = ShapeDetector()
    areas = []
    times = 0

    for c in cnts:
        area = sd.detectArea(c)
        areas.append(area)

    mediana = np.median(areas)
    media = np.mean(areas)

    maxValue = mediana + media
    for value in areas:
        if value > maxValue:
            areas.remove(value)
    maxArea = np.amax(areas)
    minArea = maxArea / 2
    # Se recorren todos los contornos encontrados.
    for c in cnts:
        # Se calcula el momento de los contornos.
        M = cv2.moments(c)
        if int(M["m00"]) == 0:
            continue
        #print(M["mm00"])
        cX = int((M["m10"] / M["m00"]) * ratio)
        cY = int((M["m01"] / M["m00"]) * ratio)
        shape, color, area, status = sd.detect(c, maxArea, minArea)
        cv2.drawContours(resized, c, -1, (0,255,0), 3)
        cv2.imshow("C", resized)

        if area > minArea:
            # Dependiendo si el area esta dentro de cierto rango dibujara
            # el contorno de la figura o un elipse.
            if status:
                # Para dibujar el contorno se multiplican los valores de 
                # # X y Y por el ratio en que se redujo la imagen.
                c = c.astype("float")
                c *= ratio
                c = c.astype("int")
                cv2.drawContours(frame, [c], -1, color, 2)
                times += 1
            else:
                # Para dibujar el elipse se hace de la misma manera salvo
                # que ahora se descompone el elipse en sus coordenadas y sus
                # acis para de esta manera multiplicarlos por el ratio.
                (x, y), (MA, ma), angle = cv2.fitEllipse(c)
                center = (x * ratio, y * ratio)
                print(MapD(center[0],0, 720, 0, imageSize))
                axes = (MA * ratio, ma * ratio)
                ellipse = (center, axes, angle)
                cv2.ellipse(frame,ellipse,color,3)
                 # Se muestra la imagen
        cv2.imshow("Image", frame)
        cv2.waitKey(0)
    cv2.destroyAllWindows()

def menu():
    option = "0"
    while option != "3":
        os.system('cls')
        print("1.- Fondo Blanco")
        print("2.- Fondo Negro")
        print("3.- Salir")
        option = input("Introduce una opción: ")
        if option == "1":
            damageDetection(0, "Malla/images/test12.jpg", 7, "white")
        elif option == "2":
            damageDetection(0, "Malla/images/test17.jpg", 7, "black")
        #damageDetection(0, 5, 7, "white", 500, 200) #Black background 2000 - 1000
                                              #White background 500 - 210
          