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


def damageDetection(camera, time, imageSize, background, maxArea, minArea):
    #cap = cv2.VideoCapture(camera)
    # Constructor del Argparse
    #ap = argparse.ArgumentParser()
    # Se obtiene la dirección de la imagen
    #ap.add_argument("-i", "--image", required=True,
        #help="Direccion de la imagen")
    #args = vars(ap.parse_args()) 
    # Se carga la imagen
    #ret, frame = cap.read()
    if background != "white":
        frame = cv2.imread("test17.jpg")
    else:
        frame = cv2.imread("test12.jpg")
    while(True):
        # Se toma cada frame de la imagen
        cv2.imshow("Image", frame)
        resized = imutils.resize(frame, width=300)
        ratio = frame.shape[0] / float(resized.shape[0])
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
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
        cv2.drawContours(resized, cnts, -1, (0,255,0), 3)
        cv2.imshow("Image8", resized)

        sd = ShapeDetector()
        areas = []
        times = 0
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
            if area > 200:
                # Dependiendo si el area esta dentro de cierto rango dibujara
                # el contorno de la figura o un elipse.
                if status:
                    # Para dibujar el contorno se multiplican los valores de 
                    # # X y Y por el ratio en que se redujo la imagen.
                    c = c.astype("float")
                    c *= ratio
                    c = c.astype("int")
                    cv2.drawContours(frame, [c], -1, color, 2)
                    areas.append(area)
                    times += 1
                else:
                    # Para dibujar el elipse se hace de la misma manera salvo
                    # que ahora se descompone el elipse en sus coordenadas y sus
                    # acis para de esta manera multiplicarlos por el ratio.
                    (x, y), (MA, ma), angle = cv2.fitEllipse(c)
                    center = (x * ratio, y * ratio)
                    print(center)
                    print(MapD(center[0],0, 720, 0, imageSize))
                    axes = (MA * ratio, ma * ratio)
                    ellipse = (center, axes, angle)
                    cv2.ellipse(frame,ellipse,color,3)
                    # Se muestra la imagen
            cv2.imshow("Image", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            cv2.waitKey(0)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    totalAreas = np.array(areas)
    print(np.average(totalAreas))

damageDetection(0, 5, 7, "black", 500, 210)