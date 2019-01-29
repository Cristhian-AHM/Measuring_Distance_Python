from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2
import threading
import time

"""def printit():
  threading.Timer(5.0, printit).start()
  print("Hello, World!")"""

# Punto medio
def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

# Medidas correctas (Top, Bottom, Mid)
steps = [6.7, 6.7, 6.0]
# Umbral
threshold = 0.5

#printit()
cap = cv2.VideoCapture(0)
# Constructor del Argparse
ap = argparse.ArgumentParser()
# Se obtiene el largo del objeto de referencia
ap.add_argument("-w", "--width", type=float, required=True,
    help="Largo del objeto mas a la izquierda (en inches)")

args = vars(ap.parse_args()) 

while(True):
    # Se toma cada frame de la imagen
    ret, frame = cap.read()

    # Se convierte a escala de grises y se le aplica una distorsión.
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    
    # Se aplica la detección de bordes
    edged = cv2.Canny(gray, 50, 100)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)
    # Se muestra la imagen
    cv2.imshow("Edges", edged)

    # Se buscan los contornos
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # Se ordenan los contornos obtenidos de Izquierda a derecha
    (cnts, _) = contours.sort_contours(cnts)
    # Se inicializa el objeto de referencia
    refObj = None

    sizeCnts = 5000
    # Se recorren todos los contornos
    for c in cnts:

        # Si el area del objeto es muy pequeña lo ignora
        if cv2.contourArea(c) < sizeCnts:
            continue


        # Se obtiene el area del objeto
        box = cv2.minAreaRect(c)
        box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
        box = np.array(box, dtype="int")
     
        # Se ordenan los puntos del contorno en tal manera que 
        # aparezcan en el siguiente orden: Esquina superior izquierda,
        # esquina superior derecha, esquina inferior izquierda y esquina
        # inferior derecha.
        box = perspective.order_points(box)
        box2 = perspective.order_points(box)

        # Se calcula el centro del objeto
        cX = box[0, 0]
        cY = np.average(box[:, 1])

        # Si es el primer contorno que se examina (El mas a la izquierda) 
        # entonces se toma como el objeto de referencia
        if refObj is None:
            # Se obtienen los puntos del contorno y se calculan los 
            # puntos medios entre las esquinas
            (tl, tr, br, bl) = box
            (tlblX, tlblY) = midpoint(tl, bl)
            (trbrX, trbrY) = midpoint(tr, br)

            # Se calcula la distancia eucladiana y se genera el objeto de referencia
            D = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
            refObj = (box, (tr[0], cY), D / args["width"])
            dX = tr[0]
            dY = np.average(box[:, 1])
            sizeCnts = 30000
            continue

        # Se dibujan los contornos de la imagen
        orig = frame.copy()
        cv2.drawContours(orig, [box.astype("int")], -1, (255, 0, 0), 2)
        cv2.drawContours(orig, [refObj[0].astype("int")], -1, (255, 0, 0), 2)
     
        # Se agrupan las coordenadas para incluir el centro del objeto
        box_copy = box2
        box[0] = box_copy[1]
        box[1] = box_copy[0]
        box[2] = box_copy[3]
        box[3] = box_copy[2]

        refCoords = np.vstack([refObj[0], refObj[1]])
        objCoords = np.vstack([box, (cX, cY)])
        # Se recorren todas las coordenadas
        index = 0

        for ((xA, yA), (xB, yB)) in zip(refCoords, objCoords):
            # Se dibujan circulos en las coordenas correspondientes
            # y se unen con una linea
            
            if(tr[0] == xA and tr[1] == yA or br[0] == xA and br[1] == yA or dX == xA and dY == yA):
                
                # Se calcula la distancia que existe entre los objetos
                D = dist.euclidean((xA, yA), (xB, yB)) / refObj[2]

                # Se verifica si las medidas estan dentro del rango
                if(D >= steps[index] - threshold and D <= steps[index] + threshold):
                    color = (0, 255, 0)
                else:
                    color = (0, 0, 255)
                    print("Medida esperada: ", steps[index], " ± ", threshold)
                    print("Medida obtenida: ", D)
                index+=1

                cv2.circle(orig, (int(xA), int(yA)), 5, color, -1)
                cv2.circle(orig, (int(xB), int(yB)), 5, color, -1)
                cv2.line(orig, (int(xA), int(yA)), (int(xB), int(yB)),
                    color, 2)
     
                # Se convierten las medidas de pixeles a inches
                (mX, mY) = midpoint((xA, yA), (xB, yB))
                cv2.putText(orig, "{:.1f} in".format(D), (int(mX), int(mY - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 2)
            else:
                continue
        # Muestra la imagen final
        cv2.imshow("Result", orig)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()