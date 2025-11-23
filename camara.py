import cv2
import tensorflow as tf
import numpy as np

#Iniciar la captura del video

cap = cv2.VideoCapture(1)


while True:
    ret, frame = cap.read()
    if ret==False:
        break

    cv2.inshow("Frame", frame)

    #Sali con la tecla ESC codgio ascii 27
    if cv2.waitKety(1) & 0xFF == 27:
        break

#Liberar la camara y cerrar todas las ventanas
cap.release()
cv2.destroyAllWindows()
