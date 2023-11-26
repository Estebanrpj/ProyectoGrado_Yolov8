import cv2
import time
from datetime import datetime
from ultralytics import YOLO
import numpy as np
from sort import Sort
import matplotlib.path as mplPath

ZONE = np.array([
    [338, 76],
    [388, 79],
    [460, 85],
    [497, 148],
    [557, 264],
    [656, 433],
    [439, 428],
    [279, 404],
    [179, 422],
    [141, 421],
    [189, 334],
    [236, 237],
    [285, 157],
    [326, 78],
    [337, 60],
])

# Variables para determinar la velocidad de los vehiculos 
Linea_Azul = [(280,150),(500,150)]
Linea_Verde = [(260,200),(520,200)]
Linea_Roja = [(230,250),(550,250)]

Cruzo_Linea_Azul={}
Cruzo_Linea_Verde={}
Cruzo_Linea_Roja={}

vels_prom = {}

#Revisar el calculo de las variables de acuerdo a nuestro caso
FPS_video= 30
KM_factor= 3.6
FPS_Latencia= 7

def distancia_eucladiana(punto1:tuple, punto2:tuple):
    x1, y1 = punto1
    x2, y2 = punto2
    distancia= ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    return distancia

def Calculo_Vel_Prom(track_id):
    try:
        tiempoAV = (Cruzo_Linea_Verde[track_id]["time"] - Cruzo_Linea_Azul[track_id]["time"]).total_seconds()
    except Exception as e:
        print(f"Error inesperado: {e}")

    tiempoVR = (Cruzo_Linea_Roja[track_id]["time"] - Cruzo_Linea_Verde[track_id]["time"]).total_seconds()

    distanciaAV = distancia_eucladiana(Cruzo_Linea_Verde[track_id]["point"],Cruzo_Linea_Azul[track_id]["point"])
    distanciaVR = distancia_eucladiana(Cruzo_Linea_Roja[track_id]["point"],Cruzo_Linea_Verde[track_id]["point"])

    velocidadAV = round((distanciaAV / (tiempoAV * FPS_video)) * (KM_factor * FPS_Latencia), 2)
    velocidadVR = round((distanciaVR / (tiempoVR * FPS_video)) * (KM_factor * FPS_Latencia), 2)

    return round((velocidadAV + velocidadVR) / 2, 2)

def load_model():
    model = YOLO("yolov8x.pt")
    return model

def load_tracker():
    tracker = Sort()
    return tracker

def get_center(xmin, ymin, xmax, ymax):
    center = ((xmin + xmax)//2, (ymin + ymax)//2)
    return center

def is_valid_detection(xc, yc):
    return mplPath.Path(ZONE).contains_point((xc, yc))

def detector(cap: object):

    model = load_model()
    tracker = load_tracker()
    
    while cap.isOpened():
        status, frame = cap.read()

        if not status:
            break

        preds = model(frame, stream = True)
        detections = 0
        #preds = model(frame, conf = 0.50)
        for res in preds:
            filtered_indices = np.where(np.isin(res.boxes.cls.cpu().numpy(), [2,3,5,7]) & (res.boxes.conf.cpu().numpy()>0.1))[0]
            boxes = res.boxes.xyxy.cpu().numpy()[filtered_indices].astype(int)
            tracks = tracker.update(boxes)

            for xmin, ymin, xmax, ymax, track_id in tracks:
                xc, yc = get_center(xmin, ymin, xmax, ymax)
                xcv, ycv = int((xmin + xmax) / 2), ymax

                if is_valid_detection(xc, yc):
                    detections +=1
                    #class_id = boxes[5]
                    #print ("cls", class_id)

                    cv2.circle(img=frame, center=(int(xc), int(yc)), radius=5, color=(0,255,0), thickness=-1)
                    cv2.circle(img=frame, center=(int(xcv), int(ycv)), radius=5, color=(255,0,0), thickness=-1)
                    cv2.rectangle(img=frame, pt1=(int(xmin), int(ymin)), pt2=(int(xmax), int(ymax)), color=(0, 255, 0), thickness=2)
                
                    if track_id not in Cruzo_Linea_Azul:
                        Cruzo_Azul = (Linea_Azul[1][0]-Linea_Azul[0][0]) * (ycv-Linea_Azul[0][1]) - (Linea_Azul[1][1] - Linea_Azul[0][1]) * (xcv - Linea_Azul[0][0])
                        if Cruzo_Azul>=0:
                            Cruzo_Linea_Azul[track_id]= {
                                "time":datetime.now(),
                                "point":(xcv , ycv)
                            }

                    elif track_id not in Cruzo_Linea_Verde and track_id in Cruzo_Linea_Azul:
                        Cruzo_Verde = (Linea_Verde[1][0]-Linea_Verde[0][0]) * (ycv-Linea_Verde[0][1]) - (Linea_Verde[1][1] - Linea_Verde[0][1]) * (xcv - Linea_Verde[0][0])
                        if Cruzo_Verde>=0:
                            Cruzo_Linea_Verde[track_id]= {
                                "time":datetime.now(),
                                "point":(xcv , ycv)
                            }

                    elif track_id not in Cruzo_Linea_Roja and track_id in Cruzo_Linea_Verde:
                        Cruzo_Roja = (Linea_Roja[1][0]-Linea_Roja[0][0]) * (ycv-Linea_Roja[0][1]) - (Linea_Roja[1][1] - Linea_Roja[0][1]) * (xcv - Linea_Roja[0][0])
                        if Cruzo_Roja>=0:
                            Cruzo_Linea_Roja[track_id]= {
                                "time":datetime.now(),
                                "point":(xcv , ycv)
                            }

                            vel_prom = Calculo_Vel_Prom(track_id)
                            vels_prom[track_id] = f"{vel_prom} Km/h"

                    if track_id in vels_prom:
                        cv2.putText(frame, vels_prom[track_id], (int(xmin),int(ymin)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36,255,12), 2)  #Colocar el texto con la velocidad 

                #dist_centr= np.sqrt(xc+yc)

        #frame = preds[0].plot()

        cv2.putText(img=frame, text=f"detecciones: {detections}", org=(50, 100), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=2, color=(0,0,0), thickness=3)
        cv2.polylines(img=frame, pts=[ZONE], isClosed=True, color=(0,0,255), thickness=4)

        cv2.line(frame,Linea_Azul[0],Linea_Azul[1],(255,0,0),3)
        cv2.line(frame,Linea_Verde[0],Linea_Verde[1],(0,255,0),3)
        cv2.line(frame,Linea_Roja[0],Linea_Roja[1],(0,0,255),3)

        cv2.imshow("frame", frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()


if __name__ == '__main__':
    cap = cv2.VideoCapture("data/full_video.mp4")
    detector(cap)
