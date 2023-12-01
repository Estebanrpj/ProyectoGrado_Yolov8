import cv2
import time
from datetime import datetime
from ultralytics import YOLO
import numpy as np
from sort import Sort
import matplotlib.path as mplPath
import pandas as pd

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
Linea_Azul = [(310,100),(470,100)]
Linea_Verde = [(300,120),(480,120)]
Linea_Roja = [(290,145),(500,145)]
Linea_Fucsia = [(275,175),(515,175)]
Linea_Amarilla = [(255,210),(525,210)]

Cruzo_Linea_Azul={}
Cruzo_Linea_Verde={}
Cruzo_Linea_Roja={}
Cruzo_Linea_Fucsia = {}
Cruzo_Linea_Amarilla = {}

vels_prom = {}
vels_prom_int = {}

arr_velocidades = []

#Revisar el calculo de las variables de acuerdo a nuestro caso
FPS_video= 30
KM_factor= 3.6
FPS_Latencia= 7

def sum_arr(arr):

    sum = 0
    for i in arr:
        sum = sum + i
    return(sum)

def distancia_eucladiana(punto1:tuple, punto2:tuple):
    x1, y1 = punto1
    x2, y2 = punto2
    distancia= ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    return distancia

def Calculo_Vel_Prom(track_id):

    tiempoAV = (Cruzo_Linea_Verde[track_id]["time"] - Cruzo_Linea_Azul[track_id]["time"]).total_seconds()
    tiempoVR = (Cruzo_Linea_Roja[track_id]["time"] - Cruzo_Linea_Verde[track_id]["time"]).total_seconds()
    tiempoRF = (Cruzo_Linea_Fucsia[track_id]["time"] - Cruzo_Linea_Roja[track_id]["time"]).total_seconds()
    tiempoFAm = (Cruzo_Linea_Amarilla[track_id]["time"] - Cruzo_Linea_Fucsia[track_id]["time"]).total_seconds()

    distanciaAV = distancia_eucladiana(Cruzo_Linea_Verde[track_id]["point"],Cruzo_Linea_Azul[track_id]["point"])
    distanciaVR = distancia_eucladiana(Cruzo_Linea_Roja[track_id]["point"],Cruzo_Linea_Verde[track_id]["point"])
    distanciaRF = distancia_eucladiana(Cruzo_Linea_Fucsia[track_id]["point"],Cruzo_Linea_Roja[track_id]["point"])
    distanciaFAm = distancia_eucladiana(Cruzo_Linea_Amarilla[track_id]["point"],Cruzo_Linea_Fucsia[track_id]["point"])

    velocidadAV = round((distanciaAV / (tiempoAV * FPS_video)) * (KM_factor * FPS_Latencia), 2)
    velocidadVR = round((distanciaVR / (tiempoVR * FPS_video)) * (KM_factor * FPS_Latencia), 2)
    velocidadRF = round((distanciaFAm / (tiempoRF * FPS_video)) * (KM_factor * FPS_Latencia), 2)
    velocidadFAm = round((distanciaFAm / (tiempoFAm * FPS_video)) * (KM_factor * FPS_Latencia), 2)

    return round((velocidadAV + velocidadVR + velocidadRF + velocidadFAm)/4 , 2)

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

def detector(cap: object, output_path: str = "resultados/output_video.avi", excel_path: str = "resultados/detected_objects.xlsx"):

    model = load_model()
    tracker = load_tracker()
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 10, (frame_width, frame_height))
    #columns = ["Frame", "Detections", "Average_Speed"]
    columns = ["Frame", "Detections"]
    results_df = pd.DataFrame(columns=columns)

    frame_count = 0
    
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
                if is_valid_detection(xc, yc):
                    detections +=1
                    cv2.circle(img=frame, center=(int(xc), int(yc)), radius=5, color=(0,255,0), thickness=-1)
                    cv2.rectangle(img=frame, pt1=(int(xmin), int(ymin)), pt2=(int(xmax), int(ymax)), color=(0, 255, 0), thickness=2)
                
                if track_id not in Cruzo_Linea_Azul:
                    Cruzo_Azul = (Linea_Azul[1][0]-Linea_Azul[0][0]) * (yc-Linea_Azul[0][1]) - (Linea_Azul[1][1] - Linea_Azul[0][1]) * (xc - Linea_Azul[0][0])
                    if Cruzo_Azul>=0:
                        Cruzo_Linea_Azul[track_id]= {
                            "time":datetime.now(),
                            "point":(xc , yc)
                        }

                elif track_id not in Cruzo_Linea_Verde:
                    Cruzo_Verde = (Linea_Verde[1][0]-Linea_Verde[0][0]) * (yc-Linea_Verde[0][1]) - (Linea_Verde[1][1] - Linea_Verde[0][1]) * (xc - Linea_Verde[0][0])
                    if Cruzo_Verde>=0:
                        Cruzo_Linea_Verde[track_id]= {
                            "time":datetime.now(),
                            "point":(xc , yc)
                        }

                elif track_id not in Cruzo_Linea_Roja:
                    Cruzo_Roja = (Linea_Roja[1][0]-Linea_Roja[0][0]) * (yc-Linea_Roja[0][1]) - (Linea_Roja[1][1] - Linea_Roja[0][1]) * (xc - Linea_Roja[0][0])
                    if Cruzo_Roja>=0:
                        Cruzo_Linea_Roja[track_id]= {
                            "time":datetime.now(),
                            "point":(xc , yc)
                        }
                
                elif track_id not in Cruzo_Linea_Fucsia:
                    Cruzo_Fucsia = (Linea_Fucsia[1][0]-Linea_Fucsia[0][0]) * (yc-Linea_Fucsia[0][1]) - (Linea_Fucsia[1][1] - Linea_Fucsia[0][1]) * (xc - Linea_Fucsia[0][0])
                    if Cruzo_Fucsia>=0:
                        Cruzo_Linea_Fucsia[track_id]= {
                            "time":datetime.now(),
                            "point":(xc , yc)
                        }

                elif track_id not in Cruzo_Linea_Amarilla:
                    Cruzo_Amarilla = (Linea_Amarilla[1][0]-Linea_Amarilla[0][0]) * (yc-Linea_Amarilla[0][1]) - (Linea_Amarilla[1][1] - Linea_Amarilla[0][1]) * (xc - Linea_Amarilla[0][0])
                    if Cruzo_Amarilla>=0:
                        Cruzo_Linea_Amarilla[track_id]= {
                            "time":datetime.now(),
                            "point":(xc , yc)
                        }

                        vel_prom = Calculo_Vel_Prom(track_id)
                        vels_prom[track_id] = f"{vel_prom} Km/h"
                        vels_prom_int[track_id] = vel_prom
                        #arr_velocidades[track_id]= int(vels_prom_int[track_id])
                        arr_velocidades.append(vels_prom_int[track_id])
                        #arr_velocidades.insert(detections,int(vels_prom_int[track_id]))

                if len(arr_velocidades)==0:
                    vel_prom_tot=0
                else:
                    vel_prom_tot = sum_arr(arr_velocidades)/len(arr_velocidades)
                    vel_prom_tot = round(vel_prom_tot,2)

                if track_id in vels_prom:
                    cv2.putText(frame, vels_prom[track_id], (int(xmin),int(ymin)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36,255,12), 2)  #Colocar el texto con la velocidad 

                # Almacenar resultados en el DataFrame
                row_data = {
                "Frame": frame_count,
                "Detections": detections,
                "Average_Speed": vel_prom_tot
                }


                results_df = pd.concat([results_df, pd.DataFrame([row_data])], ignore_index=True)

        cv2.line(frame,Linea_Azul[0],Linea_Azul[1],(255,0,0),3)
        cv2.line(frame,Linea_Verde[0],Linea_Verde[1],(0,255,0),3)
        cv2.line(frame,Linea_Roja[0],Linea_Roja[1],(0,0,255),3)
        
        cv2.line(frame,Linea_Fucsia[0],Linea_Fucsia[1],(255,0,255),3)
        cv2.line(frame,Linea_Amarilla[0],Linea_Amarilla[1],(0,233,255),3)

        cv2.putText(img=frame, text=f"detecciones: {detections}", org=(50, 100), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=2, color=(0,0,0), thickness=3)
        cv2.polylines(img=frame, pts=[ZONE], isClosed=True, color=(0,0,255), thickness=4)

        

        cv2.imshow("frame", frame)
        out.write(frame)
        frame_count += 1

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    results_df.to_excel(excel_path, index=False)


if __name__ == '__main__':
    cap = cv2.VideoCapture("data/video.mp4")
    output_path = "resultados/output_video.mp4"
    excel_path = "resultados/detected_objects.xlsx"
    detector(cap, output_path, excel_path)
