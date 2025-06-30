import cv2
import torch
import numpy as np
from transformers import VideoMAEImageProcessor, TimesformerForVideoClassification
from collections import deque
import time
from threading import Thread, Lock
from flask import Flask, render_template, Response, jsonify
import queue # se usara una cola para una comunicación mas segura entre hilos

# configuracion del modelo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Usando dispositivo:", device)
model = None
processor = None
try:
    processor = VideoMAEImageProcessor.from_pretrained("./violence_detector")
    model = TimesformerForVideoClassification.from_pretrained(
        "./violence_detector",
        id2label={0: "No violencia", 1: "Violencia"},
        num_frames=16,
        attention_type="divided_space_time"
    ).to(device)
    model.eval()
    print("Modelo cargado exitosamente.")
except Exception as e:
    print(f"Error al cargar el modelo: {e}")


# variables globales y estructuras de datos compartidas 
detection_results = {
    "label": "Inicializando...",
    "confidence": 0.0,
    "is_violence": False,
}
results_lock = Lock()

# usamos una cola para pasar frames del hilo de la camara al de deteccion
# maxsize=1 asegura que el hilo de detección siempre trabaje con el frame mas reciente
# y no se acumulen frames viejos si la detección es más lenta que la camara
frame_queue = queue.Queue(maxsize=1) 
latest_raw_frame = None
frame_lock = Lock()


# hilo 1: lector de camara 
def camera_reader_thread(url):
    
    # este hilo es el unico que se conecta a la camara
    
    global latest_raw_frame
    cap = None
    while True:
        try:
            if cap is None or not cap.isOpened():
                print(f"Intentando conectar a la cámara en {url}...")
                cap = cv2.VideoCapture(url)
                if not cap.isOpened():
                    print(f"Error: No se pudo abrir el stream. Reintentando en 5 segundos...")
                    time.sleep(5)
                    continue
                print("Conexión con la cámara establecida.")
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

            ret, frame = cap.read()
            if not ret:
                print("Se perdió la conexión con la cámara, reintentando...")
                cap.release()
                cap = None # forzamos a que se reconecte en el siguiente ciclo
                time.sleep(5)
                continue
            
            # poner el frame crudo en la cola para el hilo de deteccion
            if not frame_queue.full():
                frame_queue.put(frame.copy()) # .copy() es importatnte
            
            # guardar el frame crudo para el hilo web
            with frame_lock:
                global latest_raw_frame
                latest_raw_frame = frame.copy()

        except Exception as e:
            print(f"Error en el hilo de la cámara: {e}")
            cap.release()
            cap = None
            time.sleep(5)


# hilo 2: deteccion de violencia
def violence_detection_thread():
    
    # este hilo consume frames de la cola, los procesa y actualiza los resultados
    
    global detection_results
    if not model or not processor:
        print("El modelo no está cargado. El hilo de detección no se iniciará.")
        return
        
    frame_buffer = deque(maxlen=16)
    frame_size = (224, 224)
    time_threshold = 1.0  # procesar cada segundo
    last_process_time = time.time()
    violence_probs = deque(maxlen=5) # suavizado de resultados
    confidence_threshold = 0.65

    while True:
        try:
            # espera bloqueante hasta que haya un frame disponible en la cola
            raw_frame = frame_queue.get() 

            frame_resized = cv2.resize(raw_frame, frame_size)
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            frame_buffer.append(frame_rgb)
            
            if len(frame_buffer) == 16 and time.time() - last_process_time >= time_threshold:
                inputs = processor(
                    list(frame_buffer), 
                    return_tensors="pt"
                ).to(device)
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    probs = torch.softmax(outputs.logits, dim=1)[0]
                    violence_prob = probs[1].item()
                    violence_probs.append(violence_prob)
                    avg_violence_prob = sum(violence_probs) / len(violence_probs)
                    
                    with results_lock:
                        is_violence = avg_violence_prob > confidence_threshold
                        detection_results["label"] = "Violencia" if is_violence else "No Violencia"
                        detection_results["confidence"] = avg_violence_prob
                        detection_results["is_violence"] = is_violence
                    
                    # print(f"DEBUG: Predicción -> {detection_results['label']} (Conf: {detection_results['confidence']:.2f})")

                last_process_time = time.time()
        except queue.Empty:
            # esto no debería pasar con get() bloqueante, pero es una buena practica
            time.sleep(0.01)
            continue
        except Exception as e:
            print(f"ERROR en el hilo de predicción: {e}")


# app flask
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            frame_to_send = None
            with frame_lock:
                if latest_raw_frame is not None:
                    # codificamos a JPEG justo antes de enviarlo
                    ret, buffer = cv2.imencode('.jpg', latest_raw_frame)
                    if ret:
                        frame_to_send = buffer.tobytes()
            
            if frame_to_send:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_to_send + b'\r\n')
            
            # pausa para no saturar el servidor. 30 fps es suficiente.
            time.sleep(1/30) 

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    with results_lock:
        return jsonify(detection_results.copy())

if __name__ == '__main__':
    camera_url = "http://172.25.99.112:8080/video"  #  http://192.168.18.58:8080/video
    
    # iniciar el hilo que lee la camara
    cam_thread = Thread(target=camera_reader_thread, args=(camera_url,), daemon=True)
    cam_thread.start()
    
    # iniciar el hilo para la detección de IA
    det_thread = Thread(target=violence_detection_thread, daemon=True)
    det_thread.start()
    
    # flask ya es multi-hilo por defecto cuando se le pasa threaded=True
    # no es necesario ponerlo en su propio hilo
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True, use_reloader=False)
