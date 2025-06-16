import cv2
import torch
import numpy as np
from transformers import VideoMAEImageProcessor, TimesformerForVideoClassification
from collections import deque
import time
from threading import Thread

#configuración de dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Usando dispositivo:", device)

#cargar modelo y procesador con parámetros consistentes
try:
    processor = VideoMAEImageProcessor.from_pretrained("./violence_detector")
    model = TimesformerForVideoClassification.from_pretrained(
        "./violence_detector",
        id2label={0: "No violencia", 1: "Violencia"},
        num_frames=16,  #debe coincidir con train.py
        attention_type="divided_space_time"  #debe coincidir con train.py
    ).to(device)
    model.eval()
    print("Modelo cargado exitosamente. Configuración:")
    print(f"- Número de frames: {model.config.num_frames}")
    print(f"- Tipo de atención: {model.config.attention_type}")
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    exit()

#clase para captura de cámara en segundo plano (mejorada)
class IPCameraStream:
    def __init__(self, url):
        self.cap = cv2.VideoCapture(url)
        # Reducir resolución para mejorar rendimiento
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
        self.frame = None
        self.running = True
        self.thread = Thread(target=self.update, daemon=True)
        self.thread.start()
    
    def update(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                self.frame = frame
    
    def read(self):
        return self.frame
    
    def stop(self):
        self.running = False
        self.thread.join()
        self.cap.release()
        print("Cámara liberada correctamente")

# inicializar cámara IP y buffer de frames
stream = IPCameraStream("http://192.168.18.58:8080/video")
# buffer ajustado a 16 frames (consistente con entrenamiento)
frame_buffer = deque(maxlen=16)
frame_size = (224, 224)
time_threshold = 1.0  # Procesar cada 1 segundo
last_process_time = time.time()

# buffer para suavizar predicciones
violence_probs = deque(maxlen=5)  # almacena las últimas 5 probabilidades
confidence_threshold = 0.65  # umbral estricto para violencia

# estado de visualizacion
current_label = "Inicializando..."
current_color = (255, 255, 0)  # Amarillo (neutro)
last_update_time = time.time()

try:
    while True:
        current_frame = stream.read()
        if current_frame is None:
            #reintentar conexión si se pierde frame
            time.sleep(0.1)
            continue
        
        # 1. Preprocesamiento rápido y eficiente
        frame_resized = cv2.resize(current_frame, frame_size)
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        frame_buffer.append(frame_rgb)
        
        # 2. Procesamiento solo cuando tenemos suficientes frames
        current_time = time.time()
        if len(frame_buffer) == 16 and current_time - last_process_time >= time_threshold:
            try:
                # 3. Preprocesamiento con normalización explícita
                inputs = processor(
                    list(frame_buffer), 
                    return_tensors="pt",
                    do_normalize=True,  # ¡Importante!
                    do_rescale=True     # ¡Importante!
                ).to(device)
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    probs = torch.softmax(outputs.logits, dim=1)[0]
                    
                    # 4. Usar umbral de confianza estricto
                    violence_prob = probs[1].item()  # Probabilidad de violencia
                    predicted_class = 1 if violence_prob > confidence_threshold else 0
                    
                    # 5. Suavizar predicciones con buffer temporal
                    violence_probs.append(violence_prob)
                    avg_violence_prob = sum(violence_probs) / len(violence_probs)
                    
                    # 6. Determinar etiqueta final con histeresis
                    if avg_violence_prob > confidence_threshold:
                        current_label = "Violencia"
                        current_color = (0, 0, 255)  # Rojo
                    else:
                        current_label = "No violencia"
                        current_color = (0, 255, 0)  # Verde
                    
                    print(f"Predicción: {current_label} (Confianza: {avg_violence_prob:.2f})")
                    last_update_time = current_time
            
            except Exception as e:
                print(f"Error en predicción: {e}")
            
            finally:
                last_process_time = current_time
        
        # 7. Mostrar informacion en el frame
        display_text = f"Estado: {current_label}"
        cv2.putText(current_frame, display_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, current_color, 2)
        
        # 8. Indicador de actividad
        if current_time - last_update_time > 2.0:
            cv2.putText(current_frame, "Analizando...", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        cv2.imshow('Detección de Violencia', current_frame)
        
        # 9. Salida con tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Interrupción del usuario")
except Exception as e:
    print(f"Error inesperado: {e}")
finally:
    stream.stop()
    cv2.destroyAllWindows()
    print("Recursos liberados exitosamente")