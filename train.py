import os
import cv2
import torch
import numpy as np
import logging
from torch.utils.data import Dataset, DataLoader
from transformers import (
    VideoMAEImageProcessor,
    TimesformerForVideoClassification,
)
from tqdm import tqdm
import torch.nn as nn

# configuracion de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# hiperparametros 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 2  # reducido para compensar NUM_FRAMES=16
EPOCHS = 15     # mas epocas para aprender patrones temporales
NUM_FRAMES = 16  # mas contexto temporal
DATA_PATH = "./datasets/RWF-2000"
MODEL_SAVE_PATH = "./violence_detector"
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

# funcion de perdida con regularizacion temporal
class TemporalAwareLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        
    def forward(self, outputs, labels):
        # perdida de clasificación estandar
        loss = self.ce_loss(outputs.logits, labels)
        
        # perdida por consistencia temporal (suaviza transiciones)
        if outputs.logits.size(0) > 1:  # solo si batch_size > 1
            temporal_diff = torch.mean(torch.abs(outputs.logits[1:] - outputs.logits[:-1]))
            loss += 0.3 * temporal_diff  # peso ajustable
            
        return loss

# dataset mejorado con muestreo adaptativo
class RWFDataset(Dataset):
    def __init__(self, root_dir, processor, mode="train"):
        self.processor = processor
        self.video_paths = []
        self.labels = []
        self.mode = mode  # guardar el modo para aumentos de datos
        
        # mapeo de clases explicito
        self.class_mapping = {"NonFight": 0, "Fight": 1}
        
        # validación de estructura del dataset
        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"¡Directorio no encontrado: {root_dir}")
        
        for class_name, label in self.class_mapping.items():
            class_dir = os.path.join(root_dir, mode, class_name)
            if not os.path.exists(class_dir):
                logger.warning(f"Clase '{class_name}' no encontrada en {mode}. Ignorando...")
                continue
                
            # cargar solo videos .avi
            video_files = [
                os.path.join(class_dir, f) 
                for f in os.listdir(class_dir) 
                if f.lower().endswith('.avi')
            ]
            
            self.video_paths.extend(video_files)
            self.labels.extend([label] * len(video_files))
        
        if not self.video_paths:
            raise ValueError(f"No se encontraron videos en {os.path.join(root_dir, mode)}")

        logger.info(f"{mode}: {len(self.video_paths)} videos cargados.")

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        try:
            video_path = self.video_paths[idx]
            label = self.labels[idx]
            cap = cv2.VideoCapture(video_path)
            frames = []
            
            # tecnica de muestreo eficiente de frames
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # manejar videos demasiado cortos
            if total_frames < NUM_FRAMES:
                frames = [np.zeros((224, 224, 3), dtype=np.uint8)] * NUM_FRAMES
                cap.release()
                inputs = self.processor(frames, return_tensors="pt")
                return inputs.pixel_values.squeeze(0), torch.tensor(label, dtype=torch.long)
            
            # muestreo adaptativo: para violencia, centrar en la mitad del video
            if label == 1:  #violencia
                mid_point = total_frames // 2
                start_frame = max(0, mid_point - NUM_FRAMES // 2)
                end_frame = min(total_frames - 1, start_frame + NUM_FRAMES - 1)
                frame_indices = np.linspace(start_frame, end_frame, NUM_FRAMES, dtype=int)
            else:  # no violencia
                frame_indices = np.linspace(0, total_frames - 1, NUM_FRAMES, dtype=int)
            
            for i in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if not ret:
                    # estrategia de fallback
                    frame = frames[-1].copy() if frames else np.zeros((224, 224, 3), dtype=np.uint8)
                else:
                    frame = cv2.resize(frame, (224, 224))
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            
            cap.release()
            
            # aumento de datos: inversión temporal para violencia (solo en entrenamiento)
            if self.mode == "train" and label == 1 and np.random.rand() > 0.5:
                frames = frames[::-1]  # invertir el orden de los frames
            
            # procesamiento con el modelo
            inputs = self.processor(frames, return_tensors="pt")
            return inputs.pixel_values.squeeze(0), torch.tensor(label, dtype=torch.long)
        
        except Exception as e:
            logger.error(f"Error en {video_path}: {str(e)}")
            # return dummy data
            return torch.zeros((3, NUM_FRAMES, 224, 224)), torch.tensor(0, dtype=torch.long)

# carga del modelo con configuración mejorada
def load_model():
    logger.info("Cargando modelo y procesador...")
    processor = VideoMAEImageProcessor.from_pretrained("facebook/timesformer-base-finetuned-k400")
    
    model = TimesformerForVideoClassification.from_pretrained(
        "facebook/timesformer-base-finetuned-k400",
        num_labels=2,
        ignore_mismatched_sizes=True,
        id2label={0: "No violencia", 1: "Violencia"},
        label2id={"No violencia": 0, "Violencia": 1},
        attention_type="divided_space_time",  # Atencion espacio-temporal explicita
        num_frames=NUM_FRAMES  # crucial para consistencia
    ).to(DEVICE)
    
    # definición de capas entrenables
    trainable_layers = {
        "classifier",
        "timesformer.encoder.layer.11",  # ultima capa espacial
        "timesformer.temporal_encoder",  # codificador temporal completo
        "timesformer.embeddings.temporal_embedding"  # embeddings temporales
    }
    
    logger.info("Configurando capas entrenables...")
    for name, param in model.named_parameters():
        if any(trainable_layer in name for trainable_layer in trainable_layers):
            param.requires_grad = True
            logger.debug(f"Descongelado: {name}")
        else:
            param.requires_grad = False
    
    return model, processor

# entrenamiento con evaluación integrada
def train_model(model, train_loader, val_loader):
    # usar nuestra nueva función de perdida
    criterion = TemporalAwareLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=5e-5,
        weight_decay=0.01
    )
    
    best_val_acc = 0.0
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)  # usar nuestra nueva perdida
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Prevenir exploding gradients
            optimizer.step()
            
            train_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})
        
        #validacion despues de cada epoca
        val_loss, val_acc = evaluate_model(model, val_loader, criterion)
        logger.info(
            f"Epoch {epoch+1}/{EPOCHS} | "
            f"Train Loss: {train_loss/len(train_loader):.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.2f}%"
        )
        
        # guardar el mejor modelo
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save_pretrained(MODEL_SAVE_PATH)
            processor.save_pretrained(MODEL_SAVE_PATH)
            logger.info(f"¡Modelo guardado en {MODEL_SAVE_PATH} (Acc: {val_acc:.2f}%)!")

def evaluate_model(model, val_loader, criterion):
    model.eval()
    val_loss, correct = 0.0, 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            val_loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.logits, 1)
            correct += (predicted == labels).sum().item()
    
    val_loss /= len(val_loader)
    val_acc = 100 * correct / len(val_loader.dataset)
    return val_loss, val_acc

if __name__ == "__main__":
    try:
        # carga de datos y modelo
        model, processor = load_model()
        train_dataset = RWFDataset(DATA_PATH, processor, mode="train")
        val_dataset = RWFDataset(DATA_PATH, processor, mode="val")
        
        # dataLoaders optimizados
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            pin_memory=True,
            num_workers=4,  # paralelizar carga de datos
            persistent_workers=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            pin_memory=True,
            num_workers=2
        )
        
        logger.info(f"Videos de entrenamiento: {len(train_dataset)}")
        logger.info(f"Videos de validación: {len(val_dataset)}")
        logger.info(f"Tamaño de lote: {BATCH_SIZE}, Frames por video: {NUM_FRAMES}")
        
        # entrenamiento
        train_model(model, train_loader, val_loader)
        
    except Exception as e:
        logger.error(f"Error crítico: {str(e)}")
        raise