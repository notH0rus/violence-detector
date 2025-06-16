import torch
from transformers import TimesformerForVideoClassification, VideoMAEImageProcessor

class ViolenceDetector:
    def __init__(self, device="cuda"):
        self.device = device
        
        # 1. Carga del modelo con configuración mejorada
        self.model = TimesformerForVideoClassification.from_pretrained(
            "facebook/timesformer-base-finetuned-k400",
            num_labels=2,  # Clasificación binaria
            ignore_mismatched_sizes=True,
            id2label={0: "No violencia", 1: "Violencia"},
            label2id={"No violencia": 0, "Violencia": 1},
            attention_type="divided_space_time"  # Habilita atención espacio-temporal explícita
        ).to(device)
        
        self._freeze_layers()
        
        # 2. Procesador con normalización consistente
        self.processor = VideoMAEImageProcessor.from_pretrained(
            "facebook/timesformer-base-finetuned-k400",
            do_normalize=True,
            do_resize=True,
            size={"shortest_edge": 224}
        )

    def _freeze_layers(self):
        """Estrategia de descongelación optimizada para video"""
        # Capas a descongelar (ajustables)
        trainable_layers = {
            "classifier",  # Capa de clasificación
            "timesformer.encoder.layer.11",  # Última capa espacial
            "timesformer.temporal_encoder",  # Codificador temporal completo
            "timesformer.embeddings.temporal_embedding"  # Embeddings temporales
        }
        
        for name, param in self.model.named_parameters():
            param.requires_grad = any(trainable_layer in name for trainable_layer in trainable_layers)

    def predict(self, frames, confidence_threshold=0.7):
        """
        Predicción mejorada con manejo temporal
        
        Args:
            frames: Lista de frames (np.array/PIL.Image) de longitud 8-16
            confidence_threshold: Umbral mínimo para considerar violencia (0-1)
            
        Returns:
            tuple: (clase_predicha, probabilidad_de_violencia, logits_raw)
        """
        # 1. Preprocesamiento consistente con el entrenamiento
        inputs = self.processor(
            frames, 
            return_tensors="pt",
            do_normalize=True,  # Asegura normalización
            do_rescale=True     # Asegura reescalado
        ).to(self.device)
        
        # 2. Inferencia con captura de información temporal
        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)
            
            # Obtenemos tanto logits como atenciones
            logits = outputs.logits
            attentions = outputs.attentions[-1]  # Atenciones de la última capa
            
        # 3. Cálculo de probabilidades mejorado
        probs = torch.softmax(logits, dim=1)[0]  # Softmax para 2 clases
        prob_violence = probs[1].item()  # Probabilidad específica de violencia
        
        # 4. Predicción con umbral de confianza
        predicted_class = 1 if prob_violence > confidence_threshold else 0
        
        # 5. Análisis de atención temporal (debug)
        temporal_attention = attentions.mean(dim=1)[:, 0, 1:]  # Atención CLS sobre tiempo
        
        return {
            "class": predicted_class,
            "probability": prob_violence,
            "logits": logits.cpu().numpy(),
            "attention": temporal_attention.cpu().numpy()
        }