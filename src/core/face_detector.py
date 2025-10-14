import cv2
import dlib
import numpy as np
import urllib.request
import os
from collections import deque
from typing import List, Tuple, Optional
import time

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

class RobustFaceDetector:
    """Detector facial com múltiplas estratégias"""
    
    def __init__(self, config):
        self.config = config
        self.face_history = deque(maxlen=10)
        self.confidence_threshold = config.hardware.confidence_threshold
        self.lost_face_counter = 0
        self.max_lost_frames = 10
        self.tracking_quality_score = 0.0
        
        # ROI dinâmica
        self.current_roi = None
        self.roi_expansion_factor = 1.2
        
        # Filtros de face por tamanho
        self.min_face_size = 50
        self.max_face_size = 400
        
        # Inicializar detectores
        self._init_detectors()
    
    def _init_detectors(self):
        """Inicializa os detectores faciais"""
        # Detector DNN
        self._init_dnn_detector()
        
        # Detector dlib para landmarks
        self._init_dlib_detector()
        
        # Detector MediaPipe (se disponível)
        if MEDIAPIPE_AVAILABLE:
            self._init_mediapipe_detector()
    
    def _init_dnn_detector(self):
        """Inicializa detector DNN OpenCV"""
        # URLs dos modelos
        self.prototxt_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
        self.model_url = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
        
        self.prototxt_path = "models/deploy.prototxt"
        self.model_path = "models/res10_300x300_ssd_iter_140000.caffemodel"
        
        self._download_model_files()
        
        try:
            self.dnn_net = cv2.dnn.readNetFromCaffe(self.prototxt_path, self.model_path)
            print("✅ Detector DNN carregado com sucesso")
        except Exception as e:
            print(f"⚠️ Erro ao carregar detector DNN: {e}")
            self.dnn_net = None
    
    def _init_dlib_detector(self):
        """Inicializa detector dlib para landmarks"""
        self.landmarks_url = "https://github.com/italojs/facial-landmarks-recognition/raw/master/shape_predictor_68_face_landmarks.dat"
        self.landmarks_path = "models/shape_predictor_68_face_landmarks.dat"
        
        if not os.path.exists(self.landmarks_path):
            print("Baixando modelo de landmarks...")
            try:
                os.makedirs("models", exist_ok=True)
                urllib.request.urlretrieve(self.landmarks_url, self.landmarks_path)
                print("✅ Modelo de landmarks baixado")
            except Exception as e:
                print(f"⚠️ Erro ao baixar landmarks: {e}")
        
        try:
            self.predictor = dlib.shape_predictor(self.landmarks_path)
            print("✅ Preditor de landmarks carregado")
        except Exception as e:
            print(f"⚠️ Erro ao carregar landmarks: {e}")
            self.predictor = None
    
    def _init_mediapipe_detector(self):
        """Inicializa detector MediaPipe"""
        if MEDIAPIPE_AVAILABLE:
            self.mp_face_detection = mp.solutions.face_detection
            self.mp_face_detector = self.mp_face_detection.FaceDetection(
                model_selection=0, min_detection_confidence=0.5
            )
            print("✅ Detector MediaPipe inicializado")
        else:
            self.mp_face_detector = None
    
    def _download_model_files(self):
        """Baixa arquivos de modelo se necessário"""
        os.makedirs("models", exist_ok=True)
        
        files_to_download = [
            (self.prototxt_url, self.prototxt_path),
            (self.model_url, self.model_path)
        ]
        
        for url, filename in files_to_download:
            if not os.path.exists(filename):
                print(f"Baixando {filename}...")
                try:
                    urllib.request.urlretrieve(url, filename)
                    print(f"✅ {filename} baixado")
                except Exception as e:
                    print(f"⚠️ Erro ao baixar {filename}: {e}")
    
    def detect_faces_dnn(self, frame) -> Tuple[List, List]:
        """Detecta faces usando DNN OpenCV"""
        if self.dnn_net is None:
            return [], []
        
        h, w = frame.shape[:2]
        
        # Criar blob
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 
            1.0, (300, 300), 
            (104.0, 177.0, 123.0)
        )
        
        self.dnn_net.setInput(blob)
        detections = self.dnn_net.forward()
        
        faces = []
        confidences = []
        
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence > self.confidence_threshold:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x, y, x1, y1 = box.astype("int")
                
                # Validar coordenadas
                x = max(0, x)
                y = max(0, y)
                x1 = min(w, x1)
                y1 = min(h, y1)
                
                width = x1 - x
                height = y1 - y
                
                if width > 0 and height > 0:
                    faces.append((x, y, width, height))
                    confidences.append(float(confidence))
        
        return faces, confidences
    
    def detect_faces_mediapipe(self, frame) -> Tuple[List, List]:
        """Detecta faces usando MediaPipe"""
        if not MEDIAPIPE_AVAILABLE or self.mp_face_detector is None:
            return [], []
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.mp_face_detector.process(rgb_frame)
        
        faces = []
        confidences = []
        
        if results.detections:
            h, w = frame.shape[:2]
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                
                # Validar coordenadas
                x = max(0, x)
                y = max(0, y)
                width = min(width, w - x)
                height = min(height, h - y)
                
                if width > 0 and height > 0:
                    faces.append((x, y, width, height))
                    confidences.append(detection.score[0])
        
        return faces, confidences
    
    def get_landmarks(self, gray_frame, face_rect) -> Optional[np.ndarray]:
        """Obtém landmarks faciais usando dlib"""
        if self.predictor is None:
            return None
        
        x, y, w, h = face_rect
        dlib_rect = dlib.rectangle(x, y, x + w, y + h)
        landmarks = self.predictor(gray_frame, dlib_rect)
        
        landmarks_points = []
        for i in range(68):
            x_point = landmarks.part(i).x
            y_point = landmarks.part(i).y
            landmarks_points.append((x_point, y_point))
        
        return np.array(landmarks_points)
    
    def update_face_history(self, faces, confidences):
        """Atualiza histórico de faces detectadas"""
        if faces:
            quality = self.calculate_detection_quality(faces, confidences)
            self.tracking_quality_score = quality
            
            self.face_history.append({
                'faces': faces,
                'confidences': confidences,
                'timestamp': time.time(),
                'quality': quality
            })
            self.lost_face_counter = 0
        else:
            self.lost_face_counter += 1
    
    def calculate_detection_quality(self, faces, confidences) -> float:
        """Calcula qualidade da detecção atual"""
        if not faces:
            return 0.0
        
        quality_scores = []
        for face, conf in zip(faces, confidences):
            x, y, w, h = face
            size_score = min(1.0, (w * h) / (200 * 200))
            conf_score = min(1.0, conf)
            quality_scores.append(size_score * conf_score)
        
        return max(quality_scores) if quality_scores else 0.0
    
    def get_best_face(self) -> Tuple[Optional[Tuple], Optional[float]]:
        """Retorna a melhor face baseada no histórico"""
        if not self.face_history:
            return None, None
        
        # Se perdeu a face por muitos frames
        if self.lost_face_counter > self.max_lost_frames:
            return None, None
        
        # Pega a detecção mais recente com qualidade
        recent_detection = self.face_history[-1]
        if recent_detection['faces']:
            faces = recent_detection['faces']
            confidences = recent_detection['confidences']
            
            # Filtra faces por tamanho
            valid_faces = []
            valid_confidences = []
            for face, conf in zip(faces, confidences):
                x, y, w, h = face
                if (self.min_face_size <= w <= self.max_face_size and 
                    self.min_face_size <= h <= self.max_face_size):
                    valid_faces.append(face)
                    valid_confidences.append(conf)
            
            if valid_faces:
                # Retorna face com maior confiança
                best_idx = max(range(len(valid_confidences)), 
                             key=lambda i: valid_confidences[i])
                return valid_faces[best_idx], valid_confidences[best_idx]
        
        # Se não tem face válida atual, tenta predição
        return self.predict_face_location()
    
    def predict_face_location(self) -> Tuple[Optional[Tuple], Optional[float]]:
        """Prediz localização da face baseada no histórico"""
        if len(self.face_history) < 2:
            return None, None
        
        recent_faces = [h for h in self.face_history if h['faces']][-2:]
        if len(recent_faces) < 2:
            return None, None
        
        # Calcula velocidade de movimento
        face1 = recent_faces[-2]['faces'][0]
        face2 = recent_faces[-1]['faces'][0]
        
        dx = face2[0] - face1[0]
        dy = face2[1] - face1[1]
        
        # Prediz próxima posição
        predicted_x = face2[0] + dx
        predicted_y = face2[1] + dy
        predicted_face = (predicted_x, predicted_y, face2[2], face2[3])
        
        return predicted_face, 0.3  # Confiança baixa para predição
    
    def detect_faces(self, frame) -> Tuple[List, List]:
        """Método principal para detecção de faces"""
        method = self.config.algorithm.face_detection_method
        
        if method == "dnn":
            return self.detect_faces_dnn(frame)
        elif method == "mediapipe" and MEDIAPIPE_AVAILABLE:
            return self.detect_faces_mediapipe(frame)
        else:
            # Fallback para DNN
            return self.detect_faces_dnn(frame)
