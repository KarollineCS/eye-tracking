import os
import sys
import torch
import torchvision.transforms as transforms
import numpy as np
from typing import Dict, Optional, Tuple
import cv2
from PIL import Image

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from models.model import Model as GazeModel

class HybridGazeEstimator:
    """Estimador de gaze usando o modelo pré-treinado do Pascal Perle"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️ Usando dispositivo: {self.device}")
        
        self.face_size = 96
        self.eye_width = 96   
        self.eye_height = 64
        
        self.model = self._load_pretrained_model()
        self.model_loaded = self.model is not None
        
        if self.model_loaded:
            self.model.to(self.device)
            self.model.eval()
            print("✅ Modelo ML carregado e pronto!")
        else:
            print("⚠️ Modelo não carregado - usando método geométrico")
        
        # Transformações seguindo o padrão do Pascal
        self.transform_face = transforms.Compose([
            transforms.Resize((self.face_size, self.face_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        self.transform_eye = transforms.Compose([
            transforms.Resize((self.eye_height, self.eye_width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def _load_pretrained_model(self):
        """Carrega o modelo pré-treinado"""
        model_path = self.config.algorithm.gaze_model_path
        
        if not os.path.exists(model_path):
            print(f"❌ Arquivo não encontrado: {model_path}")
            return None
        
        try:
            print(f"📥 Carregando modelo de: {model_path}")
            
            model = GazeModel()
            checkpoint = torch.load(model_path, map_location=self.device)
            
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            model.load_state_dict(state_dict, strict=True)
            print(f"✅ Pesos do modelo carregados!")
            
            return model
            
        except Exception as e:
            print(f"❌ Erro ao carregar modelo: {e}")
            return None
    
    def prepare_inputs(self, frame, face_bbox, landmarks=None):
        """Prepara as entradas para o modelo"""
        try:
            x, y, w, h = face_bbox
            
            # Garantir coordenadas válidas
            x, y = max(0, int(x)), max(0, int(y))
            x_end = min(frame.shape[1], int(x + w))
            y_end = min(frame.shape[0], int(y + h))
            
            # Recortar face
            face_img = frame[y:y_end, x:x_end]
            
            if face_img.size == 0 or face_img.shape[0] < 20 or face_img.shape[1] < 20:
                return None
            
            # Converter BGR para RGB
            face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            
            # Extrair olhos
            if landmarks is not None and len(landmarks) >= 48:
                left_eye_img = self._crop_eye_from_landmarks(frame, landmarks, 'left')
                right_eye_img = self._crop_eye_from_landmarks(frame, landmarks, 'right')
            else:
                # Estimar posição dos olhos na face
                left_eye_img, right_eye_img = self._estimate_eye_regions(face_img_rgb)
            
            # Converter para PIL
            face_pil = Image.fromarray(face_img_rgb)
            left_eye_pil = Image.fromarray(left_eye_img)
            right_eye_pil = Image.fromarray(right_eye_img)
            
            # Aplicar transformações com tamanhos corretos
            face_tensor = self.transform_face(face_pil).unsqueeze(0).to(self.device)
            left_eye_tensor = self.transform_eye(left_eye_pil).unsqueeze(0).to(self.device)
            right_eye_tensor = self.transform_eye(right_eye_pil).unsqueeze(0).to(self.device)
            
            # Person index (0 para pessoa desconhecida)
            person_idx = torch.tensor([0], dtype=torch.long).to(self.device)
            
            return {
                'person_idx': person_idx,
                'face': face_tensor,
                'left_eye': left_eye_tensor,
                'right_eye': right_eye_tensor
            }
            
        except Exception as e:
            if not hasattr(self, '_prep_error_logged'):
                print(f"Erro ao preparar inputs: {e}")
                self._prep_error_logged = True
            return None
    
    def _estimate_eye_regions(self, face_img_rgb):
        """Estima regiões dos olhos quando não há landmarks"""
        h, w = face_img_rgb.shape[:2]
        
        # Região dos olhos (parte superior da face)
        eye_top = int(h * 0.2)
        eye_bottom = int(h * 0.5)
        eye_height = eye_bottom - eye_top
        
        # Largura de cada olho
        eye_width = int(w * 0.35)
        
        # Olho esquerdo (direita na imagem)
        left_x = int(w * 0.55)
        left_eye = face_img_rgb[eye_top:eye_bottom, left_x:min(left_x + eye_width, w)]
        
        # Olho direito (esquerda na imagem)
        right_x = max(0, int(w * 0.45) - eye_width)
        right_eye = face_img_rgb[eye_top:eye_bottom, right_x:int(w * 0.45)]
        
        # Garantir que não estão vazios
        if left_eye.size == 0:
            left_eye = face_img_rgb
        if right_eye.size == 0:
            right_eye = face_img_rgb
            
        return left_eye, right_eye
    
    def _crop_eye_from_landmarks(self, frame, landmarks, eye_side):
        """Extrai região do olho usando landmarks"""
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            if eye_side == 'left':
                indices = list(range(36, 42))
            else:
                indices = list(range(42, 48))
            
            eye_points = np.array([landmarks[i] for i in indices])
            
            # Bounding box com margem
            x_min, y_min = np.min(eye_points, axis=0).astype(int)
            x_max, y_max = np.max(eye_points, axis=0).astype(int)
            
            # Adicionar margem
            width = x_max - x_min
            height = y_max - y_min
            margin = int(max(width, height) * 0.5)
            
            x_min = max(0, x_min - margin)
            y_min = max(0, y_min - margin)
            x_max = min(frame.shape[1], x_max + margin)
            y_max = min(frame.shape[0], y_max + margin)
            
            eye_img = frame_rgb[y_min:y_max, x_min:x_max]
            
            return eye_img if eye_img.size > 0 else frame_rgb
            
        except Exception:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    def estimate_gaze(self, frame, face_bbox, landmarks=None):
        """Estima o gaze usando o modelo neural"""
        if not self.model_loaded:
            return None
        
        try:
            inputs = self.prepare_inputs(frame, face_bbox, landmarks)
            if inputs is None:
                return None
            
            with torch.no_grad():
                # Forward pass
                output = self.model(
                    inputs['person_idx'],
                    inputs['face'],
                    inputs['right_eye'],  
                    inputs['left_eye']
                )
                
                gaze = output[0].cpu().numpy()
                
                # O modelo retorna [pitch, yaw] em radianos
                return {
                    'yaw': float(gaze[1]),
                    'pitch': float(gaze[0]),
                    'confidence': 0.95
                }
                
        except Exception as e:
            if not hasattr(self, '_error_logged'):
                print(f"Erro na estimativa ML: {e}")
                self._error_logged = True
            return None


class GazeCalculator:
    """Calculador principal de gaze"""
    
    def __init__(self, config):
        self.config = config
        self.ml_estimator = HybridGazeEstimator(config)
        self.use_ml = self.ml_estimator.model_loaded
        
        # Histórico para suavização
        self.gaze_history = []
        self.max_history = 5
        
        print(f"🎯 GazeCalculator: ML {'ativado' if self.use_ml else 'desativado'}")
    
    def calculate_gaze(self, iris_data, landmarks, frame, face_bbox):
        """Calcula gaze usando ML ou método geométrico"""
        
        # Tentar ML primeiro
        if self.use_ml and frame is not None and face_bbox is not None:
            ml_result = self.ml_estimator.estimate_gaze(frame, face_bbox, landmarks)
            
            if ml_result is not None:
                # Converter para formato padrão
                gaze_vectors = self._ml_to_gaze_vectors(ml_result, iris_data)
                return self._apply_smoothing(gaze_vectors)
        
        # Fallback para método geométrico
        return self.calculate_gaze_geometric(iris_data, landmarks)
    
    def _ml_to_gaze_vectors(self, ml_result, iris_data):
        """Converte resultado ML para formato de gaze_vectors"""
        # Pegar centros dos olhos do iris_data se disponível
        left_center = iris_data.get('left', {}).get('center', [0, 0]) if iris_data else [0, 0]
        right_center = iris_data.get('right', {}).get('center', [0, 0]) if iris_data else [0, 0]
        
        # Criar vetor 3D baseado nos ângulos
        yaw = ml_result['yaw']
        pitch = ml_result['pitch']
        
        vector_3d = [
            np.sin(yaw) * np.cos(pitch),
            np.sin(pitch),
            np.cos(yaw) * np.cos(pitch)
        ]
        
        return {
            'left': {
                'yaw': yaw,
                'pitch': pitch,
                'eye_center': left_center,
                'iris_center': left_center,
                'vector_3d': vector_3d
            },
            'right': {
                'yaw': yaw,
                'pitch': pitch,
                'eye_center': right_center,
                'iris_center': right_center,
                'vector_3d': vector_3d
            }
        }
    
    def calculate_gaze_geometric(self, iris_data, landmarks):
        """Método geométrico como fallback"""
        if not iris_data:
            return {}
        
        gaze_vectors = {}
        
        for eye_side in ['left', 'right']:
            if eye_side not in iris_data:
                continue
            
            iris_info = iris_data[eye_side]
            iris_center = iris_info['center']
            
            # Estimar centro do olho
            if landmarks is not None and len(landmarks) >= 48:
                if eye_side == 'left':
                    eye_points = landmarks[36:42]
                else:
                    eye_points = landmarks[42:48]
                eye_center = np.mean(eye_points, axis=0)
            else:
                eye_center = iris_center
            
            # Calcular direção
            dx = iris_center[0] - eye_center[0]
            dy = iris_center[1] - eye_center[1]
            
            # Converter para ângulos
            focal_length = 600
            yaw = np.arctan(dx / focal_length)
            pitch = np.arctan(dy / focal_length)
            
            # Vetor 3D
            vector_3d = np.array([dx, dy, focal_length])
            vector_3d = vector_3d / np.linalg.norm(vector_3d)
            
            gaze_vectors[eye_side] = {
                'yaw': float(yaw),
                'pitch': float(pitch),
                'eye_center': eye_center.tolist() if isinstance(eye_center, np.ndarray) else list(eye_center),
                'iris_center': list(iris_center),
                'vector_3d': vector_3d.tolist()
            }
        
        return self._apply_smoothing(gaze_vectors)
    
    def _apply_smoothing(self, gaze_vectors):
        """Aplica suavização temporal"""
        if not gaze_vectors:
            return gaze_vectors
        
        self.gaze_history.append(gaze_vectors)
        if len(self.gaze_history) > self.max_history:
            self.gaze_history.pop(0)
        
        if len(self.gaze_history) < 2:
            return gaze_vectors
        
        # Suavização exponencial
        alpha = 0.7
        smoothed = {}
        
        for eye_side in ['left', 'right']:
            if eye_side not in gaze_vectors:
                continue
            
            current = gaze_vectors[eye_side]
            previous = None
            
            for hist in reversed(self.gaze_history[:-1]):
                if eye_side in hist:
                    previous = hist[eye_side]
                    break
            
            if previous:
                smoothed[eye_side] = {
                    'yaw': alpha * current['yaw'] + (1 - alpha) * previous['yaw'],
                    'pitch': alpha * current['pitch'] + (1 - alpha) * previous['pitch'],
                    'eye_center': current['eye_center'],
                    'iris_center': current['iris_center'],
                    'vector_3d': current['vector_3d']
                }
            else:
                smoothed[eye_side] = current
        
        return smoothed
    
    def calculate_average_gaze(self, gaze_vectors):
        """Calcula média entre os dois olhos"""
        if not gaze_vectors:
            return None
        
        if 'left' in gaze_vectors and 'right' in gaze_vectors:
            return {
                'yaw': (gaze_vectors['left']['yaw'] + gaze_vectors['right']['yaw']) / 2,
                'pitch': (gaze_vectors['left']['pitch'] + gaze_vectors['right']['pitch']) / 2
            }
        elif 'left' in gaze_vectors:
            return {'yaw': gaze_vectors['left']['yaw'], 'pitch': gaze_vectors['left']['pitch']}
        elif 'right' in gaze_vectors:
            return {'yaw': gaze_vectors['right']['yaw'], 'pitch': gaze_vectors['right']['pitch']}
        
        return None
