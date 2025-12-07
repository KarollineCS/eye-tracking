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
        
        self.face_size = 96
        self.eye_width = 96  
        self.eye_height = 64
        
        self.model = self._load_pretrained_model()
        self.model_loaded = self.model is not None
        
        if self.model_loaded:
            self.model.to(self.device)
            self.model.eval()
        else:
            print("⚠️ Modelo não carregado - usando método geométrico")
        
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

        # Inicializar flag de erro para evitar spam
        self._prep_error_logged = False
    
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

        # Sistema de esferas oculares 3D
        self.eye_sphere_system = EyeSphereSystem()
    
    def calculate_gaze(self, iris_data, landmarks, frame, face_bbox):
        """Calcula gaze usando método geométrico"""
        
        # Fallback para método geométrico
        frame_width = frame.shape[1] if frame is not None else 600
        return self.calculate_gaze_geometric(iris_data, landmarks, frame_width)
    
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
    
    def calculate_gaze_geometric(self, iris_data, landmarks, frame_width=600):
        """Método geométrico como fallback"""
        if not iris_data:
            return {}
        
        gaze_vectors = {}
        
        for eye_side in ['left', 'right']:
            # Verifica se temos dados mínimos necessários
            if eye_side not in iris_data or 'center' not in iris_data[eye_side]:
                continue

            # Pega os dados da íris
            iris_center = iris_data[eye_side]['center']

            # Pegar o centro do olho (com fallback se não existir)
            if 'eye_center_2d' in iris_data[eye_side]:
                eye_center = iris_data[eye_side]['eye_center_2d']
            else:
                eye_center = iris_center
            
            # Calcular direção
            dx = iris_center[0] - eye_center[0]
            dy = iris_center[1] - eye_center[1]

            if 'eye_width' in iris_data[eye_side] and iris_data[eye_side]['eye_width'] > 0:
                eye_width = iris_data[eye_side]['eye_width']
                # Normalizar: dividir pelo raio (metade da largura)
                dx_normalized = dx / (eye_width / 2.0)
                dy_normalized = dy / (eye_width / 2.0)

                # Usar valores normalizados
                dx = dx_normalized
                dy = dy_normalized
            else:
                # Fallback: normalizar por raio se configurado
                if self.config.algorithm.normalize_by_radius:
                    if 'radius' in iris_data[eye_side] and iris_data[eye_side]['radius'] > 0:
                        dx /= (iris_data[eye_side]['radius'] * 2)
                        dy /= (iris_data[eye_side]['radius'] * 2)

            focal_length = 1.0  # Focal length para valores normalizados
            yaw = np.arctan(dx / focal_length)
            pitch = np.arctan(dy / focal_length) * -1.0  # Inverte o eixo Y


            # Aplicar ganho de amplificação
            gain = self.config.algorithm.gaze_gain
            yaw *= gain
            pitch *= gain
            
            # Vetor 3D (para compatibilidade)
            vector_3d = np.array([dx, dy, focal_length])
            vector_3d = vector_3d / np.linalg.norm(vector_3d)
            
            gaze_vectors[eye_side] = {
                'yaw': float(yaw),
                'pitch': float(pitch),
                'eye_center': list(eye_center),
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
    
    def calculate_gaze_3d(self, iris_data, head_center, R_head, current_scale):
        if not self.eye_sphere_system.left_calibrated:
            return {}

        # Preparar dados
        iris_positions_3d = {}
        if 'left' in iris_data and 'center_3d' in iris_data['left']:
            iris_positions_3d['left'] = iris_data['left']['center_3d']
        if 'right' in iris_data and 'center_3d' in iris_data['right']:
            iris_positions_3d['right'] = iris_data['right']['center_3d']

        # Obter centros das esferas
        eye_centers = self.eye_sphere_system.get_eye_centers(
            head_center, R_head, current_scale
        )

        # Calcular gaze 3D
        return self.eye_sphere_system.compute_gaze_3d(
            eye_centers, iris_positions_3d, iris_data
        )

    def calibrate_eye_spheres(self, iris_data, head_center, R_head, current_scale):
        if 'left' not in iris_data or 'right' not in iris_data:
            return False
        
        iris_left_3d = iris_data['left']['center_3d']
        iris_right_3d = iris_data['right']['center_3d']
        
        self.eye_sphere_system.calibrate_spheres(
            iris_left_3d, iris_right_3d,
            head_center, R_head, current_scale
        )
        
        return True
    
class EyeSphereSystem:
    """
    Sistema de esferas oculares 3D
    """
    
    def __init__(self):
        self.base_radius = 12.0  # mm
        self.left_calibrated = False
        self.right_calibrated = False
        self.left_offset = None
        self.right_offset = None
        self.left_scale = None
        self.right_scale = None
        
        #Parâmetros de compensação de assimetria
        self.center_offset = None
        self.asymmetry_factor = 0.0
        self.calibration_samples = []
        self.offset_compensation = {'x': 0, 'y': 0, 'z': 0}
    
    def calibrate_spheres(self, iris_left_3d, iris_right_3d,
                         head_center, R_head, current_scale):
        """Calibra esferas oculares com compensação de assimetria"""
        camera_dir = np.array([0, 0, -1], dtype=float)

        # Transformar direção da câmera para o sistema da cabeça
        camera_dir_local = R_head.T @ camera_dir
        camera_dir_local = camera_dir_local / np.linalg.norm(camera_dir_local)

        # Calibração individual do olho esquerdo
        iris_left_local = R_head.T @ (iris_left_3d - head_center)

        # Calibração individual do olho direito
        iris_right_local = R_head.T @ (iris_right_3d - head_center)

        avg_z_local = (iris_left_local[2] + iris_right_local[2]) / 2.0

        iris_left_local_corrigido = iris_left_local.copy()
        iris_left_local_corrigido[2] = avg_z_local

        iris_right_local_corrigido = iris_right_local.copy()
        iris_right_local_corrigido[2] = avg_z_local

        self.left_offset = iris_left_local_corrigido - (self.base_radius * camera_dir_local)
        self.left_scale = current_scale
        self.left_calibrated = True

        self.right_offset = iris_right_local_corrigido - (self.base_radius * camera_dir_local)
        self.right_scale = current_scale
        self.right_calibrated = True
        
        # Calcular assimetria e offset médio
        self.center_offset = (self.left_offset + self.right_offset) / 2.0
        self.asymmetry_factor = np.linalg.norm(self.left_offset - self.right_offset)

        # Detectar e compensar deslocamento sistemático
        offset_diff = self.left_offset - self.right_offset
        if abs(offset_diff[0]) > 2.0:  # Se diferença X > 2mm
            self.offset_compensation['x'] = offset_diff[0] * 0.5
            print(f"⚠️ Assimetria horizontal detectada: {offset_diff[0]:.1f}mm")
    
    def get_eye_centers(self, head_center, R_head, current_scale):
        """Retorna centros das esferas com compensação completa"""
        if not (self.left_calibrated and self.right_calibrated):
            return {}

        centers = {}

        # Esquerda com compensação individual
        scale_ratio_left = current_scale / self.left_scale
        left_offset_scaled = self.left_offset * scale_ratio_left
        # compensação de assimetria
        left_offset_scaled[0] -= self.offset_compensation['x'] * 0.5
        centers['left'] = head_center + R_head @ left_offset_scaled

        # Direita com compensação individual
        scale_ratio_right = current_scale / self.right_scale
        right_offset_scaled = self.right_offset * scale_ratio_right
        # compensação de assimetria
        right_offset_scaled[0] += self.offset_compensation['x'] * 0.5
        centers['right'] = head_center + R_head @ right_offset_scaled

        return centers
    
    def compute_gaze_3d(self, eye_centers, iris_positions_3d, iris_data=None):
        """Calcula gaze vectors 3D com compensação"""
        gaze_vectors = {}

        # Garantir mesma profundidade para ambas as íris
        if 'left' in iris_positions_3d and 'right' in iris_positions_3d:
            iris_left = iris_positions_3d['left']
            iris_right = iris_positions_3d['right']

            # Usar profundidade média em Z
            avg_z = (iris_left[2] + iris_right[2]) / 2.0

            iris_positions_3d['left'] = np.array([iris_left[0], iris_left[1], avg_z])
            iris_positions_3d['right'] = np.array([iris_right[0], iris_right[1], avg_z])

        for eye in ['left', 'right']:
            if eye in eye_centers and eye in iris_positions_3d:
                eye_center = eye_centers[eye]
                iris_pos = iris_positions_3d[eye]

                # Este vetor aponta na direção que o olho está olhando
                gaze_vec = iris_pos - eye_center
                norm = np.linalg.norm(gaze_vec)

                if norm > 1e-9:
                    direction = gaze_vec / norm

                    # Guardar vetor original para mundo virtual 3D
                    direction_3d = direction.copy()

                    # Inverter X para cálculos de cursor/yaw
                    direction[0] = -direction[0]
                else:
                    # Se não há movimento, olhar para frente
                    direction = np.array([0, 0, 1])
                    direction_3d = direction.copy()

                # Manter compatibilidade com visualização 2D
                if iris_data and eye in iris_data:
                    iris_center_2d = iris_data[eye].get('center', iris_pos[:2])
                    eye_center_2d = iris_data[eye].get('eye_center_2d', eye_center[:2])
                else:
                    iris_center_2d = iris_pos[:2]
                    eye_center_2d = eye_center[:2]

                gaze_vectors[eye] = {
                    'origin': eye_center,
                    'direction': direction_3d,  # Vetor original para mundo virtual 3D
                    'iris_center_3d': iris_pos,
                    # Yaw: rotação horizontal (esquerda/direita)
                    # Negação em X converte coordenada da imagem para direção do olhar da pessoa
                    'yaw': float(np.arctan2(-direction[0], direction[2])),
                    # Pitch: rotação vertical (cima/baixo)
                    # Negação em Y porque MediaPipe Y+ = baixo, mas queremos Y+ = cima
                    'pitch': float(np.arcsin(-direction[1])),
                    'vector_3d': direction_3d,  # Vetor original para mundo virtual 3D
                    'eye_center': eye_center_2d,
                    'iris_center': iris_center_2d
                }

        if 'left' in gaze_vectors and 'right' in gaze_vectors:
            left_origin = gaze_vectors['left']['origin']
            right_origin = gaze_vectors['right']['origin']
            left_dir = gaze_vectors['left']['direction']
            right_dir = gaze_vectors['right']['direction']
            
            # Média das origens
            origin_mid = (left_origin + right_origin) / 2.0
            
            # Média das direções (normalizada)
            combined_dir = (left_dir + right_dir) / 2.0
            combined_dir = combined_dir / np.linalg.norm(combined_dir)
            
            # Adicionar ao resultado
            gaze_vectors['combined'] = {
                'origin': origin_mid,
                'direction': combined_dir,
            }
            
            # Métricas (opcional, para debug)
            dot_product = np.clip(np.dot(left_dir, right_dir), -1.0, 1.0)
            angle_deg = np.degrees(np.arccos(dot_product))
            gaze_vectors['discrepancy_angle'] = float(angle_deg)

        return gaze_vectors
