import cv2
import numpy as np
from collections import deque
from typing import Dict, List, Tuple, Optional
import time

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

class IrisTracker:
    """Rastreador de íris com múltiplas estratégias"""
    
    def __init__(self, config):
        self.config = config
        self.iris_history = {'left': deque(maxlen=5), 'right': deque(maxlen=5)}
        self.max_iris_movement = config.algorithm.max_iris_movement
        self.last_iris_positions = {'left': None, 'right': None}
        
        # Índices dos landmarks da íris no MediaPipe
        self.LEFT_IRIS_LANDMARKS = [474, 475, 476, 477]
        self.RIGHT_IRIS_LANDMARKS = [469, 470, 471, 472]
        
        # Índices dos landmarks dos olhos (dlib 68 pontos)
        self.LEFT_EYE_POINTS = list(range(36, 42))
        self.RIGHT_EYE_POINTS = list(range(42, 48))
        
        # Inicializar MediaPipe se disponível
        if MEDIAPIPE_AVAILABLE:
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.5
            )
        else:
            self.face_mesh = None
            print("⚠️ MediaPipe não disponível. Funcionalidade limitada.")
    
    def detect_iris_mediapipe(self, frame) -> Dict:
        """Detecta íris usando MediaPipe"""
        if not MEDIAPIPE_AVAILABLE or self.face_mesh is None:
            return {}
        
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)
            
            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                h, w = frame.shape[:2]
                
                iris_data = {}
                
                # Detectar íris esquerda
                left_iris_points = []
                for idx in self.LEFT_IRIS_LANDMARKS:
                    if idx < len(face_landmarks.landmark):
                        landmark = face_landmarks.landmark[idx]
                        x = int(landmark.x * w)
                        y = int(landmark.y * h)
                        left_iris_points.append((x, y))
                
                if len(left_iris_points) >= 3:
                    left_points = np.array(left_iris_points)
                    center_x = int(np.mean(left_points[:, 0]))
                    center_y = int(np.mean(left_points[:, 1]))
                    distances = np.sqrt(np.sum((left_points - [center_x, center_y])**2, axis=1))
                    radius = max(8, int(np.mean(distances) * 1.8))
                    
                    iris_data['left'] = {
                        'center': (center_x, center_y),
                        'radius': radius,
                        'points': left_iris_points
                    }
                
                # Detectar íris direita
                right_iris_points = []
                for idx in self.RIGHT_IRIS_LANDMARKS:
                    if idx < len(face_landmarks.landmark):
                        landmark = face_landmarks.landmark[idx]
                        x = int(landmark.x * w)
                        y = int(landmark.y * h)
                        right_iris_points.append((x, y))
                
                if len(right_iris_points) >= 3:
                    right_points = np.array(right_iris_points)
                    center_x = int(np.mean(right_points[:, 0]))
                    center_y = int(np.mean(right_points[:, 1]))
                    distances = np.sqrt(np.sum((right_points - [center_x, center_y])**2, axis=1))
                    radius = max(8, int(np.mean(distances) * 1.8))
                    
                    iris_data['right'] = {
                        'center': (center_x, center_y),
                        'radius': radius,
                        'points': right_iris_points
                    }
                
                return iris_data
                
        except Exception as e:
            if self.config.debug_enabled:
                print(f"Erro no MediaPipe: {e}")
        
        return {}
    
    def detect_iris_direct(self, frame, face_landmarks) -> Dict:
        """Detecção direta sem validação complexa"""
        # Usar MediaPipe diretamente
        raw_iris_data = self.detect_iris_mediapipe(frame)
        
        if not raw_iris_data:
            return self._get_fallback_iris()
        
        # Aplicar apenas filtros básicos
        filtered_iris = self._apply_basic_filters(raw_iris_data)
        
        # Verificar mapeamento por posição relativa
        corrected_iris = filtered_iris
        
        # Atualizar histórico
        self._update_history(corrected_iris)
        
        return self._get_smoothed_iris()
    
    def _is_movement_reasonable(self, eye_side, new_center) -> bool:
        """Verifica se movimento é razoável comparado à posição anterior"""
        if self.last_iris_positions[eye_side] is None:
            return True
        
        last_x, last_y = self.last_iris_positions[eye_side]
        new_x, new_y = new_center
        
        # Calcular distância simples
        dx = new_x - last_x
        dy = new_y - last_y
        distance = (dx * dx + dy * dy) ** 0.5
        
        return distance <= self.max_iris_movement
    
    def _simple_mapping_check(self, iris_data, face_landmarks) -> Dict:
        """Verificação de mapeamento SEM arrays NumPy"""
        if not iris_data or face_landmarks is None:
            return iris_data
        
        if 'left' not in iris_data or 'right' not in iris_data:
            return iris_data
        
        try:
            # Calcular centros dos olhos usando listas Python simples
            left_eye_points = [face_landmarks[i] for i in self.LEFT_EYE_POINTS]
            right_eye_points = [face_landmarks[i] for i in self.RIGHT_EYE_POINTS]
            
            # Média simples sem NumPy
            left_eye_center_x = sum(p[0] for p in left_eye_points) / len(left_eye_points)
            left_eye_center_y = sum(p[1] for p in left_eye_points) / len(left_eye_points)
            
            right_eye_center_x = sum(p[0] for p in right_eye_points) / len(right_eye_points)
            right_eye_center_y = sum(p[1] for p in right_eye_points) / len(right_eye_points)
            
            # Posições das íris
            left_iris_x, left_iris_y = iris_data['left']['center']
            right_iris_x, right_iris_y = iris_data['right']['center']
            
            # Calcular distâncias usando matemática simples
            dist_left_to_left_eye = ((left_iris_x - left_eye_center_x) ** 2 + 
                                   (left_iris_y - left_eye_center_y) ** 2) ** 0.5
            
            dist_left_to_right_eye = ((left_iris_x - right_eye_center_x) ** 2 + 
                                    (left_iris_y - right_eye_center_y) ** 2) ** 0.5
            
            dist_right_to_left_eye = ((right_iris_x - left_eye_center_x) ** 2 + 
                                    (right_iris_y - left_eye_center_y) ** 2) ** 0.5
            
            dist_right_to_right_eye = ((right_iris_x - right_eye_center_x) ** 2 + 
                                     (right_iris_y - right_eye_center_y) ** 2) ** 0.5
            
            # Verificar se mapeamento está trocado
            if (dist_left_to_right_eye < dist_left_to_left_eye and 
                dist_right_to_left_eye < dist_right_to_right_eye):
                
                if self.config.debug_enabled:
                    print("🔄 Mapeamento trocado detectado - corrigindo")
                return {
                    'left': iris_data['right'].copy(),
                    'right': iris_data['left'].copy()
                }
            
            return iris_data
            
        except Exception as e:
            if self.config.debug_enabled:
                print(f"Erro na verificação de mapeamento: {e}")
            return iris_data
    
    def _simple_smoothing(self, iris_data) -> Dict:
        """Suavização simples sem arrays complexos"""
        smoothed = {}
        
        for eye_side in ['left', 'right']:
            if eye_side in iris_data:
                current_center = iris_data[eye_side]['center']
                
                # Se tem posição anterior, aplicar suavização simples
                if self.last_iris_positions[eye_side] is not None:
                    last_x, last_y = self.last_iris_positions[eye_side]
                    curr_x, curr_y = current_center
                    
                    # Média ponderada simples (80% atual, 20% anterior)
                    smooth_x = int(curr_x * 0.8 + last_x * 0.2)
                    smooth_y = int(curr_y * 0.8 + last_y * 0.2)
                    
                    smoothed_center = (smooth_x, smooth_y)
                else:
                    smoothed_center = current_center
                
                smoothed[eye_side] = iris_data[eye_side].copy()
                smoothed[eye_side]['center'] = smoothed_center
                
                # Atualizar posição anterior
                self.last_iris_positions[eye_side] = smoothed_center
        
        return smoothed
    
    def _update_history(self, iris_data):
        """Atualiza histórico simples"""
        for eye_side in ['left', 'right']:
            if eye_side in iris_data:
                self.iris_history[eye_side].append(iris_data[eye_side])
    
    def _get_fallback_iris(self) -> Dict:
        """Fallback usando histórico"""
        fallback = {}
        for eye_side in ['left', 'right']:
            if len(self.iris_history[eye_side]) > 0:
                fallback[eye_side] = self.iris_history[eye_side][-1]
        return fallback
    
    def detect_iris_with_validation(self, frame, face_landmarks) -> Dict:
        """Detecção de íris com validação e debug"""
        try:
            if self.config.debug_enabled:
                print("DEBUG IrisTracker: Iniciando detecção...")
            
            # Usar método original
            iris_data = self.detect_iris_mediapipe(frame)
            
            if self.config.debug_enabled:
                print(f"DEBUG IrisTracker: Dados recebidos: {list(iris_data.keys())}")
            
            # Se não tem dados, retorna histórico
            if not iris_data:
                return self._get_fallback_iris()
            
            # Validação baseada nos landmarks faciais
            validated_iris = self._validate_iris_positions(iris_data, face_landmarks)
            
            if self.config.debug_enabled:
                print(f"DEBUG IrisTracker: Após validação: {list(validated_iris.keys())}")
            
            # Se validação rejeitou tudo, usa dados originais
            if not validated_iris and iris_data:
                validated_iris = iris_data
            
            # Atualiza histórico
            self._update_history(validated_iris)
            
            # Retorna resultado suavizado
            smoothed = self._get_smoothed_iris()
            return smoothed
            
        except Exception as e:
            if self.config.debug_enabled:
                print(f"DEBUG IrisTracker: Erro na detecção: {e}")
            return self._get_fallback_iris()
    
    def _validate_iris_positions(self, iris_data, face_landmarks) -> Dict:
        """Valida posições de íris baseada nos landmarks dos olhos"""
        if not iris_data or face_landmarks is None:
            return {}
        
        validated = {}
        
        for eye_side in ['left', 'right']:
            if eye_side in iris_data:
                iris_center = iris_data[eye_side]['center']
                
                # Define pontos do olho
                if eye_side == 'left':
                    eye_points = [face_landmarks[i] for i in self.LEFT_EYE_POINTS]
                else:
                    eye_points = [face_landmarks[i] for i in self.RIGHT_EYE_POINTS]
                
                # Calcula região válida do olho
                eye_bbox = self._calculate_eye_bbox(eye_points)
                
                # Verifica se íris está dentro da região válida
                if self._is_iris_position_valid(iris_center, eye_bbox):
                    validated[eye_side] = iris_data[eye_side]
        
        return validated
    
    def _calculate_eye_bbox(self, eye_points) -> Tuple[int, int, int, int]:
        """Calcula bounding box do olho com margem generosa"""
        min_x = min(point[0] for point in eye_points)
        max_x = max(point[0] for point in eye_points)
        min_y = min(point[1] for point in eye_points)
        max_y = max(point[1] for point in eye_points)
        
        # Margem generosa
        margin_x = (max_x - min_x) * 0.3  # 30% da largura do olho
        margin_y = (max_y - min_y) * 0.5  # 50% da altura do olho
        
        return (
            int(min_x - margin_x), 
            int(min_y - margin_y), 
            int(max_x + margin_x), 
            int(max_y + margin_y)
        )
    
    def _is_iris_position_valid(self, iris_center, eye_bbox) -> bool:
        """Verifica se posição da íris é válida"""
        x, y = iris_center
        min_x, min_y, max_x, max_y = eye_bbox
        
        x_valid = min_x <= x <= max_x
        y_valid = min_y <= y <= max_y
        
        return x_valid and y_valid
    
    def _get_smoothed_iris(self) -> Dict:
        """Retorna posição de íris suavizada"""
        smoothed = {}
        
        for eye_side in ['left', 'right']:
            if len(self.iris_history[eye_side]) > 0:
                recent_iris = list(self.iris_history[eye_side])
                
                # Média ponderada (mais peso para detecções recentes)
                weights = [0.5, 0.3, 0.2][:len(recent_iris)]
                weights = weights[::-1]  # Inverte para dar mais peso ao recente
                
                center_x = sum(iris['center'][0] * w for iris, w in zip(recent_iris, weights)) / sum(weights)
                center_y = sum(iris['center'][1] * w for iris, w in zip(recent_iris, weights)) / sum(weights)
                avg_radius = sum(iris['radius'] for iris in recent_iris) / len(recent_iris)
                
                smoothed[eye_side] = {
                    'center': (int(center_x), int(center_y)),
                    'radius': int(avg_radius),
                    'points': recent_iris[-1]['points']  # Usa pontos mais recentes
                }
        
        return smoothed
    
    def track_iris(self, frame, face_landmarks=None) -> Dict:
        #Método principal para rastreamento de íris
        method = self.config.algorithm.iris_detection_method
        
        if method == "direct":
            return self.detect_iris_direct(frame, face_landmarks)
        elif method == "improved":
            return self.detect_iris_with_validation(frame, face_landmarks)
        elif method == "hybrid":
            # Tenta detecção direta primeiro, fallback para validada
            direct_result = self.detect_iris_direct(frame, face_landmarks)
            if len(direct_result) >= 2:  # Ambos os olhos detectados
                return direct_result
            else:
                return self.detect_iris_with_validation(frame, face_landmarks)
        else:
            # Default para detecção direta
            return self.detect_iris_direct(frame, face_landmarks)
