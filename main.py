import cv2
import time
import numpy as np
import tkinter as tk
import math
from typing import Tuple, Optional
from collections import deque

# Imports dos módulos locais
from src.config.settings import GazeTrackingConfig, DEFAULT_CONFIG
from src.core.face_detector import RobustFaceDetector
from src.core.iris_tracker import IrisTracker
from src.core.gaze_calculator import GazeCalculator
#from src.core.calibration import AdaptiveCalibrationSystem, ScreenCalibrationSystem
#from src.analysis.fatigue_detection import AdvancedFatigueDetector
from src.utils.performance import PerformanceOptimizer
from src.utils.visualization import VisualizationManager
from src.calibration.screen_calibration import AdaptiveCalibration, ManualScreenCalibration
from src.calibration.gaze_to_screen_3d import GazeToScreen3D  # Mapeamento angular
from src.filters.kalman_filter import GazeKalmanFilter, AdaptiveKalmanFilter
from src.calibration.advanced_calibration import AdvancedScreenCalibration
from src.filters.binocular_fusion import (
    BinocularFusion, 
    create_binocular_fusion_system,
    integrate_with_gaze_vectors
)
from src.calibration.chessboard_validator import ChessboardValidator
from src.calibration.refinement_calibration import RefinementCalibration

def _rot_x(a):
    ca, sa = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0],
                     [0, ca, -sa],
                     [0, sa,  ca]], dtype=float)

def _rot_y(a):
    ca, sa = math.cos(a), math.sin(a)
    return np.array([[ ca, 0, sa],
                     [  0, 1,  0],
                     [-sa, 0, ca]], dtype=float)

def _normalize(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v

def _focal_px(width, fov_deg):
    # Foco pinhole horizontal
    return 0.5 * width / math.tan(math.radians(fov_deg) * 0.5)

def clamp_to_monitor_bounds(
    intersection_point: np.ndarray,
    monitor_corners: list,
    return_normalized: bool = False
) -> Tuple[np.ndarray, float, float, bool]:
    """
    Limita um ponto de interseção às bordas do monitor.
    
    Args:
        intersection_point: Ponto 3D de interseção do raio com o plano do monitor
        monitor_corners: Lista de 4 pontos [p0, p1, p2, p3] definindo o monitor
                        p0 = canto superior esquerdo
                        p1 = canto superior direito
                        p2 = canto inferior direito
                        p3 = canto inferior esquerdo
        return_normalized: Se True, retorna também as coordenadas normalizadas
        
    Returns:
        Tupla (ponto_clamped, a, b, was_clamped)
        - ponto_clamped: Ponto 3D dentro dos limites do monitor
        - a: Coordenada horizontal normalizada [0, 1]
        - b: Coordenada vertical normalizada [0, 1]
        - was_clamped: True se o ponto original estava fora
    """
    P = np.asarray(intersection_point, dtype=float)
    p0, p1, p2, p3 = [np.asarray(p, dtype=float) for p in monitor_corners]
    
    # Vetores das bordas do monitor
    u = p1 - p0  # Horizontal (esquerda -> direita)
    v = p3 - p0  # Vertical (cima -> baixo)
    
    # Comprimentos ao quadrado
    u_len2 = np.dot(u, u)
    v_len2 = np.dot(v, v)
    
    if u_len2 < 1e-9 or v_len2 < 1e-9:
        # Monitor degenerado
        return P, 0.5, 0.5, False
    
    # Vetor do canto para o ponto de interseção
    w = P - p0
    
    # Coordenadas normalizadas (projeção no sistema local do monitor)
    a = np.dot(w, u) / u_len2  # 0 = esquerda, 1 = direita
    b = np.dot(w, v) / v_len2  # 0 = cima, 1 = baixo
    
    # Verificar se precisa de clamp
    a_original, b_original = a, b
    was_clamped = False
    
    # Clamp horizontal
    if a < 0.0:
        a = 0.0
        was_clamped = True
    elif a > 1.0:
        a = 1.0
        was_clamped = True
    
    # Clamp vertical
    if b < 0.0:
        b = 0.0
        was_clamped = True
    elif b > 1.0:
        b = 1.0
        was_clamped = True
    
    # Reconstruir ponto 3D dentro dos limites
    clamped_point = p0 + a * u + b * v
    
    return clamped_point, a, b, was_clamped


def compute_gaze_on_monitor(
    gaze_origin: np.ndarray,
    gaze_direction: np.ndarray,
    monitor_corners: list,
    monitor_center: np.ndarray,
    monitor_normal: np.ndarray,
    clamp_to_bounds: bool = True
) -> Optional[dict]:
    """
    Calcula onde o gaze intersecta o monitor, com opção de clamping.
    
    Args:
        gaze_origin: Origem do raio de gaze (ponto médio entre os olhos)
        gaze_direction: Direção normalizada do gaze
        monitor_corners: Lista [p0, p1, p2, p3] dos cantos do monitor
        monitor_center: Centro do monitor
        monitor_normal: Normal do plano do monitor
        clamp_to_bounds: Se True, limita o ponto às bordas
        
    Returns:
        Dicionário com:
        - 'point_3d': Ponto 3D de interseção (clamped se aplicável)
        - 'a': Coordenada horizontal normalizada [0, 1]
        - 'b': Coordenada vertical normalizada [0, 1]
        - 'is_inside': True se estava dentro do monitor
        - 'was_clamped': True se foi aplicado clamp
        - 'distance': Distância do olho até o ponto
        - 'raw_a': Coordenada 'a' antes do clamp
        - 'raw_b': Coordenada 'b' antes do clamp
    """
    O = np.asarray(gaze_origin, dtype=float)
    D = np.asarray(gaze_direction, dtype=float)
    D = D / np.linalg.norm(D)  # Garantir normalização
    
    C = np.asarray(monitor_center, dtype=float)
    N = np.asarray(monitor_normal, dtype=float)
    N = N / np.linalg.norm(N)
    
    # Interseção raio-plano
    denom = np.dot(N, D)
    
    if abs(denom) < 1e-6:
        # Raio paralelo ao monitor
        return None
    
    t = np.dot(N, C - O) / denom
    
    if t <= 0:
        # Interseção atrás do olho
        return None
    
    # Ponto de interseção
    P = O + t * D
    
    # Calcular coordenadas normalizadas
    p0, p1, p2, p3 = [np.asarray(p, dtype=float) for p in monitor_corners]
    u = p1 - p0
    v = p3 - p0
    
    u_len2 = np.dot(u, u)
    v_len2 = np.dot(v, v)
    
    if u_len2 < 1e-9 or v_len2 < 1e-9:
        return None
    
    w = P - p0
    raw_a = np.dot(w, u) / u_len2
    raw_b = np.dot(w, v) / v_len2
    
    # Verificar se está dentro
    is_inside = (0.0 <= raw_a <= 1.0) and (0.0 <= raw_b <= 1.0)
    
    # Aplicar clamp se necessário
    if clamp_to_bounds and not is_inside:
        clamped_point, a, b, was_clamped = clamp_to_monitor_bounds(P, monitor_corners)
        point_3d = clamped_point
    else:
        a, b = raw_a, raw_b
        point_3d = P
        was_clamped = False
    
    return {
        'point_3d': point_3d,
        'a': a,
        'b': b,
        'is_inside': is_inside,
        'was_clamped': was_clamped,
        'distance': t,
        'raw_a': raw_a,
        'raw_b': raw_b
    }

class GazeTrackingSystem:
    """Sistema completo: 3 algoritmos + melhorias MonitorTracking"""
    
    def __init__(self, config=None):
        self.config = config or GazeTrackingConfig()
        
        # Componentes principais
        self.face_detector = RobustFaceDetector(self.config)
        self.iris_tracker = IrisTracker(self.config)
        self.gaze_calculator = GazeCalculator(self.config)
        
        # === ALGORITMO 1: SCREEN MAPPING 3D ===
        root = tk.Tk()
        screen_w = root.winfo_screenwidth()
        screen_h = root.winfo_screenheight()
        root.destroy()

        # Validador de tabuleiro de xadrez para calibração
        self.validator = ChessboardValidator(
            screen_width=screen_w,
            screen_height=screen_h,
            grid_rows=3,
            grid_cols=4,
            capture_duration=5.0,  # 10 segundos por célula (ajuste conforme necessário)
            countdown_seconds=3
        )
        self.validation_window_name = "Validacao - Eye Tracking"

        # Sistema 3D com mapeamento angular
        # Calibração rápida de centro + offset
        self.flip_x_axis = False  # Toggle para inverter eixo X
        self.gaze_to_screen_3d = GazeToScreen3D(screen_w, screen_h)

        # Sistema antigo ainda disponível como fallback
        self.screen_calibrator = AdvancedScreenCalibration(screen_w, screen_h)
        self.adaptive_screen_cal = AdaptiveCalibration(self.screen_calibrator)
        self.use_3d_mapping = True  # Por padrão, usar sistema 3D
        
        # Calibração de refinamento (9 pontos)
        self.refinement_calibrator = RefinementCalibration(screen_w, screen_h)
        self.use_refinement = False

        # === ALGORITMO 2: KALMAN FILTER ===
        self.kalman_filter = AdaptiveKalmanFilter()
        self.use_kalman = True

        # === FUSÃO BINOCULAR ===
        self.binocular_fusion = create_binocular_fusion_system(
            use_temporal_filter=True,
            vergence_threshold=30.0,      # mm - tolerância para vergência
            discrepancy_threshold=20.0,   # graus - limite para usar um só olho
            filter_window=5               # frames para média móvel
        )
        
        # === ALGORITMO 3: FATIGUE DETECTOR ===
        self.enable_fatigue_detection = False
        
        # Utilitários
        self.performance_optimizer = PerformanceOptimizer(self.config)
        self.visualization_manager = VisualizationManager(self.config)
        
        # === MELHORIAS 3D (MonitorTracking) ===
        self.spheres_calibrated = False
        self.current_head_center = None
        self.current_R_head = None
        self.current_scale = None
        self.mediapipe_results = None
        
        # Estados
        self.is_calibrating = False
        self.use_adaptive_calibration = True
        self.is_running = False
        self.show_3d_gaze_target = True  # Mostrar visualização 3D por padrão
        
        # Métricas
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        self.gaze_history = deque(maxlen=100)
        self._last_gaze = None

        self.cap = None

        # Loggers (serão definidos externamente)
        self.data_logger = None
        self.session_logger = None

        # === ESTABILIZAÇÃO DE GAZE (baseado em MonitorTracking.py) ===
        self.gaze_smoothing_enabled = True
        self.gaze_smoothing_buffer_size = 10  # Tamanho do buffer de suavização
        self.gaze_direction_buffer = deque(maxlen=self.gaze_smoothing_buffer_size)  # Buffer de direções 3D
        self.screen_position_buffer = deque(maxlen=self.gaze_smoothing_buffer_size)  # Buffer de posições 2D

        # === ESTADO DO MUNDO VIRTUAL 3D (Debug) ===
        self.orbit_yaw   = math.radians(-15.0) # Gira esquerda/direita
        self.orbit_pitch = math.radians(-15.0) # Gira cima/baixo
        self.orbit_radius = 1200.0         # Zoom (distância)
        self.orbit_fov_deg = 50.0          # Campo de visão

        # Pivô da câmera
        self.debug_world_frozen = False
        self.orbit_pivot_frozen = None

        # === CONTROLE DE MOUSE PARA CÂMERA 3D ===
        self.mouse_dragging = False
        self.mouse_last_x = 0
        self.mouse_last_y = 0
        self.mouse_sensitivity = 0.005  # Sensibilidade do drag
        self.zoom_sensitivity = 50.0    # Sensibilidade do scroll

        # Geometria do monitor 3D
        self.monitor_corners = None
        self.monitor_center_w = None
        self.monitor_normal_w = None
        self.units_per_cm = None
        self.monitor_vertical_offset = 0.0  # Offset vertical do plano (em unidades)
        self.monitor_corners_original = None  # Posições originais
        self.monitor_center_original = None
        
        # Lógica do marcador de gaze
        self.gaze_markers = [] # Marcadores de gaze no plano
        self._last_combined_dir = None
        self._last_combined_origin = None

        # Dados do gaze com clamp para marcação
        self._last_gaze_a = None
        self._last_gaze_b = None
        self._last_gaze_point_3d = None
        self._last_gaze_was_clamped = False
    
    def initialize_camera(self) -> bool:
        """Inicializa câmera (local ou DroidCAM Client)"""
        try:
            # Usar DroidCam Client (câmera virtual)
            # Conecte primeiro pelo DroidCam Client, depois execute este código
            use_droidcam_client = True  # Mude para False para usar webcam local
            droidcam_index = 1  # Geralmente índice 1, mas pode ser 0, 2, etc.

            if use_droidcam_client:

                # DroidCam geralmente fica nos índices 1, 2 ou 3 (não no 0 que é a webcam)
                # Se souber o índice exato, coloque-o primeiro na lista
                for idx in [1, 2, 3, 4]:
                    self.cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
                    if self.cap.isOpened():
                        ret, _ = self.cap.read()
                        if ret:
                            break
                        else:
                            self.cap.release()

                if not self.cap.isOpened():
                    return False
            else:
                self.cap = cv2.VideoCapture(self.config.hardware.camera_id)

            if not self.cap.isOpened():
                return False

            width, height = self.config.hardware.resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.cap.set(cv2.CAP_PROP_FPS, self.config.hardware.target_fps)

            return True
        except Exception:
            return False
    
    def process_frame(self, frame):
        """Processa frame com TUDO integrado"""
        frame_start_time = time.time()
        h, w = frame.shape[:2]
        
        # Resultados
        best_face = None
        face_confidence = 0.0
        landmarks = None
        iris_data = {}
        gaze_vectors = {}
        gaze_vectors_3d = {}
        screen_point = None
        filtered_screen_point = None
        fatigue_state = None
        
        try:
            # === 1. DETECÇÃO FACIAL ===
            # 'gray' é necessário para get_landmarks
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if self.performance_optimizer.should_detect_face():
                faces, confidences = self.face_detector.detect_faces(frame)
                self.face_detector.update_face_history(faces, confidences)
            
            best_face, face_confidence = self.face_detector.get_best_face()
            
            if best_face is not None:
                # === 2. LANDMARKS (DLIB) ===
                # Tenta obter os landmarks, mas não é mais um bloqueio
                landmarks = self.face_detector.get_landmarks(gray, best_face)
                
                # === 3. ÍRIS (com MediaPipe se habilitado) ===
                # Esta função é robusta para 'landmarks=None'
                iris_data = self.iris_tracker.track_iris(frame, landmarks)
                
                # === 4. POSE 3D (PCA) ===
                if self.config.algorithm.use_pca_pose:
                    if (hasattr(self.iris_tracker, 'improved_detection') and
                        self.iris_tracker.improved_detection is not None):

                        # A lógica interna do iris_tracker já processou o mediapipe
                        # Esta lógica busca esse resultado
                        if hasattr(self.iris_tracker.improved_detection, 'face_mesh_results') and self.iris_tracker.improved_detection.face_mesh_results:
                            results = self.iris_tracker.improved_detection.face_mesh_results

                            if results and results.multi_face_landmarks:
                                face_landmarks_mp = results.multi_face_landmarks[0].landmark

                                head_center, R_head, nose_points = self.face_detector.get_head_pose_pca(
                                    face_landmarks_mp, w, h
                                )

                                if head_center is not None:
                                    self.current_head_center = head_center
                                    self.current_R_head = R_head
                                    self.current_scale = self.face_detector.improved_pose.compute_scale(nose_points)

                # === 5. GAZE (3D se calibrado, senão 2D) ===
                # Esta função é robusta para 'landmarks=None'
                if (self.config.algorithm.use_eye_spheres and 
                    self.spheres_calibrated and
                    self.current_head_center is not None):
                    
                    gaze_vectors_3d = self.gaze_calculator.calculate_gaze_3d(
                        iris_data, 
                        self.current_head_center, 
                        self.current_R_head, 
                        self.current_scale
                    )
                    
                    if gaze_vectors_3d:
                        gaze_vectors = gaze_vectors_3d
                    else:
                        gaze_vectors = self.gaze_calculator.calculate_gaze(
                            iris_data, landmarks, frame, best_face
                        )
                else:
                    gaze_vectors = self.gaze_calculator.calculate_gaze(
                        iris_data, landmarks, frame, best_face
                    )
                
                # === 6. SCREEN POINT ===
                if gaze_vectors and 'left' in gaze_vectors and 'right' in gaze_vectors:
                    try:
                        raw_screen_point = None

                        # NOVO: Usar sistema 3D se disponível e tiver gaze 3D
                        if self.use_3d_mapping and gaze_vectors_3d and self.spheres_calibrated:
                            # Sistema 3D com interseção raio-plano
                            raw_screen_point = self.gaze_to_screen_3d.map_gaze_to_screen(gaze_vectors_3d)
                            if raw_screen_point and self.use_refinement:
                                if self.refinement_calibrator.is_calibrated():
                                    corrected = self.refinement_calibrator.correct_gaze(
                                        raw_screen_point[0], raw_screen_point[1]
                                    )
                                    if corrected:
                                        raw_screen_point = corrected

                        # Fallback para sistema 2D (antigo)
                        elif self.screen_calibrator.is_calibrated():
                            avg_yaw = (gaze_vectors['left']['yaw'] + gaze_vectors['right']['yaw']) / 2
                            avg_pitch = (gaze_vectors['left']['pitch'] + gaze_vectors['right']['pitch']) / 2
                            raw_screen_point = self.screen_calibrator.map_gaze_to_screen(avg_yaw, avg_pitch)

                        # === 6.5. SUAVIZAÇÃO POR MÉDIA MÓVEL (MonitorTracking.py) ===
                        if raw_screen_point and self.gaze_smoothing_enabled:
                            # Adicionar ao buffer
                            self.screen_position_buffer.append(raw_screen_point)

                            # Calcular média das posições no buffer
                            if len(self.screen_position_buffer) > 0:
                                avg_x = np.mean([p[0] for p in self.screen_position_buffer])
                                avg_y = np.mean([p[1] for p in self.screen_position_buffer])
                                screen_point = (int(avg_x), int(avg_y))
                            else:
                                screen_point = raw_screen_point
                        else:
                            screen_point = raw_screen_point

                        # === 7. KALMAN FILTER ===
                        if self.use_kalman and screen_point:
                            filtered_screen_point = self.kalman_filter.update(screen_point)
                        else:
                            filtered_screen_point = screen_point

                    except Exception:
                        pass
        
        except Exception as e:
            pass  # Erros são registrados silenciosamente
        
        processing_time = (time.time() - frame_start_time) * 1000
        
        return {
            'best_face': best_face,
            'face_confidence': face_confidence,
            'landmarks': landmarks,
            'iris_data': iris_data,
            'gaze_vectors': gaze_vectors,
            'gaze_vectors_3d': gaze_vectors_3d,
            'screen_point': screen_point,
            'filtered_point': filtered_screen_point,
            'fatigue_state': fatigue_state,
            'processing_time': processing_time,
            'head_center': self.current_head_center,
            'head_rotation': self.current_R_head,
            'scale_ratio': self.face_detector.get_scale_ratio() if self.config.algorithm.use_pca_pose else None,
            'spheres_calibrated': self.spheres_calibrated
        }
    
    def _apply_monitor_vertical_offset(self):
        """Aplica o offset vertical ao plano do monitor"""
        if self.monitor_corners_original is None or self.monitor_center_original is None:
            return

        # Vetor vertical (para cima no mundo)
        world_up = np.array([0, -1, 0], dtype=float)
        offset_vector = world_up * self.monitor_vertical_offset

        # Aplicar offset às cópias originais
        self.monitor_center_w = self.monitor_center_original.copy() + offset_vector
        self.monitor_corners = [corner.copy() + offset_vector for corner in self.monitor_corners_original]

    def create_monitor_plane(self, head_center, R_final, face_landmarks_mp, w, h,
                             forward_hint=None, gaze_origin=None, gaze_dir=None):
        """
        Cria um plano de monitor 3D a 50cm NA FRENTE da cabeça.
        Baseado em Webcam3DTracker - mantém geometria fixa relativa à pose da cabeça.
        """
        try:
            # 1) Estimar escala (unidades por cm)
            lm_chin = face_landmarks_mp[152]
            lm_fore = face_landmarks_mp[10]
            chin_w = np.array([lm_chin.x * w,  lm_chin.y * h,  lm_chin.z * w], dtype=float)
            fore_w = np.array([lm_fore.x * w,  lm_fore.y * h,  lm_fore.z * w], dtype=float)
            face_h_units = np.linalg.norm(fore_w - chin_w)
            upc = face_h_units / 15.0  # Assumindo 15cm de altura
            self.units_per_cm = upc
        except Exception as e:
            upc = 5.0 # Fallback
            self.units_per_cm = upc

        # 2) Geometria do Monitor (em cm, depois convertido para unidades)
        mon_w_cm, mon_h_cm = (self.gaze_to_screen_3d.width_mm / 10.0, self.gaze_to_screen_3d.height_mm / 10.0)

        half_w = (mon_w_cm * 0.5) * upc
        half_h = (mon_h_cm * 0.5) * upc

        # 3) Determinar direção para frente
        dist_cm = 50.0

        # Coluna 2 de R_final é o eixo Z (frente/trás)
        head_forward = -R_final[:, 2]

        # Usar forward_hint se fornecido (direção do gaze na calibração)
        if forward_hint is not None:
            head_forward = forward_hint / np.linalg.norm(forward_hint)

        # Centro do monitor: 50cm na frente da cabeça
        center_w = head_center + head_forward * (dist_cm * upc)

        # 4) Calcular cantos do monitor
        world_up = np.array([0, -1, 0], dtype=float)
        head_right = np.cross(world_up, head_forward)
        head_right /= np.linalg.norm(head_right)
        head_up = np.cross(head_forward, head_right)
        head_up /= np.linalg.norm(head_up)

        p0 = center_w - head_right * half_w - head_up * half_h
        p1 = center_w + head_right * half_w - head_up * half_h
        p2 = center_w + head_right * half_w + head_up * half_h
        p3 = center_w - head_right * half_w + head_up * half_h

        self.monitor_corners = [p0, p1, p2, p3]
        self.monitor_center_w = center_w
        self.monitor_normal_w = head_forward / (np.linalg.norm(head_forward) + 1e-9)

        # Salvar posições originais para ajustes posteriores
        self.monitor_corners_original = [p.copy() for p in self.monitor_corners]
        self.monitor_center_original = center_w.copy()
        self.monitor_vertical_offset = 0.0  # Resetar offset

        # Congelar pivô da câmera
        self.debug_world_frozen = True
        self.orbit_pivot_frozen = self.monitor_center_w.copy()


    def render_debug_view_orbit(self, h, w, result_data):
        """
        Renderiza a cena 3D com uma câmera orbital.
        Adaptado de MonitorTracking.py e modificado para ser um método de classe.
        """
        
        # Extrai dados do frame processado
        head_center3d = result_data.get('head_center')
        gaze_vectors_3d = result_data.get('gaze_vectors_3d', {})
        
        # Dados do olho esquerdo
        left_data = gaze_vectors_3d.get('left', {})
        sphere_world_l = left_data.get('origin')
        iris3d_l = left_data.get('iris_center_3d') # <-- Graças ao Passo 3!
        left_dir = left_data.get('direction')
        
        # Dados do olho direito
        right_data = gaze_vectors_3d.get('right', {})
        sphere_world_r = right_data.get('origin')
        iris3d_r = right_data.get('iris_center_3d')
        right_dir = right_data.get('direction')

        left_locked = result_data.get('spheres_calibrated', False)
        right_locked = result_data.get('spheres_calibrated', False)
        
        # Landmarks (se disponíveis)
        landmarks3d = None
        if self.iris_tracker.improved_detection and self.iris_tracker.improved_detection.face_mesh_results:
             if self.iris_tracker.improved_detection.face_mesh_results.multi_face_landmarks:
                lm = self.iris_tracker.improved_detection.face_mesh_results.multi_face_landmarks[0].landmark
                landmarks3d = np.array([[p.x * w, p.y * h, p.z * w] for p in lm], dtype=float)

        # --- Criação da imagem de debug ---
        debug_frame = np.zeros((h, w, 3), dtype=np.uint8)
        if head_center3d is None:
            cv2.putText(debug_frame, "Sem dados 3D", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return debug_frame

        head_w = np.asarray(head_center3d, dtype=float)

        # --- Escolher pivô da órbita ---
        if self.debug_world_frozen and self.orbit_pivot_frozen is not None:
            pivot_w = np.asarray(self.orbit_pivot_frozen, dtype=float)
        else:
            pivot_w = head_w

        # --- Posição da Câmera ---
        f_px = _focal_px(w, self.orbit_fov_deg)
        cam_offset = _rot_y(self.orbit_yaw) @ (_rot_x(self.orbit_pitch) @ np.array([0.0, 0.0, self.orbit_radius]))
        cam_pos = pivot_w + cam_offset

        up_world = np.array([0.0, -1.0, 0.0]) # Y do mundo aponta para cima
        fwd = _normalize(pivot_w - cam_pos)
        right = _normalize(np.cross(fwd, up_world))
        up = _normalize(np.cross(right, fwd))
        V = np.stack([right, up, fwd], axis=0) # Matriz de Visualização

        def project_point(P):
            Pw = np.asarray(P, dtype=float)
            Pc = V @ (Pw - cam_pos) # Ponto no espaço da câmera
            if Pc[2] <= 1e-3: return None # Atrás da câmera
            x = f_px * (Pc[0] / Pc[2]) + w * 0.5
            y = -f_px * (Pc[1] / Pc[2]) + h * 0.5 # Y da imagem é invertido
            if not (np.isfinite(x) and np.isfinite(y)): return None
            return (int(x), int(y)), Pc[2] # Retorna (ponto_2d, profundidade)

        # --- Funções de Desenho 3D ---
        def draw_poly_3d(pts, color=(0, 200, 255), thickness=2):
            projs = [project_point(p) for p in pts]
            if any(p is None for p in projs): return
            p2 = [p[0] for p in projs]
            for a, b in zip(p2, p2[1:] + [p2[0]]):
                cv2.line(debug_frame, a, b, color, thickness)

        def draw_cross_3d(P, size=12, color=(255, 0, 255), thickness=2):
            res = project_point(P)
            if res is None: return
            (x, y), _ = res
            cv2.line(debug_frame, (x - size, y), (x + size, y), color, thickness)
            cv2.line(debug_frame, (x, y - size), (x, y + size), color, thickness)

        def draw_arrow_3d(P0, P1, color=(0, 200, 255), thickness=3):
            a = project_point(P0); b = project_point(P1)
            if a is None or b is None: return
            p0, p1 = a[0], b[0]
            cv2.line(debug_frame, p0, p1, color, thickness)
            # (Desenho da ponta da seta omitido por brevidade)

        # --- Desenhar Landmarks ---
        if landmarks3d is not None:
            for P in landmarks3d[::5]: # Desenha 1 em cada 5 para performance
                res = project_point(P)
                if res is not None:
                    cv2.circle(debug_frame, res[0], 0, (150, 150, 150), -1)

        # --- Desenhar Centro da Cabeça ---
        draw_cross_3d(head_w, size=12, color=(255, 0, 255), thickness=2)

        # --- Desenhar Olhos e Gaze ---
        gaze_len = 1000 # Comprimento do raio de gaze em mm
        combined_dir = None
        
        if left_locked and sphere_world_l is not None:
            res = project_point(sphere_world_l)
            if res is not None:
                (cx, cy), z = res
                r_px = max(2, int((12.0) * f_px / max(z, 1e-3))) # 12.0 = raio real do olho em mm
                cv2.circle(debug_frame, (cx, cy), r_px, (255, 255, 25), 1)
                if iris3d_l is not None and left_dir is not None:
                    p1 = project_point(np.asarray(sphere_world_l) + _normalize(left_dir) * gaze_len)
                    if p1 is not None:
                        cv2.line(debug_frame, (cx, cy), p1[0], (155, 155, 25), 1)

        if right_locked and sphere_world_r is not None:
            res = project_point(sphere_world_r)
            if res is not None:
                (cx, cy), z = res
                r_px = max(2, int((12.0) * f_px / max(z, 1e-3))) # 12.0 = raio real do olho em mm
                cv2.circle(debug_frame, (cx, cy), r_px, (25, 255, 255), 1)
                if iris3d_r is not None and right_dir is not None:
                    p1 = project_point(np.asarray(sphere_world_r) + _normalize(right_dir) * gaze_len)
                    if p1 is not None:
                        cv2.line(debug_frame, (cx, cy), p1[0], (25, 155, 155), 1)
        
        # Gaze combinado com validação de qualidade
        if left_dir is not None and right_dir is not None:
            # Usar sistema de fusão binocular
            fusion_result = self.binocular_fusion.fuse(
                origin_left=sphere_world_l,
                direction_left=left_dir,
                origin_right=sphere_world_r,
                direction_right=right_dir
            )
            
            origin_mid = fusion_result.combined_origin
            combined_dir = fusion_result.combined_direction
            
            # Informações adicionais para debug
            fusion_method = fusion_result.method.value
            vergence_quality = fusion_result.vergence_quality
            discrepancy_angle = fusion_result.discrepancy_angle
            fusion_confidence = fusion_result.confidence
            
            # (Opcional) Atualizar cor da linha baseado na confiança
            if fusion_confidence > 0.8:
                line_color = (155, 200, 10)  # Verde - alta confiança
            elif fusion_confidence > 0.5:
                line_color = (0, 200, 255)   # Amarelo - média
            else:
                line_color = (0, 0, 255)     # Vermelho - baixa

            self._last_combined_dir = combined_dir.copy()
            self._last_combined_origin = origin_mid.copy()

            p0 = project_point(origin_mid)
            p1 = project_point(origin_mid + _normalize(combined_dir) * (gaze_len * 1.2))
            if p0 is not None and p1 is not None:
                # Cor da linha indica qualidade
                quality = gaze_vectors_3d.get('quality', 'unknown') if gaze_vectors_3d else 'unknown'
                line_color = (155, 200, 10) if quality == 'good' else (0, 165, 255)  # Laranja se poor
                cv2.line(debug_frame, p0[0], p1[0], line_color, 2)

        # --- DEBUG: Desenhar Íris para verificar paralelismo ---
        if iris3d_l is not None and iris3d_r is not None:
            # Desenhar íris esquerda
            res_l = project_point(iris3d_l)
            if res_l is not None:
                cv2.circle(debug_frame, res_l[0], 8, (255, 255, 0), -1)  # Amarelo
                cv2.putText(debug_frame, "L-Iris", (res_l[0][0] + 10, res_l[0][1]),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

            # Desenhar íris direita
            res_r = project_point(iris3d_r)
            if res_r is not None:
                cv2.circle(debug_frame, res_r[0], 8, (0, 255, 255), -1)  # Ciano
                cv2.putText(debug_frame, "R-Iris", (res_r[0][0] + 10, res_r[0][1]),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

            # Desenhar linha conectando as íris (deve ser horizontal se paralelas)
            if res_l is not None and res_r is not None:
                cv2.line(debug_frame, res_l[0], res_r[0], (255, 0, 255), 1)

            # Calcular e mostrar ângulo entre os vetores de gaze
            if left_dir is not None and right_dir is not None:
                dot_product = np.clip(np.dot(left_dir, right_dir), -1.0, 1.0)
                angle_deg = np.degrees(np.arccos(dot_product))
                cv2.putText(debug_frame, f"Angulo entre iris: {angle_deg:.1f} graus",
                           (10, h - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                if angle_deg > 10:
                    cv2.putText(debug_frame, "DIVERGENTES!",
                               (10, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # --- Desenhar Plano do Monitor ---
        if self.monitor_corners is not None:
            draw_poly_3d(self.monitor_corners, (0, 200, 255), 2)
            draw_cross_3d(self.monitor_center_w, size=8, color=(0, 200, 255), thickness=2)
            if self.monitor_normal_w is not None:
                tip = np.asarray(self.monitor_center_w) + np.asarray(self.monitor_normal_w) * (20.0 * (self.units_per_cm or 1.0))
                draw_arrow_3d(self.monitor_center_w, tip, color=(0, 220, 255), thickness=2)

        # --- Desenhar Marcadores de Gaze Salvos ('x') ---
        if (self.gaze_markers and self.monitor_corners is not None):
            p0, p1, p2, p3 = [np.asarray(p, dtype=float) for p in self.monitor_corners]
            u = p1 - p0
            v = p3 - p0
            for (a, b) in self.gaze_markers:
                Pm = p0 + a * u + b * v
                draw_cross_3d(Pm, size=5, color=(0, 255, 0), thickness=1)

        origin_mid = None
        combined_dir = None
        
        if gaze_vectors_3d:
            left_data = gaze_vectors_3d.get('left', {})
            right_data = gaze_vectors_3d.get('right', {})
            
            left_origin = left_data.get('origin')
            right_origin = right_data.get('origin')
            left_dir = left_data.get('direction')
            right_dir = right_data.get('direction')
            
            # Média simples
            if left_origin is not None and right_origin is not None:
                origin_mid = (left_origin + right_origin) / 2.0
                combined_dir = _normalize(left_dir + right_dir)
            elif left_origin is not None:
                origin_mid = left_origin
                combined_dir = _normalize(left_dir)
            elif right_origin is not None:
                origin_mid = right_origin
                combined_dir = _normalize(right_dir)

        # --- Desenhar Interseção Atual do Gaze (Círculo Amarelo) ---
        if (self.monitor_corners is not None and combined_dir is not None and left_locked):
            O = origin_mid if 'origin_mid' in locals() else None
            D = _normalize(np.asarray(combined_dir, dtype=float))

            if O is not None:
                C = np.asarray(self.monitor_center_w, dtype=float)
                N = _normalize(np.asarray(self.monitor_normal_w, dtype=float))

                denom = float(np.dot(N, D))
                if abs(denom) > 1e-6:
                    t = float(np.dot(N, (C - O)) / denom)
                    if t > 0.0:
                        P = O + t * D  # Ponto de interseção 3D
                        
                        # Calcular coordenadas normalizadas
                        p0, p1, p2, p3 = [np.asarray(p, dtype=float) for p in self.monitor_corners]
                        u = p1 - p0
                        v = p3 - p0
                        wv = P - p0
                        u_len2 = float(np.dot(u, u))
                        v_len2 = float(np.dot(v, v))
                        
                        if u_len2 > 1e-9 and v_len2 > 1e-9:
                            a = float(np.dot(wv, u) / u_len2)
                            b = float(np.dot(wv, v) / v_len2)
                            
                            # ========== CLAMP ÀS BORDAS DO MONITOR ==========
                            was_clamped = False
                            
                            # Clamp horizontal
                            if a < 0.0:
                                a = 0.0
                                was_clamped = True
                            elif a > 1.0:
                                a = 1.0
                                was_clamped = True
                            
                            # Clamp vertical
                            if b < 0.0:
                                b = 0.0
                                was_clamped = True
                            elif b > 1.0:
                                b = 1.0
                                was_clamped = True
                            
                            # Reconstruir ponto 3D (sempre dentro do monitor)
                            P_clamped = p0 + a * u + b * v
                            
                            # Salvar para marcação (tecla 'x')
                            self._last_gaze_a = a
                            self._last_gaze_b = b
                            self._last_gaze_point_3d = P_clamped
                            self._last_gaze_was_clamped = was_clamped
                            
                            # Desenhar círculo (cor indica se foi clamped)
                            res = project_point(P_clamped)
                            if res is not None:
                                if was_clamped:
                                    # Laranja quando na borda
                                    cv2.circle(debug_frame, res[0], 10, (0, 165, 255), 2)
                                    cv2.putText(debug_frame, "BORDA", 
                                               (res[0][0] + 15, res[0][1]),
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)
                                else:
                                    # Amarelo quando dentro do monitor
                                    cv2.circle(debug_frame, res[0], 10, (0, 255, 255), 2)
        
                

        return debug_frame

    def _mouse_callback_3d(self, event, x, y, flags, param):
        """Callback de mouse para controle da câmera 3D"""

        if event == cv2.EVENT_LBUTTONDOWN:
            # Começar drag
            self.mouse_dragging = True
            self.mouse_last_x = x
            self.mouse_last_y = y

        elif event == cv2.EVENT_LBUTTONUP:
            # Parar drag
            self.mouse_dragging = False

        elif event == cv2.EVENT_MOUSEMOVE:
            # Arrastar para rotacionar
            if self.mouse_dragging:
                dx = x - self.mouse_last_x
                dy = y - self.mouse_last_y

                # Atualizar ângulos de órbita
                self.orbit_yaw += dx * self.mouse_sensitivity
                self.orbit_pitch -= dy * self.mouse_sensitivity  # Invertido para ser intuitivo

                # Limitar pitch
                self.orbit_pitch = max(math.radians(-89), min(math.radians(89), self.orbit_pitch))

                # Atualizar última posição
                self.mouse_last_x = x
                self.mouse_last_y = y

        elif event == cv2.EVENT_MOUSEWHEEL:
            # Scroll para zoom
            if flags > 0:  # Scroll up
                self.orbit_radius = max(80.0, self.orbit_radius - self.zoom_sensitivity)
            else:  # Scroll down
                self.orbit_radius = min(3000.0, self.orbit_radius + self.zoom_sensitivity)

    def _handle_debug_keys(self, key, result_data):
        """Lida com as teclas de controle do mundo virtual 3D"""

        changed = False

        # Resetar Câmera
        if key == ord('r'):
            self.orbit_yaw = math.radians(-15.0)
            self.orbit_pitch = math.radians(-15.0)
            self.orbit_radius = 1200.0
            changed = True

        # Travar Plano do Monitor
        elif key == ord('p'):
            if result_data.get('head_center') is not None and \
               result_data.get('head_rotation') is not None and \
               self.iris_tracker.improved_detection and \
               self.iris_tracker.improved_detection.face_mesh_results and \
               result_data.get('spheres_calibrated'):

                gaze_vectors_3d = result_data.get('gaze_vectors_3d', {})
                left_data = gaze_vectors_3d.get('left', {})
                right_data = gaze_vectors_3d.get('right', {})

                # Usar lógica de validação de qualidade
                use_single = gaze_vectors_3d.get('use_single_eye', None)
                if use_single == 'right':
                    gaze_origin = right_data.get('origin')
                    gaze_dir = _normalize(right_data.get('direction'))
                elif use_single == 'left':
                    gaze_origin = left_data.get('origin')
                    gaze_dir = _normalize(left_data.get('direction'))
                else:
                    gaze_origin = (left_data.get('origin') + right_data.get('origin')) / 2.0
                    gaze_dir = _normalize(left_data.get('direction') + right_data.get('direction'))
                
                face_landmarks_mp = self.iris_tracker.improved_detection.face_mesh_results.multi_face_landmarks[0].landmark
                h, w = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT), self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                
                self.create_monitor_plane(
                    result_data['head_center'],
                    result_data['head_rotation'],
                    face_landmarks_mp, w, h,
                    forward_hint=gaze_dir,
                    gaze_origin=gaze_origin,
                    gaze_dir=gaze_dir
                )

        # Marcar Ponto de Gaze
        elif key == ord('x'):
            if (self.monitor_corners is not None and 
                result_data.get('spheres_calibrated') and
                self._last_gaze_a is not None and 
                self._last_gaze_b is not None):
                
                # Usar coordenadas já calculadas (com clamp aplicado)
                a = self._last_gaze_a
                b = self._last_gaze_b
                was_clamped = self._last_gaze_was_clamped
                
                # Adicionar marcador
                self.gaze_markers.append((a, b))
                
                # Feedback visual
                if was_clamped:
                    print(f"[Marker] Adicionado na BORDA: a={a:.3f}, b={b:.3f}")
                else:
                    print(f"[Marker] Adicionado: a={a:.3f}, b={b:.3f}")
            else:
                if self._last_gaze_a is None:
                    print("[Marker] Aguardando gaze válido")
                else:
                    print("[Marker] Calibração não completa")

        if changed:
            self.orbit_pitch = max(math.radians(-89), min(math.radians(89), self.orbit_pitch))

    def run_calibration(self):
        """
        Executa calibração MANUAL com confirmação por ESPAÇO
        Muito mais preciso que o sistema automático!
        """
        if not self.cap or not self.cap.isOpened():
            return False
        
        # Iniciar calibração manual
        self.screen_calibrator.start_calibration()
        
        # Loop de calibração
        calibration_window = 'Manual Calibration'
        
        while not self.screen_calibrator.calibration_complete:
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            # Processar frame para obter gaze
            results = self.process_frame(frame)
            
            # Coletar amostra se houver gaze detectado
            if results and results.get('gaze_vectors'):
                gaze = results['gaze_vectors']
                if 'left' in gaze and 'right' in gaze:
                    # Média dos dois olhos
                    avg_yaw = (gaze['left']['yaw'] + gaze['right']['yaw']) / 2
                    avg_pitch = (gaze['left']['pitch'] + gaze['right']['pitch']) / 2
                    
                    # Passar dados para o calibrador manual
                    gaze_data = {'yaw': avg_yaw, 'pitch': avg_pitch}
                    
                    # Coletar amostra (só coleta se ESPAÇO pressionado)
                    should_advance = self.screen_calibrator.collect_gaze_sample(gaze_data)
            
            # Exibir ponto de calibração
            calibration_screen = self.screen_calibrator.display_calibration_point()
            cv2.imshow(calibration_window, calibration_screen)
            
            # Verificar se foi cancelado
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or self.screen_calibrator.calibration_aborted:  # ESC
                cv2.destroyWindow(calibration_window)
                return False
        
        cv2.destroyWindow(calibration_window)

    def run_validation(self):
        """
        Executa o modo de validação com chessboard.
        
        CORREÇÃO: 
        - Usa coordenadas invertidas para a TELA (cursor)
        - Usa coordenadas originais para o HEATMAP (visualização)
        """
        if not self.spheres_calibrated:
            print("[Validação] ERRO: Calibre as esferas primeiro (tecla 'c')")
            return
        
        if self.monitor_corners is None:
            print("[Validação] ERRO: Calibre o monitor virtual primeiro (tecla 'p')")
            return
        
        screen_w = self.validator.screen_width
        screen_h = self.validator.screen_height
        
        cv2.namedWindow(self.validation_window_name, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(self.validation_window_name, 
                             cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        self.validator.start_session(validate_all=True)
        
        print("=" * 60)
        print("[Validação] SESSÃO INICIADA")
        print("[Validação] Olhe para a célula VERDE e pressione ESPAÇO")
        print("=" * 60)
        
        debug_info = {"frames": 0, "gaze_ok": 0, "samples": 0}
        
        while self.validator.is_running:
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            debug_info["frames"] += 1
            results = self.process_frame(frame)
            state = self.validator.update()
            
            current_gaze = None
            norm_a_heatmap, norm_b_heatmap = None, None
            
            if results and results.get('gaze_vectors_3d'):
                gaze_3d = results['gaze_vectors_3d']
                left_data = gaze_3d.get('left', {})
                right_data = gaze_3d.get('right', {})
                
                left_origin = left_data.get('origin')
                right_origin = right_data.get('origin')
                left_dir = left_data.get('direction')
                right_dir = right_data.get('direction')
                
                if (left_origin is not None and right_origin is not None and
                    left_dir is not None and right_dir is not None):
                    
                    origin_mid = (np.asarray(left_origin) + np.asarray(right_origin)) / 2.0
                    
                    combined_dir = np.asarray(left_dir) + np.asarray(right_dir)
                    norm = np.linalg.norm(combined_dir)
                    if norm > 1e-9:
                        combined_dir = combined_dir / norm
                    else:
                        combined_dir = None
                    
                    if combined_dir is not None and self.monitor_corners is not None:
                        O = origin_mid
                        D = combined_dir
                        C = np.asarray(self.monitor_center_w, dtype=float)
                        N = self.monitor_normal_w / np.linalg.norm(self.monitor_normal_w)
                        
                        denom = float(np.dot(N, D))
                        if abs(denom) > 1e-6:
                            t = float(np.dot(N, (C - O)) / denom)
                            if t > 0.0:
                                P = O + t * D
                                
                                p0, p1, p2, p3 = [np.asarray(p, dtype=float) for p in self.monitor_corners]
                                u = p1 - p0
                                v = p3 - p0
                                
                                u_len2 = float(np.dot(u, u))
                                v_len2 = float(np.dot(v, v))
                                
                                if u_len2 > 1e-9 and v_len2 > 1e-9:
                                    wv = P - p0
                                    
                                    # Coordenadas ORIGINAIS do monitor virtual
                                    a_original = float(np.dot(wv, u) / u_len2)
                                    b_original = float(np.dot(wv, v) / v_len2)
                                    
                                    # Clamp às bordas
                                    a_original = max(0.0, min(1.0, a_original))
                                    b_original = max(0.0, min(1.0, b_original))
                                    
                                    # ========================================
                                    # COORDENADAS PARA A TELA (invertidas)
                                    # ========================================
                                    a_screen = 1.0 - a_original
                                    b_screen = 1.0 - b_original
                                    
                                    screen_x = int(a_screen * screen_w)
                                    screen_y = int(b_screen * screen_h)
                                    
                                    screen_x = max(0, min(screen_w - 1, screen_x))
                                    screen_y = max(0, min(screen_h - 1, screen_y))
                                    
                                    current_gaze = (screen_x, screen_y)
                                    
                                    # ========================================
                                    # COORDENADAS PARA O HEATMAP (originais)
                                    # ========================================
                                    # O heatmap deve usar as mesmas coordenadas
                                    # que são exibidas na tela
                                    norm_a_heatmap = a_screen
                                    norm_b_heatmap = b_screen
                                    
                                    debug_info["gaze_ok"] += 1
            
            # Adicionar amostra com coordenadas corretas para o heatmap
            if current_gaze is not None and state == "capturing":
                self.validator.add_gaze_sample(
                    current_gaze[0], current_gaze[1],
                    norm_a_heatmap, norm_b_heatmap
                )
                debug_info["samples"] += 1
            
            # Renderizar
            board_img = self.validator.render_chessboard(
                current_gaze=current_gaze,
                show_gaze=True
            )
            
            info_text = f"Gaze: {debug_info['gaze_ok']}/{debug_info['frames']} | Samples: {debug_info['samples']}"
            cv2.putText(board_img, info_text, 
                       (board_img.shape[1] - 400, board_img.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
            
            cv2.imshow(self.validation_window_name, board_img)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:
                self.validator.cancel_session()
                break
            elif key == ord(' '):
                if state == "waiting_ready":
                    self.validator.user_ready()
                    debug_info["samples"] = 0
            elif key == ord('s') or key == ord('S'):
                if state in ["waiting_ready", "capturing"]:
                    self.validator.skip_current_cell()
                    debug_info["samples"] = 0
            elif state == "finished":
                report = self.validator.generate_report_image()
                cv2.imshow(self.validation_window_name, report)
                cv2.waitKey(0)
                self.validator.save_results("validation_results")
                break
        
        cv2.destroyWindow(self.validation_window_name)
        
        if self.validator.current_session and self.validator.current_session.cell_results:
            s = self.validator.current_session
            total = sum(r.num_samples for r in s.cell_results.values())
            print(f"\\nRESUMO: {len(s.cell_results)} células | {total} amostras | {s.overall_accuracy:.1f}% precisão")
    
    def calibrate_eye_spheres(self, iris_data):
        """Calibra esferas oculares 3D"""

        try:
            success = self.gaze_calculator.calibrate_eye_spheres(
                iris_data, self.current_head_center, self.current_R_head, self.current_scale
            )

            if success:
                if hasattr(self.face_detector, 'calibrate_pose_scale'):
                    self.face_detector.calibrate_pose_scale()

                self.spheres_calibrated = True
                return True

        except Exception as e:
            pass  # Erros silenciosos

        return False
    
    def run_refinement_calibration(self):
        """Executa calibração de refinamento com 9 pontos."""
        if not self.spheres_calibrated:
            print("❌ Calibre as esferas primeiro (tecla 'c')")
            return False
        
        if self.monitor_corners is None:
            print("❌ Calibre o monitor primeiro (tecla 'p')")
            return False
        
        self.refinement_calibrator.start_calibration()
        
        calibration_window = 'Refinement Calibration'
        cv2.namedWindow(calibration_window, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(calibration_window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        while self.refinement_calibrator.calibration_active:
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            # Processar frame
            results = self.process_frame(frame)
            
            # Se está coletando, adicionar amostras
            if self.refinement_calibrator.is_collecting:
                screen_point = results.get('screen_point') or results.get('filtered_point')
                if screen_point:
                    self.refinement_calibrator.add_gaze_sample(screen_point[0], screen_point[1])
            
            # Exibir tela de calibração
            calib_screen = self.refinement_calibrator.display_calibration_point()
            cv2.imshow(calibration_window, calib_screen)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC
                self.refinement_calibrator.calibration_active = False
                break
            elif key == 32:  # ESPAÇO
                if not self.refinement_calibrator.is_collecting:
                    self.refinement_calibrator.start_collection()
                else:
                    self.refinement_calibrator.stop_collection()
        
        cv2.destroyWindow(calibration_window)
        
        if self.refinement_calibrator.is_calibrated():
            self.use_refinement = True
            print("✅ Refinamento ativado!")
            return True
        return False
    
    def check_calibration_quality(self):
        """Verifica e reporta a qualidade da calibração atual"""
        quality = self.screen_calibrator.get_calibration_quality()
        return quality
    
    def _calculate_gaze_velocity(self, gaze_vectors):
        """Velocidade de gaze (para fadiga)"""
        if not self._last_gaze or not gaze_vectors:
            self._last_gaze = gaze_vectors
            return 0.0
        
        try:
            left_diff = abs(gaze_vectors['left']['yaw'] - self._last_gaze['left']['yaw'])
            right_diff = abs(gaze_vectors['right']['yaw'] - self._last_gaze['right']['yaw'])
            velocity = (left_diff + right_diff) / 2
            self._last_gaze = gaze_vectors
            return velocity
        except:
            return 0.0
    
    def run(self):
        """Loop principal"""
        if not self.initialize_camera():
            if self.session_logger:
                self.session_logger.log_event("ERRO", "Falha ao inicializar câmera")
            return

        if self.session_logger:
            self.session_logger.log_event("CAMERA", "Câmera inicializada com sucesso")

        self.is_running = True

        while self.is_running:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                time.sleep(0.05) # Evita loop 100% CPU se a câmera morrer
                continue # Pula para a próxima iteração

            result = self.process_frame(frame)

            # Logar dados do frame (se logger disponível)
            if self.data_logger:
                self.data_logger.log_frame(result)

            vis_frame = self.visualization_manager.render_frame(
                frame, result, self.is_calibrating, None,
                self.screen_calibrator, self.current_fps
            )

            self._draw_info_overlay(vis_frame, result)

            # Desenhar alvo 3D do gaze (se habilitado)
            if self.show_3d_gaze_target and result.get('screen_point'):
                # Pegar resolução da tela do gaze_to_screen_3d
                screen_res = (self.gaze_to_screen_3d.width_px,
                             self.gaze_to_screen_3d.height_px)

                self.visualization_manager.draw_gaze_3d_target(
                    vis_frame,
                    result['screen_point'],
                    label="3D Gaze" if self.use_3d_mapping else "2D Gaze",
                    screen_resolution=screen_res
                )

            # 3. Renderizar MUNDO VIRTUAL 3D (NOVO)
            h, w = frame.shape[:2]
            debug_frame = self.render_debug_view_orbit(h, w, result)

            cv2.imshow('Gaze Tracking - Sistema Completo', vis_frame)
            cv2.imshow('Mundo Virtual 3D (Debug)', debug_frame)

            # Configurar callback de mouse para a janela 3D (apenas na primeira vez)
            if not hasattr(self, '_mouse_callback_set'):
                cv2.setMouseCallback('Mundo Virtual 3D (Debug)', self._mouse_callback_3d)
                self._mouse_callback_set = True

            key = cv2.waitKey(1) & 0xFF

            self._handle_debug_keys(key, result)
            
            if key == ord('q'):
                break
            elif key == ord('e'):
                # Calibrar esferas oculares
                if self.session_logger:
                    self.session_logger.log_event("CALIBRACAO", "Iniciando calibração de esferas oculares")

                # Processar vários frames para garantir que a pose seja calculada
                attempts = 0
                max_attempts = 30  # Tentar por ~1 segundo (30fps)
                last_result = None

                while attempts < max_attempts:
                    ret, frame = self.cap.read()
                    if not ret:
                        attempts += 1
                        time.sleep(0.03)
                        continue

                    last_result = self.process_frame(frame)

                    attempts += 1
                    time.sleep(0.03)

                # Tentar calibrar com os últimos dados
                if last_result and last_result.get('iris_data'):
                    success = self.calibrate_eye_spheres(last_result['iris_data'])

                    if success and self.session_logger:
                        self.session_logger.log_event("CALIBRACAO", "Esferas calibradas com sucesso")
                    elif self.session_logger:
                        self.session_logger.log_event("CALIBRACAO", "Falha na calibração de esferas")

            elif key == ord('r'):
                self.screen_calibrator.reset_calibration()
                self.spheres_calibrated = False
                if self.session_logger:
                    self.session_logger.log_event("CALIBRACAO", "Calibração resetada")
            elif key == ord('k'):
                self.use_kalman = not self.use_kalman
                if self.session_logger:
                    status = "ativado" if self.use_kalman else "desativado"
                    self.session_logger.log_event("CONFIG", f"Filtro Kalman {status}")
            elif key == ord('s'):
                # Toggle suavização de gaze
                self.gaze_smoothing_enabled = not self.gaze_smoothing_enabled
                if not self.gaze_smoothing_enabled:
                    # Limpar buffers quando desabilitar
                    self.screen_position_buffer.clear()
                    self.gaze_direction_buffer.clear()
                if self.session_logger:
                    status = "ativada" if self.gaze_smoothing_enabled else "desativada"
                    self.session_logger.log_event("CONFIG", f"Suavização de gaze {status}")
            elif key == ord('['):
                # Diminuir buffer de suavização
                self.gaze_smoothing_buffer_size = max(1, self.gaze_smoothing_buffer_size - 1)
                self.screen_position_buffer = deque(self.screen_position_buffer, maxlen=self.gaze_smoothing_buffer_size)
                self.gaze_direction_buffer = deque(self.gaze_direction_buffer, maxlen=self.gaze_smoothing_buffer_size)
                if self.session_logger:
                    self.session_logger.log_event("CONFIG", f"Buffer de suavização: {self.gaze_smoothing_buffer_size}")
            elif key == ord(']'):
                # Aumentar buffer de suavização
                self.gaze_smoothing_buffer_size = min(30, self.gaze_smoothing_buffer_size + 1)
                self.screen_position_buffer = deque(self.screen_position_buffer, maxlen=self.gaze_smoothing_buffer_size)
                self.gaze_direction_buffer = deque(self.gaze_direction_buffer, maxlen=self.gaze_smoothing_buffer_size)
                if self.session_logger:
                    self.session_logger.log_event("CONFIG", f"Buffer de suavização: {self.gaze_smoothing_buffer_size}")
            elif key == ord('f'):
                self.enable_fatigue_detection = not self.enable_fatigue_detection
            elif key == ord('i'):
                self._print_3d_info()
            elif key == ord('m'):
                if self.use_kalman:
                    metrics = self.kalman_filter.get_smoothness_metrics()
            elif key == ord('d'):
                self.config.debug_enabled = not self.config.debug_enabled
                status = "ATIVADO" if self.config.debug_enabled else "DESATIVADO"
            elif key == ord('c'):
                # Toggle clamping (limitar gaze às bordas do monitor)
                self.gaze_to_screen_3d.clamp_to_screen = not self.gaze_to_screen_3d.clamp_to_screen
                if self.session_logger:
                    status = "ativado" if self.gaze_to_screen_3d.clamp_to_screen else "desativado"
                    self.session_logger.log_event("CONFIG", f"Clamping de gaze {status}")
            elif key == ord('+') or key == ord('='):
                self.config.algorithm.gaze_gain += 0.1
            elif key == ord('-') or key == ord('_'):
                self.config.algorithm.gaze_gain = max(0.1, self.config.algorithm.gaze_gain - 0.1)
            elif key == ord('g'):
                self._print_gaze_settings()
            elif key == ord('t'):
                # Alternar entre sistema 2D e 3D
                self.use_3d_mapping = not self.use_3d_mapping
            elif key == ord('o'):
                # Calibrar offset de 1 ponto (sistema 3D)
                self._calibrate_3d_offset()
            elif key == ord('v'):
                # Toggle visualização do alvo 3D
                self.show_3d_gaze_target = not self.show_3d_gaze_target
                status = "ATIVADA" if self.show_3d_gaze_target else "DESATIVADA"
            elif key == ord('h'):  # Nova tecla 'h' para Home/Center calibration
                self.run_refinement_calibration()
            elif key == ord('z'):
                # Toggle inversão do eixo X (mudado de 'x' para 'z' para evitar conflito)
                self.flip_x_axis = not self.flip_x_axis
                # Recriar o sistema 3D com a nova configuração
                root = tk.Tk()
                screen_w = root.winfo_screenwidth()
                screen_h = root.winfo_screenheight()
                root.destroy()
                self.gaze_to_screen_3d = GazeToScreen3D(screen_w, screen_h)
                status = "INVERTIDO" if self.flip_x_axis else "NORMAL"
            elif key == ord('y'):
                # Iniciar modo de validação
                self.run_validation()
            elif key == 82:  # Seta para cima
                if self.monitor_corners is not None:
                    self.monitor_vertical_offset += 5.0  # Ajustar 5 unidades
                    self._apply_monitor_vertical_offset()
            elif key == 84:  # Seta para baixo
                if self.monitor_corners is not None:
                    self.monitor_vertical_offset -= 5.0
                    self._apply_monitor_vertical_offset()

            # FPS counter
            self.fps_counter += 1
            if time.time() - self.fps_start_time >= 1.0:
                self.current_fps = self.fps_counter
                self.fps_counter = 0
                self.fps_start_time = time.time()
        
        self.cleanup()
    
    def _draw_info_overlay(self, frame, result):
        """Overlay com info dos algoritmos"""
        y_pos = 30

        # Sistema de mapeamento ativo
        if self.use_3d_mapping and self.spheres_calibrated:
            mode_text = "MODO: 3D RAIO-PLANO"
            mode_color = (0, 255, 255)  # Amarelo
        elif not self.use_3d_mapping and self.screen_calibrator.is_calibrated():
            mode_text = "MODO: 2D HOMOGRAFIA"
            mode_color = (255, 165, 0)  # Laranja
        else:
            mode_text = "MODO: NAO CALIBRADO"
            mode_color = (0, 0, 255)  # Vermelho

        cv2.putText(frame, mode_text, (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, mode_color, 2)
        y_pos += 30

        # Kalman
        if self.use_kalman:
            cv2.putText(frame, "KALMAN: ON", (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            y_pos += 25

        # Suavização de gaze
        if self.gaze_smoothing_enabled:
            cv2.putText(frame, f"SUAVIZACAO: {self.gaze_smoothing_buffer_size}", (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            y_pos += 25

        # Clamping
        if hasattr(self.gaze_to_screen_3d, 'clamp_to_screen') and self.gaze_to_screen_3d.clamp_to_screen:
            cv2.putText(frame, "CLAMP: ON", (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            y_pos += 25

        # Esferas calibradas?
        if self.spheres_calibrated:
            cv2.putText(frame, "ESFERAS: OK", (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            y_pos += 25

        # Fadiga
        if result.get('fatigue_state'):
            fatigue = result['fatigue_state']
            level = fatigue.get('level', 'unknown')

            color = {
                'alert': (0, 255, 0),
                'mild': (0, 255, 255),
                'moderate': (0, 165, 255),
                'severe': (0, 100, 255),
                'critical': (0, 0, 255)
            }.get(level, (255, 255, 255))

            cv2.putText(frame, f"FADIGA: {level.upper()}", (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y_pos += 30

    def _calibrate_3d_offset(self):
        """Calibração simplificada de offset com 1 ponto de referência"""

        # Calcular centro da tela
        center_x = self.gaze_to_screen_3d.width_px // 2
        center_y = self.gaze_to_screen_3d.height_px // 2

        # Desenhar ponto de calibração
        cal_frame = np.zeros((self.gaze_to_screen_3d.height_px,
                             self.gaze_to_screen_3d.width_px, 3), dtype=np.uint8)

        # Loop de calibração
        samples = []
        max_samples = 30

        while len(samples) < max_samples:
            ret, frame = self.cap.read()
            if not ret:
                continue

            # Processar frame
            result = self.process_frame(frame)

            # Desenhar tela de calibração
            cal_display = cal_frame.copy()

            # Ponto central
            cv2.circle(cal_display, (center_x, center_y), 20, (0, 255, 0), -1)
            cv2.circle(cal_display, (center_x, center_y), 25, (255, 255, 255), 2)

            # Crosshair
            cv2.line(cal_display, (center_x - 15, center_y), (center_x + 15, center_y), (255, 255, 255), 2)
            cv2.line(cal_display, (center_x, center_y - 15), (center_x, center_y + 15), (255, 255, 255), 2)

            # Progresso
            progress = int((len(samples) / max_samples) * 200)
            cv2.rectangle(cal_display, (center_x - 100, center_y + 50), (center_x + 100, center_y + 70), (50, 50, 50), -1)
            cv2.rectangle(cal_display, (center_x - 100, center_y + 50), (center_x - 100 + progress, center_y + 70), (0, 255, 0), -1)

            cv2.putText(cal_display, f"Amostras: {len(samples)}/{max_samples}", (center_x - 80, center_y + 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow('Calibração 3D - Olhe para o centro', cal_display)

            # Coletar amostras
            if result.get('gaze_vectors_3d') and self.spheres_calibrated:
                samples.append(result['gaze_vectors_3d'])

            key = cv2.waitKey(30) & 0xFF
            if key == 27:  # ESC
                cv2.destroyWindow('Calibração 3D - Olhe para o centro')
                return

        cv2.destroyWindow('Calibração 3D - Olhe para o centro')

        # Calcular média das amostras
        if len(samples) > 0:
            # Usar a última amostra
            gaze_data = samples[-1]

            # Calibrar offset
            success = self.gaze_to_screen_3d.calibrate_offset(gaze_data, center_x, center_y)

            if success:
                self.gaze_to_screen_3d.save_calibration()

    def quick_center_calibration(self, duration=3):
        """
        Calibração rápida de offset central para corrigir deslocamento
        """
        if not self.spheres_calibrated:
            return False
        
        # Posição central da tela
        center_x = self.gaze_to_screen_3d.width_px // 2
        center_y = self.gaze_to_screen_3d.height_px // 2
        
        samples_x = []
        samples_y = []
        start_time = time.time()
        
        # Criar tela de calibração
        cal_display = np.zeros((self.gaze_to_screen_3d.height_px,
                            self.gaze_to_screen_3d.width_px, 3), dtype=np.uint8)
        
        while time.time() - start_time < duration:
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            # Processar frame
            result = self.process_frame(frame)
            
            # Desenhar tela de calibração
            cal_display.fill(0)
            
            # Ponto central
            cv2.circle(cal_display, (center_x, center_y), 20, (0, 255, 0), -1)
            cv2.circle(cal_display, (center_x, center_y), 25, (255, 255, 255), 2)
            
            # Progresso
            progress = ((time.time() - start_time) / duration) * 100
            cv2.putText(cal_display, f"Progresso: {progress:.0f}%", 
                    (center_x - 80, center_y + 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Se temos gaze 3D, usar ele
            if result.get('gaze_vectors_3d') and self.spheres_calibrated:
                screen_point = self.gaze_to_screen_3d.map_gaze_to_screen(result['gaze_vectors_3d'])
                if screen_point:
                    samples_x.append(screen_point[0])
                    samples_y.append(screen_point[1])
                    # Mostrar ponto detectado
                    cv2.circle(cal_display, screen_point, 5, (0, 0, 255), -1)
            
            cv2.imshow('Calibração Rápida - Olhe para o centro', cal_display)
            
            key = cv2.waitKey(30) & 0xFF
            if key == 27:  # ESC
                cv2.destroyWindow('Calibração Rápida - Olhe para o centro')
                return False
        
        cv2.destroyWindow('Calibração Rápida - Olhe para o centro')
        
        if samples_x and samples_y:
            # Calcular média das amostras
            avg_x = np.mean(samples_x)
            avg_y = np.mean(samples_y)
            
            # Calcular offset CORRIGIDO: centro - gaze_detectado
            offset_x = center_x - avg_x
            offset_y = center_y - avg_y
            
            # Aplicar correção
            if hasattr(self.gaze_to_screen_3d, 'calibrate_center_offset'):
                self.gaze_to_screen_3d.calibrate_center_offset(
                    (avg_x, avg_y), 
                    (center_x, center_y)
                )
            else:
                # Se o método não existir, criar manualmente
                self.gaze_to_screen_3d.center_offset_x = offset_x
                self.gaze_to_screen_3d.center_offset_y = offset_y
                self.gaze_to_screen_3d.has_center_calibration = True
            
            # Aplicar ao filtro Kalman também
            if self.use_kalman and hasattr(self.kalman_filter, 'apply_offset_correction'):
                self.kalman_filter.apply_offset_correction(offset_x, offset_y)
            
            # Salvar calibração
            self.gaze_to_screen_3d.save_calibration()

            return True

        return False

    def cleanup(self):
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()



def main():
    """Função principal - Executa o sistema de gaze tracking com logging em Excel"""

    # Importar logger
    from src.utils.data_logger import GazeDataLogger, SessionLogger

    # Criar configuração
    config = GazeTrackingConfig()
    config.debug_enabled = False  # Desabilitar debug para remover prints

    # Criar sistema
    system = GazeTrackingSystem(config)

    # Criar loggers
    data_logger = GazeDataLogger()
    session_logger = SessionLogger()

    # Registrar início da sessão
    session_logger.log_event("INICIO", "Sistema de Gaze Tracking iniciado")

    # Vincular logger ao sistema
    system.data_logger = data_logger
    system.session_logger = session_logger

    # Informações básicas no console (apenas início e fim)
    print("\n" + "="*60)
    print("   SISTEMA DE GAZE TRACKING INICIADO")
    print("="*60)
    print("\n📊 Sistema rodando em modo silencioso")
    print("📝 Dados sendo salvos em Excel automaticamente")
    print(f"📁 Arquivo: {data_logger.excel_file}")
    print("\n⌨️  Controles:")
    print("   'q' - Sair e salvar dados")
    print("   'e' - Calibrar esferas oculares")
    print("   'p' - Calibrar plano do monitor (após 'e')")
    print("   'h' - Calibração rápida de centro")
    print("="*60 + "\n")

    try:
        # Executar sistema
        system.run()

    except KeyboardInterrupt:
        session_logger.log_event("INTERROMPIDO", "Sistema interrompido pelo usuário")
        print("\n\n⚠️  Sistema interrompido pelo usuário")

    except Exception as e:
        session_logger.log_event("ERRO", f"Erro no sistema: {str(e)}")
        print(f"\n\n❌ Erro no sistema: {e}")

    finally:
        # Salvar dados e fechar
        print("\n\n" + "="*60)
        print("   FINALIZANDO E SALVANDO DADOS")
        print("="*60 + "\n")

        session_logger.log_event("FIM", "Sistema finalizado")
        session_logger.save()

        # Salvar e mostrar resumo
        summary = data_logger.close()
        print(summary)

        print(f"\n📄 Log de eventos: {session_logger.session_file}")
        print("\n✅ Sistema finalizado com sucesso!\n")

if __name__ == "__main__":
    main()
