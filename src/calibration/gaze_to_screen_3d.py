import numpy as np
import yaml
import os
from typing import Optional, Tuple, Dict
import time
import math
import pyautogui

MONITOR_WIDTH, MONITOR_HEIGHT = pyautogui.size()

def convert_gaze_to_screen_coordinates(combined_gaze_direction, calibration_offset_yaw, calibration_offset_pitch):
    """
    Converte o vetor de direção 3D em coordenadas 2D da tela
    """
    avg_direction = np.array(combined_gaze_direction, dtype=float)
    
    # Normalizar
    norm = np.linalg.norm(avg_direction)
    if norm > 1e-9:
        avg_direction = avg_direction / norm
    else:
        return MONITOR_WIDTH//2, MONITOR_HEIGHT//2, 0, 0
    
    yaw_rad = np.arctan2(-avg_direction[0], avg_direction[2])
    
    # Pitch: ângulo vertical (cima/baixo)
    pitch_rad = np.arcsin(avg_direction[1])
    
    # Converter para graus
    yaw_deg = np.degrees(yaw_rad)
    pitch_deg = np.degrees(pitch_rad)
    
    # Parâmetros de sensibilidade
    yawDegrees = 8.0    # Campo visual horizontal
    pitchDegrees = 3.0  # Campo visual vertical
    
    # Aplicar offsets de calibração
    yaw_deg += calibration_offset_yaw
    pitch_deg += calibration_offset_pitch
    
    # Mapear para a tela
    # Normalizar para 0-1
    x_normalized = (yaw_deg + yawDegrees) / (2 * yawDegrees)
    y_normalized = (pitch_deg + pitchDegrees) / (2 * pitchDegrees)
    
    # Converter para pixels
    screen_x = int(x_normalized * MONITOR_WIDTH)
    screen_y = int(y_normalized * MONITOR_HEIGHT)
    
    # Limitar aos bounds
    screen_x = max(0, min(screen_x, MONITOR_WIDTH - 1))
    screen_y = max(0, min(screen_y, MONITOR_HEIGHT - 1))
    
    return screen_x, screen_y, yaw_deg - calibration_offset_yaw, pitch_deg - calibration_offset_pitch

class GazeToScreen3D:
    """
    Sistema de mapeamento Gaze-to-Screen usando geometria 3D
    """

    def __init__(self, screen_width_px: int, screen_height_px: int,
                 screen_width_mm: float = 344.0, screen_height_mm: float = 194.0,
                 flip_x: bool = False):
        # Dimensões da tela
        self.width_px = screen_width_px
        self.height_px = screen_height_px
        self.width_mm = screen_width_mm
        self.height_mm = screen_height_mm
        self.flip_x = flip_x

        # Offsets de calibração
        self.calibration_offset_yaw = 0.0
        self.calibration_offset_pitch = 0.0
        self.is_calibrated_flag = False

        # Histórico
        self.last_valid_point = None

        # Clamping (limitar gaze às bordas do monitor)
        self.clamp_to_screen = True 

    def calibrate_center_offset(self, current_screen_point, true_center_point):
        """
        Calibra o offset quando o usuário olha para o centro da tela
        """
        # Se gaze está em 41 e deveria estar em 683, offset = 683-41 = +642
        offset_x = true_center_point[0] - current_screen_point[0]
        offset_y = true_center_point[1] - current_screen_point[1]
        
        # Armazenar offsets para correção futura
        self.center_offset_x = offset_x
        self.center_offset_y = offset_y
        self.has_center_calibration = True

        # Salvar no arquivo de calibração
        self.save_calibration()
        
        return True

    def map_gaze_to_screen(self, gaze_data_3d: Dict) -> Optional[Tuple[int, int]]:
        """
        Mapeia gaze 3D para coordenadas da tela COM CORREÇÃO DE OFFSET
        """
        
        # Extrair dados do gaze
        if 'origin' in gaze_data_3d and 'direction' in gaze_data_3d:
            direction = np.array(gaze_data_3d['direction'])
        elif 'left' in gaze_data_3d and 'right' in gaze_data_3d:
            directions = []
            for eye in ['left', 'right']:
                if eye in gaze_data_3d and 'direction' in gaze_data_3d[eye]:
                    directions.append(np.array(gaze_data_3d[eye]['direction']))
            
            if len(directions) == 0:
                return self.last_valid_point
            
            direction = np.mean(directions, axis=0)
            direction = direction / np.linalg.norm(direction)
        else:
            return self.last_valid_point

        screen_x, screen_y, _, _ = convert_gaze_to_screen_coordinates(
            direction,
            self.calibration_offset_yaw,
            self.calibration_offset_pitch
        )
        # Aplicar offsets de calibração angular
        if 'yaw' in locals():
            yaw += self.calibration_offset_yaw
            pitch += self.calibration_offset_pitch
        
        # Calcular coordenadas base
        # Valores menores = maior sensibilidade
        yaw_degrees = 8.0    # Campo visual horizontal
        pitch_degrees = 3.0  # Campo visual vertical

        if 'yaw' in locals():
            yaw_deg = np.degrees(yaw)
            pitch_deg = np.degrees(pitch)
        else:
            yaw_deg = np.degrees(np.arctan2(direction[0], -direction[2]))
            pitch_deg = np.degrees(np.arcsin(-direction[1]))  # Inverter apenas Y

        # Mapear para tela
        screen_x = int(((yaw_deg + yaw_degrees) / (2 * yaw_degrees)) * self.width_px)
        screen_y = int(((pitch_deg + pitch_degrees) / (2 * pitch_degrees)) * self.height_px)

        # Aplicar correção de offset do centro
        if hasattr(self, 'has_center_calibration') and self.has_center_calibration:
            screen_x += int(self.center_offset_x) 
            screen_y += int(self.center_offset_y)
        
        # Limitar aos bounds da tela
        screen_x = max(0, min(screen_x, self.width_px - 1))
        screen_y = max(0, min(screen_y, self.height_px - 1))
        
        # Aplicar flip se configurado
        if self.flip_x:
            screen_x = self.width_px - 1 - screen_x
        
        self.last_valid_point = (screen_x, screen_y)
        return (screen_x, screen_y)

    def calibrate_offset(self, gaze_data_3d: Dict, target_x: int, target_y: int):
        # Extrair origin e direction
        if 'left' in gaze_data_3d or 'right' in gaze_data_3d:
            directions = []
            for eye in ['left', 'right']:
                if eye in gaze_data_3d and 'direction' in gaze_data_3d[eye]:
                    directions.append(np.array(gaze_data_3d[eye]['direction']))
            if len(directions) == 0: return False
            direction = np.mean(directions, axis=0)
        else:
            direction = np.array(gaze_data_3d['direction'])
        
        if np.linalg.norm(direction) < 1e-6: return False

        # Calcular ângulos brutos
        _, _, raw_yaw, raw_pitch = convert_gaze_to_screen_coordinates(
            direction, 0.0, 0.0
        )
        
        # Definir o offset como o oposto do ângulo bruto
        self.calibration_offset_yaw = 0 - raw_yaw
        self.calibration_offset_pitch = 0 - raw_pitch
        self.is_calibrated_flag = True
        return True

    def load_calibration(self, filepath: str = 'calibration/gaze_3d_angular.yaml') -> bool:
        """Carrega calibração incluindo offset do centro"""
        try:
            if not os.path.exists(filepath):
                return False

            with open(filepath, 'r') as f:
                data = yaml.safe_load(f)

            self.calibration_offset_yaw = data.get('offset_yaw', 0.0)
            self.calibration_offset_pitch = data.get('offset_pitch', 0.0)
            self.is_calibrated_flag = data.get('is_calibrated', False)
            
            # Carregar offsets do centro
            self.center_offset_x = data.get('center_offset_x', 0.0)
            self.center_offset_y = data.get('center_offset_y', 0.0)
            self.has_center_calibration = data.get('has_center_calibration', False)

            return True

        except Exception as e:
            print(f"⚠️ Erro ao carregar calibração: {e}")
            return False

    def save_calibration(self, filepath: str = 'calibration/gaze_3d_angular.yaml'):
        """Salva calibração incluindo offset do centro"""
        data = {
            'offset_yaw': float(self.calibration_offset_yaw),
            'offset_pitch': float(self.calibration_offset_pitch),
            'is_calibrated': self.is_calibrated_flag,
            'screen_width_px': int(self.width_px),
            'screen_height_px': int(self.height_px),
            'center_offset_x': float(getattr(self, 'center_offset_x', 0)),
            'center_offset_y': float(getattr(self, 'center_offset_y', 0)),
            'has_center_calibration': getattr(self, 'has_center_calibration', False),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }

        os.makedirs('calibration', exist_ok=True)
        with open(filepath, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)

    def is_calibrated(self) -> bool:
        """Verifica se tem calibração de offset"""
        return self.is_calibrated_flag

    def reset_calibration(self):
        self.calibration_offset_yaw = 0.0
        self.calibration_offset_pitch = 0.0
        self.is_calibrated_flag = False

class GazeToScreen3DRayPlane:
    """
    Sistema de mapeamento Gaze-to-Screen usando RAY-PLANE INTERSECTION
    Baseado no Webcam3DTracker - resolve problema de amplitude vertical
    """

    def __init__(self, screen_width_px: int, screen_height_px: int,
                 screen_width_mm: float = 344.0, screen_height_mm: float = 194.0):
        self.width_px = screen_width_px
        self.height_px = screen_height_px
        self.width_mm = screen_width_mm
        self.height_mm = screen_height_mm

        # Plano da tela
        self.monitor_corners = None
        self.monitor_center = None
        self.monitor_normal = None

        # Calibração de offset
        self.center_offset_x = 0.0
        self.center_offset_y = 0.0
        self.has_center_calibration = False

        # Histórico
        self.last_valid_point = None

        self.clamp_to_screen = True 

    def set_monitor_plane(self, corners, center, normal):
        """
        Define o plano do monitor no espaço 3D
        """
        self.monitor_corners = corners
        self.monitor_center = center
        self.monitor_normal = normal

    def ray_plane_intersection(self, ray_origin, ray_direction):
        """
        Calcula interseção do raio de gaze com o plano da tela

        Returns:
            (a, b) coordenadas normalizadas (0-1) no plano, ou None se sem interseção
        """
        if self.monitor_corners is None:
            return None

        # Normalizar direção
        D = ray_direction / np.linalg.norm(ray_direction)
        O = ray_origin
        C = self.monitor_center
        N = self.monitor_normal

        # Calcular interseção raio-plano
        denom = float(np.dot(N, D))
        if abs(denom) < 1e-6:
            return None  # Raio paralelo ao plano

        t = float(np.dot(N, (C - O)) / denom)
        if t < 0:
            return None  # Interseção atrás do olho

        # Ponto de interseção 3D
        P = O + t * D

        # Converter para coordenadas do plano (0-1)
        p0, p1, p2, p3 = self.monitor_corners
        u = p1 - p0  # Vetor largura
        v = p3 - p0  # Vetor altura

        u_len2 = float(np.dot(u, u))
        v_len2 = float(np.dot(v, v))

        if u_len2 < 1e-9 or v_len2 < 1e-9:
            return None

        wv = P - p0
        a = float(np.dot(wv, u) / u_len2)
        b = float(np.dot(wv, v) / v_len2)

        return (a, b)

    def map_gaze_to_screen(self, gaze_data_3d: Dict) -> Optional[Tuple[int, int]]:
        """
        Mapeia gaze 3D para coordenadas da tela usando ray-plane intersection
        """
        if self.monitor_corners is None:
            return self.last_valid_point

        # Extrair origem e direção do gaze
        if 'left' in gaze_data_3d and 'right' in gaze_data_3d:
            # Combinar ambos os olhos
            left_data = gaze_data_3d['left']
            right_data = gaze_data_3d['right']

            if 'origin' not in left_data or 'origin' not in right_data:
                return self.last_valid_point
            if 'direction' not in left_data or 'direction' not in right_data:
                return self.last_valid_point

            # Origem = ponto médio entre os olhos
            ray_origin = (np.array(left_data['origin']) + np.array(right_data['origin'])) / 2.0

            # Direção = média das direções
            ray_direction = (np.array(left_data['direction']) + np.array(right_data['direction'])) / 2.0
        else:
            return self.last_valid_point

        # Calcular interseção
        result = self.ray_plane_intersection(ray_origin, ray_direction)

        if result is None:
            return self.last_valid_point

        a, b = result

        # NOVO: Clamping - limitar coordenadas aos limites do monitor
        if self.clamp_to_screen:
            # Clampar a e b para [0, 1]
            a = max(0.0, min(1.0, a))
            b = max(0.0, min(1.0, b))
        else:
            # Comportamento antigo: rejeitar se fora dos limites
            if not (0.0 <= a <= 1.0 and 0.0 <= b <= 1.0):
                return self.last_valid_point

        # Converter para pixels
        screen_x = int(a * self.width_px)
        screen_y = int(b * self.height_px)

        # Aplicar offset de calibração (se houver)
        if self.has_center_calibration:
            screen_x -= int(self.center_offset_x)
            screen_y -= int(self.center_offset_y)

        # Limitar aos bounds
        screen_x = max(0, min(screen_x, self.width_px - 1))
        screen_y = max(0, min(screen_y, self.height_px - 1))

        self.last_valid_point = (screen_x, screen_y)
        return (screen_x, screen_y)

    def calibrate_center_offset(self, current_screen_point, true_center_point):
        """Calibra offset do centro"""
        offset_x = current_screen_point[0] - true_center_point[0]
        offset_y = current_screen_point[1] - true_center_point[1]

        self.center_offset_x = offset_x
        self.center_offset_y = offset_y
        self.has_center_calibration = True

        return True

    def is_calibrated(self) -> bool:
        return self.monitor_corners is not None

    def reset_calibration(self):
        self.center_offset_x = 0.0
        self.center_offset_y = 0.0
        self.has_center_calibration = False

    def load_calibration(self, filepath: str = 'calibration/gaze_3d_rayplane.yaml') -> bool:
        """Carrega calibração de ray-plane"""
        try:
            if not os.path.exists(filepath):
                return False

            with open(filepath, 'r') as f:
                data = yaml.safe_load(f)

            # Carregar geometria do plano
            if 'monitor_corners' in data:
                self.monitor_corners = [np.array(c) for c in data['monitor_corners']]
                self.monitor_center = np.array(data['monitor_center'])
                self.monitor_normal = np.array(data['monitor_normal'])

            # Carregar offsets do centro
            self.center_offset_x = data.get('center_offset_x', 0.0)
            self.center_offset_y = data.get('center_offset_y', 0.0)
            self.has_center_calibration = data.get('has_center_calibration', False)

            return True

        except Exception:
            return False

    def save_calibration(self, filepath: str = 'calibration/gaze_3d_rayplane.yaml'):
        """Salva calibração de ray-plane"""
        data = {
            'screen_width_px': int(self.width_px),
            'screen_height_px': int(self.height_px),
            'center_offset_x': float(self.center_offset_x),
            'center_offset_y': float(self.center_offset_y),
            'has_center_calibration': self.has_center_calibration,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }

        # Salvar geometria do plano se disponível
        if self.monitor_corners is not None:
            data['monitor_corners'] = [c.tolist() for c in self.monitor_corners]
            data['monitor_center'] = self.monitor_center.tolist()
            data['monitor_normal'] = self.monitor_normal.tolist()

        os.makedirs('calibration', exist_ok=True)
        with open(filepath, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
