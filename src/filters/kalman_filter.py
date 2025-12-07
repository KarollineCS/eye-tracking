import numpy as np
from typing import Optional, Tuple
import cv2

class GazeKalmanFilter:
    """
    Filtro de Kalman especializado para suavização de dados de gaze.
    Reduz ruído e prevê movimento ocular baseado em modelo físico.
    """
    
    def __init__(self, process_noise: float = 1e-3, measurement_noise: float = 1e-1):
        # Estado: [x, y, vx, vy] - posição e velocidade
        self.state_dim = 4
        self.measurement_dim = 2
        
        # Inicializar filtro OpenCV
        self.kalman = cv2.KalmanFilter(self.state_dim, self.measurement_dim)
        
        # Configurar matrizes do filtro
        self._setup_kalman_matrices(process_noise, measurement_noise)
        
        # Estado inicial
        self.initialized = False
        self.last_prediction = None
        self.confidence = 0.0
        
        # Histórico para análise
        self.prediction_history = []
        self.measurement_history = []
        self.max_history = 100
        
    def _setup_kalman_matrices(self, process_noise: float, measurement_noise: float):
        """Configura as matrizes do filtro de Kalman"""
        
        dt = 1.0 / 30.0  # Assumindo 30 FPS
        
        self.kalman.transitionMatrix = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        # Matriz de medição (observamos apenas posição)
        # [z_x] = [1  0  0  0] [x]
        # [z_y]   [0  1  0  0] [y]
        #                      [vx]
        #                      [vy]
        self.kalman.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)
        
        # Matriz de covariância do ruído do processo
        q = process_noise
        self.kalman.processNoiseCov = np.array([
            [q, 0, 0, 0],
            [0, q, 0, 0],
            [0, 0, q*10, 0],  # Maior ruído na velocidade
            [0, 0, 0, q*10]
        ], dtype=np.float32)
        
        # Matriz de covariância do ruído de medição
        r = measurement_noise
        self.kalman.measurementNoiseCov = np.array([
            [r, 0],
            [0, r]
        ], dtype=np.float32)
        
        # Matriz de covariância inicial do erro
        self.kalman.errorCovPost = np.eye(4, dtype=np.float32) * 1000
        
    def update(self, measurement: Tuple[float, float]) -> Tuple[float, float]:
        """
        Atualiza o filtro com uma nova medição e retorna a estimativa filtrada.
        
        Args:
            measurement: (x, y) coordenadas medidas
            
        Returns:
            (x, y) coordenadas filtradas
        """
        measurement_array = np.array([[measurement[0]], [measurement[1]]], dtype=np.float32)
        
        # Inicializar estado na primeira medição
        if not self.initialized:
            self.kalman.statePre = np.array([
                [measurement[0]],
                [measurement[1]],
                [0],  # Velocidade inicial zero
                [0]
            ], dtype=np.float32)
            self.kalman.statePost = self.kalman.statePre.copy()
            self.initialized = True
        
        # Predição
        prediction = self.kalman.predict()
        
        # Correção com nova medição
        corrected = self.kalman.correct(measurement_array)
        
        # Extrair posição filtrada
        filtered_x = float(corrected[0])
        filtered_y = float(corrected[1])
        
        # Atualizar histórico
        self._update_history(measurement, (filtered_x, filtered_y))
        
        # Calcular confiança baseada na covariância
        self._update_confidence()
        
        self.last_prediction = (filtered_x, filtered_y)
        
        return filtered_x, filtered_y
    
    def update_with_confidence(self, measurement: Tuple[float, float], 
                              external_confidence: float = 1.0) -> Tuple[float, float]:
        """
        Atualiza o filtro ajustando o ruído baseado na confiança externa.
        Use este método quando tiver uma medida de qualidade do tracking.
        
        Args:
            measurement: (x, y) coordenadas medidas
            external_confidence: 0.0 a 1.0, onde 1.0 é máxima confiança
            
        Returns:
            (x, y) coordenadas filtradas
        """
        # Ajustar ruído de medição baseado na confiança
        # Baixa confiança = mais ruído de medição = filtro confia menos na medição
        adjusted_noise = self.kalman.measurementNoiseCov.copy()
        
        if external_confidence < 0.9:  # Se confiança não é perfeita
            # Aumentar ruído inversamente à confiança
            noise_multiplier = 1.0 / max(external_confidence, 0.1)
            adjusted_noise = adjusted_noise * noise_multiplier
            
            # Aplicar temporariamente o ruído ajustado
            original_noise = self.kalman.measurementNoiseCov.copy()
            self.kalman.measurementNoiseCov = adjusted_noise
            
            # Fazer update normal
            result = self.update(measurement)
            
            # Restaurar ruído original
            self.kalman.measurementNoiseCov = original_noise
            
            return result
        else:
            # Alta confiança - usar update normal
            return self.update(measurement)
        
    def apply_offset_correction(self, offset_x: float, offset_y: float):
        """
        Aplica correção de offset ao estado atual do filtro.
        Use após calibração do centro para corrigir deslocamento sistemático.
        
        Args:
            offset_x: Correção em X (pixels)
            offset_y: Correção em Y (pixels)
        """
        if self.initialized:
            # Ajustar o estado atual
            self.kalman.statePost[0] -= offset_x
            self.kalman.statePost[1] -= offset_y
            
            # Ajustar também o estado previsto
            self.kalman.statePre[0] -= offset_x
            self.kalman.statePre[1] -= offset_y
    
    def predict(self) -> Optional[Tuple[float, float]]:
        """
        Faz uma predição sem nova medição (útil para frames perdidos).
        
        Returns:
            (x, y) coordenadas preditas ou None se não inicializado
        """
        if not self.initialized:
            return None
        
        prediction = self.kalman.predict()
        return float(prediction[0]), float(prediction[1])
    
    def get_velocity(self) -> Optional[Tuple[float, float]]:
        """
        Retorna a velocidade estimada do movimento do olho.
        
        Returns:
            (vx, vy) velocidade em pixels/frame ou None
        """
        if not self.initialized:
            return None
        
        state = self.kalman.statePost
        return float(state[2]), float(state[3])
    
    def _update_history(self, measurement: Tuple[float, float], 
                       filtered: Tuple[float, float]):
        """Atualiza histórico de medições e predições"""
        self.measurement_history.append(measurement)
        self.prediction_history.append(filtered)
        
        # Limitar tamanho do histórico
        if len(self.measurement_history) > self.max_history:
            self.measurement_history.pop(0)
        if len(self.prediction_history) > self.max_history:
            self.prediction_history.pop(0)
    
    def _update_confidence(self):
        """Calcula confiança baseada na covariância do erro"""
        # Usar o traço da matriz de covariância como medida de incerteza
        error_cov = self.kalman.errorCovPost
        uncertainty = np.trace(error_cov[:2, :2])  # Apenas posição
        
        # Converter para confiança (0-1)
        # Menor incerteza = maior confiança
        self.confidence = 1.0 / (1.0 + uncertainty * 0.01)
    
    def get_smoothness_metrics(self) -> dict:
        """
        Calcula métricas de suavização do filtro.
        
        Returns:
            Dicionário com métricas de desempenho
        """
        if len(self.measurement_history) < 2:
            return {
                'smoothness_improvement': 0.0,
                'noise_reduction': 0.0,
                'lag': 0.0
            }
        
        # Calcular ruído nas medições originais
        measurement_diffs = np.diff(self.measurement_history, axis=0)
        measurement_noise = np.std(measurement_diffs, axis=0)
        
        # Calcular ruído nas predições filtradas
        prediction_diffs = np.diff(self.prediction_history, axis=0)
        prediction_noise = np.std(prediction_diffs, axis=0)
        
        # Métrica de redução de ruído
        noise_reduction = 1.0 - (np.mean(prediction_noise) / 
                                (np.mean(measurement_noise) + 1e-6))
        
        # Métrica de lag (atraso entre medição e predição)
        if len(self.measurement_history) > 10:
            recent_measurements = self.measurement_history[-10:]
            recent_predictions = self.prediction_history[-10:]
            
            lag = np.mean([
                np.linalg.norm(np.array(m) - np.array(p))
                for m, p in zip(recent_measurements, recent_predictions)
            ])
        else:
            lag = 0.0
        
        return {
            'smoothness_improvement': noise_reduction * 100,
            'noise_reduction_x': float(1.0 - prediction_noise[0] / (measurement_noise[0] + 1e-6)),
            'noise_reduction_y': float(1.0 - prediction_noise[1] / (measurement_noise[1] + 1e-6)),
            'lag_pixels': float(lag),
            'confidence': self.confidence
        }
    
    def reset(self):
        """Reinicia o filtro de Kalman"""
        self.initialized = False
        self.last_prediction = None
        self.confidence = 0.0
        self.prediction_history.clear()
        self.measurement_history.clear()
        
        # Reinicializar matrizes
        self.kalman.errorCovPost = np.eye(4, dtype=np.float32) * 1000


class AdaptiveKalmanFilter(GazeKalmanFilter):
    """
    Filtro de Kalman adaptativo que ajusta parâmetros baseado no comportamento observado.
    """
    
    def __init__(self):
        super().__init__()
        self.saccade_detector = SaccadeDetector()
        self.fixation_detector = FixationDetector()
        
        # Parâmetros adaptativos
        self.base_process_noise = 0.01
        self.base_measurement_noise = 0.1
        
    def update(self, measurement: Tuple[float, float], 
               iris_confidence: float = 1.0) -> Tuple[float, float]:
        """
        Atualiza com adaptação baseada no tipo de movimento E qualidade da íris.
        
        Args:
            measurement: (x, y) coordenadas medidas
            iris_confidence: Confiança da detecção da íris (0.0 a 1.0)
        """
        # Detectar tipo de movimento
        is_saccade = self.saccade_detector.detect(measurement)
        is_fixation = self.fixation_detector.detect(measurement)
        
        # NOVO: Combinar tipo de movimento com confiança da íris
        if iris_confidence < 0.5:
            # Baixa confiança na íris - aumentar filtragem
            process_noise = self.base_process_noise * 0.5
            measurement_noise = self.base_measurement_noise * 3
        elif is_saccade:
            # Durante sacadas COM boa detecção
            process_noise = self.base_process_noise * 10
            measurement_noise = self.base_measurement_noise * 0.5
        elif is_fixation:
            # Durante fixações COM boa detecção
            process_noise = self.base_process_noise * 0.1
            measurement_noise = self.base_measurement_noise * 2
        else:
            # Movimento normal
            process_noise = self.base_process_noise
            measurement_noise = self.base_measurement_noise
        
        # Atualizar matrizes do filtro
        self._setup_kalman_matrices(process_noise, measurement_noise)
        
        # Chamar update da classe pai
        return super().update(measurement)


class SaccadeDetector:
    """Detecta movimentos rápidos dos olhos (sacadas)"""
    
    def __init__(self, threshold: float = 50.0, window_size: int = 3):
        self.threshold = threshold
        self.window_size = window_size
        self.position_history = []
        
    def detect(self, position: Tuple[float, float]) -> bool:
        """Detecta se o movimento atual é uma sacada"""
        self.position_history.append(position)
        
        # Manter janela de tamanho fixo
        if len(self.position_history) > self.window_size:
            self.position_history.pop(0)
        
        if len(self.position_history) < 2:
            return False
        
        # Calcular velocidade
        velocities = []
        for i in range(1, len(self.position_history)):
            prev = self.position_history[i-1]
            curr = self.position_history[i]
            velocity = np.linalg.norm(np.array(curr) - np.array(prev))
            velocities.append(velocity)
        
        # Sacada detectada se velocidade exceder threshold
        return np.mean(velocities) > self.threshold


class FixationDetector:
    """Detecta quando o olho está fixado em um ponto"""
    
    def __init__(self, threshold: float = 5.0, min_duration: int = 10):
        self.threshold = threshold
        self.min_duration = min_duration
        self.position_history = []
        self.fixation_start = None
        
    def detect(self, position: Tuple[float, float]) -> bool:
        """Detecta se o olho está em fixação"""
        self.position_history.append(position)
        
        # Manter histórico limitado
        if len(self.position_history) > self.min_duration * 2:
            self.position_history.pop(0)
        
        if len(self.position_history) < self.min_duration:
            return False
        
        # Verificar estabilidade nas últimas amostras
        recent_positions = self.position_history[-self.min_duration:]
        center = np.mean(recent_positions, axis=0)
        
        # Calcular dispersão
        dispersions = [np.linalg.norm(np.array(p) - center) 
                      for p in recent_positions]
        
        # Fixação detectada se dispersão for baixa
        return np.mean(dispersions) < self.threshold
