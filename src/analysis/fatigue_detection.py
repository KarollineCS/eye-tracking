# src/analysis/fatigue_detection.py
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import deque
import time
from datetime import datetime, timedelta
from enum import Enum

class FatigueLevel(Enum):
    """Níveis de fadiga detectados"""
    ALERT = "alert"           # Alerta e atento
    MILD_FATIGUE = "mild"     # Fadiga leve
    MODERATE_FATIGUE = "moderate"  # Fadiga moderada
    SEVERE_FATIGUE = "severe"      # Fadiga severa
    CRITICAL = "critical"           # Estado crítico - risco alto

class AdvancedFatigueDetector:
    """
    Sistema avançado de detecção de fadiga que combina múltiplos indicadores:
    - PERCLOS (Percentage of Eye Closure)
    - Frequência de piscadas
    - Duração das piscadas
    - Padrões de fixação
    - Velocidade de sacadas
    - Variabilidade do diâmetro pupilar
    """
    
    def __init__(self, fps: int = 30):
        self.fps = fps
        
        # Buffers para análise temporal
        self.blink_history = deque(maxlen=300)  # 10 segundos a 30fps
        self.eye_openness_history = deque(maxlen=900)  # 30 segundos
        self.pupil_size_history = deque(maxlen=150)  # 5 segundos
        self.fixation_history = deque(maxlen=600)  # 20 segundos
        self.saccade_velocity_history = deque(maxlen=100)
        
        # Métricas de fadiga
        self.perclos_score = 0.0
        self.blink_rate = 0.0
        self.microsleep_count = 0
        self.long_blink_count = 0
        
        # Histórico de estados
        self.fatigue_level_history = deque(maxlen=100)
        self.last_alert_time = time.time()
        
        # Parâmetros adaptativos
        self.baseline_metrics = None
        self.calibration_period = 60  # segundos
        self.calibration_start = time.time()
        
        # Thresholds
        self.perclos_threshold = 0.15  # 15% olhos fechados = fadiga
        self.long_blink_threshold = 0.5  # 500ms = piscada longa
        self.microsleep_threshold = 1.0  # 1 segundo = microsono
        
    def update(self, eye_metrics: Dict) -> Dict:
        """
        Atualiza o detector com novas métricas dos olhos.
        
        Args:
            eye_metrics: Dicionário com métricas atuais dos olhos
                - left_eye_aspect_ratio: EAR do olho esquerdo
                - right_eye_aspect_ratio: EAR do olho direito
                - pupil_size: Tamanho da pupila
                - gaze_velocity: Velocidade do olhar
                - is_blinking: Se está piscando
                
        Returns:
            Dicionário com análise de fadiga
        """
        # Calcular abertura média dos olhos
        left_ear = eye_metrics.get('left_eye_aspect_ratio', 0.3)
        right_ear = eye_metrics.get('right_eye_aspect_ratio', 0.3)
        eye_openness = (left_ear + right_ear) / 2
        
        # Atualizar históricos
        self.eye_openness_history.append(eye_openness)
        self.blink_history.append(eye_metrics.get('is_blinking', False))
        
        if 'pupil_size' in eye_metrics:
            self.pupil_size_history.append(eye_metrics['pupil_size'])
        
        if 'gaze_velocity' in eye_metrics:
            self.saccade_velocity_history.append(eye_metrics['gaze_velocity'])
        
        # Calcular métricas de fadiga
        perclos = self._calculate_perclos()
        blink_metrics = self._analyze_blinks()
        pupil_metrics = self._analyze_pupil_dynamics()
        saccade_metrics = self._analyze_saccades()
        
        # Determinar nível de fadiga
        fatigue_level = self._determine_fatigue_level(
            perclos, blink_metrics, pupil_metrics, saccade_metrics
        )
        
        # Atualizar histórico
        self.fatigue_level_history.append(fatigue_level)
        
        # Gerar alertas se necessário
        alerts = self._generate_alerts(fatigue_level)
        
        return {
            'fatigue_level': fatigue_level.value,
            'perclos': perclos,
            'blink_rate': blink_metrics['rate'],
            'long_blinks': blink_metrics['long_blinks'],
            'microsleeps': blink_metrics['microsleeps'],
            'pupil_stability': pupil_metrics['stability'],
            'saccade_velocity_decline': saccade_metrics['velocity_decline'],
            'alerts': alerts,
            'confidence': self._calculate_confidence(),
            'time_since_last_alert': time.time() - self.last_alert_time
        }
    
    def _calculate_perclos(self) -> float:
        """
        Calcula PERCLOS (Percentage of Eye Closure over Time).
        Métrica padrão: P80 (olhos 80% fechados contam como fechados).
        """
        if len(self.eye_openness_history) < 30:
            return 0.0
        
        # Threshold para considerar olho fechado (20% de abertura)
        closure_threshold = 0.2
        
        # Calcular percentual de frames com olhos fechados
        closed_frames = sum(1 for openness in self.eye_openness_history 
                          if openness < closure_threshold)
        
        perclos = closed_frames / len(self.eye_openness_history)
        self.perclos_score = perclos
        
        return perclos
    
    def _analyze_blinks(self) -> Dict:
        """Analisa padrões de piscadas"""
        if len(self.blink_history) < self.fps:
            return {
                'rate': 0.0,
                'long_blinks': 0,
                'microsleeps': 0,
                'avg_duration': 0.0
            }
        
        # Detectar sequências de piscadas
        blink_sequences = []
        current_blink = []
        
        for i, is_blinking in enumerate(self.blink_history):
            if is_blinking:
                current_blink.append(i)
            elif current_blink:
                blink_sequences.append(current_blink)
                current_blink = []
        
        if current_blink:
            blink_sequences.append(current_blink)
        
        # Calcular métricas
        total_blinks = len(blink_sequences)
        long_blinks = 0
        microsleeps = 0
        durations = []
        
        for seq in blink_sequences:
            duration = len(seq) / self.fps
            durations.append(duration)
            
            if duration > self.microsleep_threshold:
                microsleeps += 1
            elif duration > self.long_blink_threshold:
                long_blinks += 1
        
        # Taxa de piscadas por minuto
        time_window = len(self.blink_history) / self.fps
        blink_rate = (total_blinks / time_window) * 60 if time_window > 0 else 0
        
        self.blink_rate = blink_rate
        self.long_blink_count = long_blinks
        self.microsleep_count = microsleeps
        
        return {
            'rate': blink_rate,
            'long_blinks': long_blinks,
            'microsleeps': microsleeps,
            'avg_duration': np.mean(durations) if durations else 0.0
        }
    
    def _analyze_pupil_dynamics(self) -> Dict:
        """Analisa variações no tamanho da pupila"""
        if len(self.pupil_size_history) < 10:
            return {
                'stability': 1.0,
                'fluctuation': 0.0,
                'trend': 0.0
            }
        
        pupil_sizes = list(self.pupil_size_history)
        
        # Calcular estabilidade (fadiga causa instabilidade)
        std_dev = np.std(pupil_sizes)
        mean_size = np.mean(pupil_sizes)
        cv = std_dev / mean_size if mean_size > 0 else 0  # Coeficiente de variação
        
        # Estabilidade inversa ao CV
        stability = 1.0 / (1.0 + cv * 10)
        
        # Tendência (pupilas contraem com fadiga)
        if len(pupil_sizes) > 30:
            recent = np.mean(pupil_sizes[-10:])
            earlier = np.mean(pupil_sizes[:10])
            trend = (recent - earlier) / earlier if earlier > 0 else 0
        else:
            trend = 0.0
        
        return {
            'stability': stability,
            'fluctuation': cv,
            'trend': trend
        }
    
    def _analyze_saccades(self) -> Dict:
        """Analisa velocidade e padrões de sacadas"""
        if len(self.saccade_velocity_history) < 10:
            return {
                'velocity_decline': 0.0,
                'irregularity': 0.0
            }
        
        velocities = list(self.saccade_velocity_history)
        
        # Detectar declínio na velocidade (indicador de fadiga)
        if len(velocities) > 30:
            recent_avg = np.mean(velocities[-10:])
            earlier_avg = np.mean(velocities[:10])
            
            if earlier_avg > 0:
                velocity_decline = (earlier_avg - recent_avg) / earlier_avg
            else:
                velocity_decline = 0.0
        else:
            velocity_decline = 0.0
        
        # Calcular irregularidade
        if len(velocities) > 2:
            diffs = np.diff(velocities)
            irregularity = np.std(diffs) / (np.mean(np.abs(velocities)) + 1e-6)
        else:
            irregularity = 0.0
        
        return {
            'velocity_decline': max(0, velocity_decline),
            'irregularity': irregularity
        }
    
    def _determine_fatigue_level(self, perclos: float, blink_metrics: Dict,
                                 pupil_metrics: Dict, saccade_metrics: Dict) -> FatigueLevel:
        """
        Determina o nível de fadiga baseado em múltiplas métricas.
        Usa um sistema de pontuação ponderada.
        """
        score = 0.0
        
        # PERCLOS (peso: 35%)
        if perclos > 0.25:
            score += 35
        elif perclos > 0.20:
            score += 28
        elif perclos > 0.15:
            score += 21
        elif perclos > 0.10:
            score += 14
        elif perclos > 0.05:
            score += 7
        
        # Taxa de piscadas (peso: 20%)
        # Normal: 15-20 piscadas/min
        if blink_metrics['rate'] < 10 or blink_metrics['rate'] > 30:
            score += 20
        elif blink_metrics['rate'] < 12 or blink_metrics['rate'] > 25:
            score += 15
        elif blink_metrics['rate'] < 14 or blink_metrics['rate'] > 22:
            score += 10
        else:
            score += 5
        
        # Piscadas longas e microssonos (peso: 25%)
        if blink_metrics['microsleeps'] > 0:
            score += 25
        elif blink_metrics['long_blinks'] > 3:
            score += 20
        elif blink_metrics['long_blinks'] > 1:
            score += 15
        elif blink_metrics['long_blinks'] > 0:
            score += 10
        
        # Estabilidade pupilar (peso: 10%)
        if pupil_metrics['stability'] < 0.3:
            score += 10
        elif pupil_metrics['stability'] < 0.5:
            score += 7
        elif pupil_metrics['stability'] < 0.7:
            score += 4
        
        # Declínio de velocidade de sacadas (peso: 10%)
        if saccade_metrics['velocity_decline'] > 0.3:
            score += 10
        elif saccade_metrics['velocity_decline'] > 0.2:
            score += 7
        elif saccade_metrics['velocity_decline'] > 0.1:
            score += 4
        
        # Determinar nível baseado no score
        if score >= 80:
            return FatigueLevel.CRITICAL
        elif score >= 60:
            return FatigueLevel.SEVERE_FATIGUE
        elif score >= 40:
            return FatigueLevel.MODERATE_FATIGUE
        elif score >= 20:
            return FatigueLevel.MILD_FATIGUE
        else:
            return FatigueLevel.ALERT
    
    def _generate_alerts(self, fatigue_level: FatigueLevel) -> List[str]:
        """Gera alertas baseados no nível de fadiga"""
        alerts = []
        current_time = time.time()
        
        # Alertas específicos por nível
        if fatigue_level == FatigueLevel.CRITICAL:
            alerts.append("⚠️ ALERTA CRÍTICO: Fadiga extrema detectada! Pare imediatamente!")
            self.last_alert_time = current_time
            
        elif fatigue_level == FatigueLevel.SEVERE_FATIGUE:
            if current_time - self.last_alert_time > 30:  # Alerta a cada 30s
                alerts.append("⚠️ Fadiga severa detectada. Considere parar para descansar.")
                self.last_alert_time = current_time
                
        elif fatigue_level == FatigueLevel.MODERATE_FATIGUE:
            if current_time - self.last_alert_time > 60:  # Alerta a cada minuto
                alerts.append("⚠️ Fadiga moderada. Faça uma pausa em breve.")
                self.last_alert_time = current_time
        
        # Alertas específicos por evento
        if self.microsleep_count > 0:
            alerts.append(f"💤 {self.microsleep_count} microsono(s) detectado(s)!")
        
        if self.perclos_score > 0.20:
            alerts.append(f"👁️ Olhos fechados {self.perclos_score*100:.1f}% do tempo")
        
        return alerts
    
    def _calculate_confidence(self) -> float:
        """Calcula confiança na detecção"""
        # Confiança baseada na quantidade de dados
        data_completeness = min(
            len(self.eye_openness_history) / 900,  # 30s de dados
            len(self.blink_history) / 300,  # 10s de dados
            1.0
        )
        
        # Confiança baseada na consistência
        if len(self.fatigue_level_history) > 10:
            recent_levels = list(self.fatigue_level_history)[-10:]
            consistency = 1.0 - (len(set(recent_levels)) - 1) / 4  # Máximo 5 níveis
        else:
            consistency = 0.5
        
        return (data_completeness * 0.6 + consistency * 0.4)
    
    def get_detailed_report(self) -> Dict:
        """Gera relatório detalhado de fadiga"""
        if len(self.fatigue_level_history) == 0:
            return {'status': 'no_data'}
        
        # Contar ocorrências de cada nível
        level_counts = {}
        for level in FatigueLevel:
            level_counts[level.value] = sum(1 for l in self.fatigue_level_history 
                                           if l == level)
        
        # Calcular tempo em cada estado
        total_samples = len(self.fatigue_level_history)
        level_percentages = {
            level: (count / total_samples * 100) if total_samples > 0 else 0
            for level, count in level_counts.items()
        }
        
        return {
            'status': 'active',
            'current_level': self.fatigue_level_history[-1].value if self.fatigue_level_history else 'unknown',
            'perclos_avg': self.perclos_score,
            'blink_rate_avg': self.blink_rate,
            'microsleep_total': self.microsleep_count,
            'long_blink_total': self.long_blink_count,
            'level_distribution': level_percentages,
            'monitoring_duration': len(self.eye_openness_history) / self.fps,
            'confidence': self._calculate_confidence()
        }
    
    def reset_baseline(self):
        """Reseta a baseline para recalibração"""
        self.baseline_metrics = None
        self.calibration_start = time.time()
        
    def clear_history(self):
        """Limpa todo o histórico"""
        self.blink_history.clear()
        self.eye_openness_history.clear()
        self.pupil_size_history.clear()
        self.fixation_history.clear()
        self.saccade_velocity_history.clear()
        self.fatigue_level_history.clear()
        self.perclos_score = 0.0
        self.blink_rate = 0.0
        self.microsleep_count = 0
        self.long_blink_count = 0