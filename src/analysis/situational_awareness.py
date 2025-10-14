import numpy as np
import time
import math
from collections import deque, defaultdict
from typing import Dict, List, Tuple, Optional

class SituationalAwarenessAnalyzer:
    """Analisador de Consciência Situacional baseado em eye-tracking"""
    
    def __init__(self, config):
        self.config = config
        self.attention_zones = config.situational_awareness.attention_zones
        self.fixation_threshold = config.situational_awareness.fixation_threshold
        self.saccade_threshold = config.situational_awareness.saccade_velocity_threshold
        self.fatigue_threshold = config.situational_awareness.fatigue_blink_rate_threshold
        self.attention_switch_threshold = config.situational_awareness.attention_switch_threshold
        
        # Históricos para análise
        self.gaze_history = deque(maxlen=100)
        self.attention_history = deque(maxlen=50)
        self.blink_history = deque(maxlen=30)
        self.saccade_history = deque(maxlen=20)
        
        # Métricas acumuladas
        self.session_metrics = {
            'total_fixations': 0,
            'total_saccades': 0,
            'attention_switches': 0,
            'out_of_zone_time': 0,
            'start_time': time.time()
        }
        
        # Estados atuais
        self.current_attention_zone = None
        self.current_fixation_duration = 0
        self.last_gaze_timestamp = None
        
        print("👁️ Analisador de Consciência Situacional inicializado")
    
    def analyze_gaze_data(self, gaze_vectors, screen_point=None, timestamp=None) -> Dict:
        """Análise principal dos dados de gaze"""
        if timestamp is None:
            timestamp = time.time()
        
        if not gaze_vectors:
            return {}
        
        # Adiciona ao histórico
        gaze_data = {
            'timestamp': timestamp,
            'gaze_vectors': gaze_vectors,
            'screen_point': screen_point
        }
        self.gaze_history.append(gaze_data)
        
        # Análises principais
        attention_analysis = self.analyze_attention_distribution(gaze_data)
        fixation_analysis = self.analyze_fixation_patterns(gaze_data)
        saccade_analysis = self.analyze_saccade_patterns(gaze_data)
        fatigue_analysis = self.analyze_fatigue_indicators(gaze_data)
        
        # Análise de consciência situacional
        sa_score = self.calculate_situational_awareness_score({
            'attention': attention_analysis,
            'fixation': fixation_analysis,
            'saccade': saccade_analysis,
            'fatigue': fatigue_analysis
        })
        
        return {
            'attention_distribution': attention_analysis,
            'fixation_patterns': fixation_analysis,
            'saccade_patterns': saccade_analysis,
            'fatigue_indicators': fatigue_analysis,
            'situational_awareness_score': sa_score,
            'timestamp': timestamp
        }
    
    def analyze_attention_distribution(self, gaze_data) -> Dict:
        """Analisa distribuição da atenção nas áreas críticas"""
        screen_point = gaze_data['screen_point']
        
        if screen_point is None:
            return {
                'current_zone': 'unknown',
                'zone_confidence': 0.0,
                'attention_switching_rate': 0.0,
                'zone_coverage': {}
            }
        
        # Determina zona de atenção atual
        current_zone = self.get_attention_zone(screen_point)
        zone_confidence = self.calculate_zone_confidence(screen_point, current_zone)
        
        # Atualiza histórico de atenção
        self.attention_history.append({
            'timestamp': gaze_data['timestamp'],
            'zone': current_zone,
            'confidence': zone_confidence,
            'screen_point': screen_point
        })
        
        # Calcula taxa de mudança de atenção
        switching_rate = self.calculate_attention_switching_rate()
        
        # Calcula cobertura das zonas
        zone_coverage = self.calculate_zone_coverage()
        
        # Atualiza estado atual
        if current_zone != self.current_attention_zone:
            self.session_metrics['attention_switches'] += 1
            self.current_attention_zone = current_zone
        
        return {
            'current_zone': current_zone,
            'zone_confidence': zone_confidence,
            'attention_switching_rate': switching_rate,
            'zone_coverage': zone_coverage,
            'total_switches': self.session_metrics['attention_switches']
        }
    
    def get_attention_zone(self, screen_point) -> str:
        """Determina qual zona de atenção contém o ponto"""
        x, y = screen_point
        
        for zone_name, rectangles in self.attention_zones.items():
            for rect in rectangles:
                rect_x, rect_y, rect_w, rect_h = rect
                if (rect_x <= x <= rect_x + rect_w and 
                    rect_y <= y <= rect_y + rect_h):
                    return zone_name
        
        return 'out_of_zone'
    
    def calculate_zone_confidence(self, screen_point, zone) -> float:
        """Calcula confiança da classificação da zona"""
        if zone == 'out_of_zone':
            return 1.0
        
        x, y = screen_point
        
        # Encontra o retângulo da zona
        for rect in self.attention_zones[zone]:
            rect_x, rect_y, rect_w, rect_h = rect
            if (rect_x <= x <= rect_x + rect_w and 
                rect_y <= y <= rect_y + rect_h):
                
                # Calcula distância do centro
                center_x = rect_x + rect_w / 2
                center_y = rect_y + rect_h / 2
                
                distance_from_center = math.sqrt(
                    ((x - center_x) / (rect_w / 2)) ** 2 + 
                    ((y - center_y) / (rect_h / 2)) ** 2
                )
                
                # Confiança baseada na proximidade do centro
                confidence = max(0.1, 1.0 - distance_from_center)
                return confidence
        
        return 0.1
    
    def calculate_attention_switching_rate(self) -> float:
        """Calcula taxa de mudança de atenção"""
        if len(self.attention_history) < 2:
            return 0.0
        
        # Conta mudanças nos últimos 10 segundos
        current_time = time.time()
        recent_attention = [
            att for att in self.attention_history 
            if current_time - att['timestamp'] <= 10.0
        ]
        
        if len(recent_attention) < 2:
            return 0.0
        
        switches = 0
        for i in range(1, len(recent_attention)):
            if recent_attention[i]['zone'] != recent_attention[i-1]['zone']:
                switches += 1
        
        time_span = recent_attention[-1]['timestamp'] - recent_attention[0]['timestamp']
        return switches / max(time_span, 1.0) if time_span > 0 else 0.0
    
    def calculate_zone_coverage(self) -> Dict:
        """Calcula cobertura das zonas de atenção"""
        if not self.attention_history:
            return {}
        
        # Conta tempo em cada zona (últimos 30 segundos)
        current_time = time.time()
        recent_attention = [
            att for att in self.attention_history 
            if current_time - att['timestamp'] <= 30.0
        ]
        
        zone_times = defaultdict(float)
        total_time = 0
        
        for i in range(1, len(recent_attention)):
            duration = recent_attention[i]['timestamp'] - recent_attention[i-1]['timestamp']
            zone_times[recent_attention[i-1]['zone']] += duration
            total_time += duration
        
        # Converte para percentuais
        coverage = {}
        for zone, time_spent in zone_times.items():
            coverage[zone] = (time_spent / total_time) * 100 if total_time > 0 else 0
        
        return coverage
    
    def analyze_fixation_patterns(self, gaze_data) -> Dict:
        """Analisa padrões de fixação"""
        if len(self.gaze_history) < 3:
            return {
                'is_fixating': False,
                'fixation_duration': 0,
                'fixation_stability': 0,
                'average_fixation_duration': 0
            }
        
        # Calcula estabilidade baseada nos últimos pontos
        recent_points = list(self.gaze_history)[-5:]  # Últimos 5 pontos
        
        if not all(point['screen_point'] for point in recent_points):
            return {
                'is_fixating': False,
                'fixation_duration': 0,
                'fixation_stability': 0,
                'average_fixation_duration': 0
            }
        
        # Calcula variação de posição
        positions = np.array([point['screen_point'] for point in recent_points])
        position_variance = np.var(positions, axis=0)
        total_variance = np.sum(position_variance)
        
        # Determina se está fixando
        is_fixating = total_variance < self.fixation_threshold
        
        # Atualiza duração da fixação atual
        if is_fixating:
            if self.last_gaze_timestamp:
                self.current_fixation_duration += (gaze_data['timestamp'] - self.last_gaze_timestamp)
        else:
            if self.current_fixation_duration > 0:
                self.session_metrics['total_fixations'] += 1
            self.current_fixation_duration = 0
        
        self.last_gaze_timestamp = gaze_data['timestamp']
        
        # Calcula estabilidade
        stability_score = 1.0 / (1.0 + total_variance / 1000.0)
        
        # Calcula duração média de fixações
        avg_fixation = self.calculate_average_fixation_duration()
        
        return {
            'is_fixating': is_fixating,
            'fixation_duration': self.current_fixation_duration,
            'fixation_stability': stability_score,
            'position_variance': float(total_variance),
            'total_fixations': self.session_metrics['total_fixations'],
            'average_fixation_duration': avg_fixation
        }
    
    def calculate_average_fixation_duration(self) -> float:
        """Calcula duração média das fixações"""
        # Implementação simplificada baseada no histórico
        if self.session_metrics['total_fixations'] == 0:
            return 0.0
        
        session_time = time.time() - self.session_metrics['start_time']
        return session_time / max(self.session_metrics['total_fixations'], 1)
    
    def analyze_saccade_patterns(self, gaze_data) -> Dict:
        """Analisa padrões de sacadas"""
        if len(self.gaze_history) < 2:
            return {
                'is_saccade': False,
                'saccade_velocity': 0,
                'saccade_amplitude': 0,
                'saccade_frequency': 0
            }
        
        current_gaze = gaze_data['gaze_vectors']
        previous_gaze_data = self.gaze_history[-2]
        previous_gaze = previous_gaze_data['gaze_vectors']
        
        if not current_gaze or not previous_gaze:
            return {
                'is_saccade': False,
                'saccade_velocity': 0,
                'saccade_amplitude': 0,
                'saccade_frequency': 0
            }
        
        # Calcula velocidade de movimento ocular
        time_delta = gaze_data['timestamp'] - previous_gaze_data['timestamp']
        
        if time_delta <= 0:
            return {
                'is_saccade': False,
                'saccade_velocity': 0,
                'saccade_amplitude': 0,
                'saccade_frequency': 0
            }
        
        # Média dos dois olhos
        velocities = []
        amplitudes = []
        
        for eye_side in ['left', 'right']:
            if eye_side in current_gaze and eye_side in previous_gaze:
                curr_yaw = current_gaze[eye_side]['yaw']
                curr_pitch = current_gaze[eye_side]['pitch']
                prev_yaw = previous_gaze[eye_side]['yaw']
                prev_pitch = previous_gaze[eye_side]['pitch']
                
                # Diferença angular
                yaw_diff = curr_yaw - prev_yaw
                pitch_diff = curr_pitch - prev_pitch
                
                # Amplitude (distância angular)
                amplitude = math.sqrt(yaw_diff**2 + pitch_diff**2)
                amplitudes.append(amplitude)
                
                # Velocidade angular
                velocity = amplitude / time_delta
                velocities.append(velocity)
        
        if not velocities:
            return {
                'is_saccade': False,
                'saccade_velocity': 0,
                'saccade_amplitude': 0,
                'saccade_frequency': 0
            }
        
        avg_velocity = np.mean(velocities)
        avg_amplitude = np.mean(amplitudes)
        
        # Determina se é uma sacada
        is_saccade = avg_velocity > math.radians(self.saccade_threshold)
        
        # Atualiza histórico de sacadas
        self.saccade_history.append({
            'timestamp': gaze_data['timestamp'],
            'velocity': avg_velocity,
            'amplitude': avg_amplitude,
            'is_saccade': is_saccade
        })
        
        if is_saccade:
            self.session_metrics['total_saccades'] += 1
        
        # Calcula frequência de sacadas
        saccade_frequency = self.calculate_saccade_frequency()
        
        return {
            'is_saccade': is_saccade,
            'saccade_velocity': math.degrees(avg_velocity),
            'saccade_amplitude': math.degrees(avg_amplitude),
            'saccade_frequency': saccade_frequency,
            'total_saccades': self.session_metrics['total_saccades']
        }
    
    def calculate_saccade_frequency(self) -> float:
        """Calcula frequência de sacadas por segundo"""
        if len(self.saccade_history) < 2:
            return 0.0
        
        # Conta sacadas nos últimos 10 segundos
        current_time = time.time()
        recent_saccades = [
            sacc for sacc in self.saccade_history 
            if current_time - sacc['timestamp'] <= 10.0 and sacc['is_saccade']
        ]
        
        return len(recent_saccades) / 10.0
    
    def analyze_fatigue_indicators(self, gaze_data) -> Dict:
        """Analisa indicadores de fadiga"""
        # Análise básica de fadiga baseada em padrões de movimento
        fatigue_score = 0.0
        indicators = {}
        
        # 1. Taxa de piscadas (simulada - seria preciso detectar piscadas reais)
        blink_rate = self.estimate_blink_rate()
        indicators['blink_rate'] = blink_rate
        
        if blink_rate > self.fatigue_threshold * 2:  # Piscadas excessivas
            fatigue_score += 0.3
        elif blink_rate < self.fatigue_threshold * 0.5:  # Poucas piscadas
            fatigue_score += 0.2
        
        # 2. Duração de fixações (fixações muito longas podem indicar fadiga)
        if hasattr(self, 'current_fixation_duration'):
            long_fixation = self.current_fixation_duration > 3.0  # 3 segundos
            indicators['long_fixation'] = long_fixation
            if long_fixation:
                fatigue_score += 0.2
        
        # 3. Variabilidade de sacadas
        saccade_variability = self.calculate_saccade_variability()
        indicators['saccade_variability'] = saccade_variability
        if saccade_variability < 0.3:  # Baixa variabilidade
            fatigue_score += 0.2
        
        # 4. Atenção fora das zonas críticas
        out_of_zone_ratio = self.calculate_out_of_zone_ratio()
        indicators['out_of_zone_ratio'] = out_of_zone_ratio
        if out_of_zone_ratio > 0.4:  # Mais de 40% fora das zonas
            fatigue_score += 0.3
        
        return {
            'fatigue_score': min(1.0, fatigue_score),
            'indicators': indicators,
            'fatigue_level': self.classify_fatigue_level(fatigue_score)
        }
    
    def estimate_blink_rate(self) -> float:
        """Estima taxa de piscadas baseada na instabilidade do gaze"""
        if len(self.gaze_history) < 5:
            return 0.5  # Valor padrão
        
        # Conta "interruptions" no gaze que podem indicar piscadas
        interruptions = 0
        recent_gaze = list(self.gaze_history)[-10:]  # Últimos 10 pontos
        
        for i in range(1, len(recent_gaze)):
            current = recent_gaze[i]
            previous = recent_gaze[i-1]
            
            # Se não há dados de gaze, pode ser uma piscada
            if not current['gaze_vectors'] and previous['gaze_vectors']:
                interruptions += 1
        
        # Estima baseado no tempo total
        if len(recent_gaze) > 1:
            time_span = recent_gaze[-1]['timestamp'] - recent_gaze[0]['timestamp']
            return interruptions / max(time_span, 1.0)
        
        return 0.5
    
    def calculate_saccade_variability(self) -> float:
        """Calcula variabilidade das sacadas"""
        if len(self.saccade_history) < 3:
            return 0.5
        
        recent_saccades = list(self.saccade_history)[-10:]
        velocities = [sacc['velocity'] for sacc in recent_saccades]
        
        if len(velocities) < 2:
            return 0.5
        
        return np.std(velocities) / (np.mean(velocities) + 0.001)
    
    def calculate_out_of_zone_ratio(self) -> float:
        """Calcula proporção de tempo fora das zonas críticas"""
        if not self.attention_history:
            return 0.0
        
        recent_attention = list(self.attention_history)[-20:]  # Últimos 20 pontos
        out_of_zone_count = sum(1 for att in recent_attention if att['zone'] == 'out_of_zone')
        
        return out_of_zone_count / len(recent_attention)
    
    def classify_fatigue_level(self, fatigue_score) -> str:
        """Classifica nível de fadiga"""
        if fatigue_score < 0.3:
            return 'low'
        elif fatigue_score < 0.6:
            return 'moderate'
        else:
            return 'high'
    
    def calculate_situational_awareness_score(self, analysis_data) -> Dict:
        """Calcula score geral de consciência situacional"""
        attention_data = analysis_data['attention']
        fixation_data = analysis_data['fixation']
        saccade_data = analysis_data['saccade']
        fatigue_data = analysis_data['fatigue']
        
        # Componentes do score
        scores = {}
        
        # 1. Score de Atenção (0-1)
        zone_coverage = attention_data.get('zone_coverage', {})
        critical_zones = ['instruments', 'road', 'mirrors']
        
        attention_score = 0.0
        for zone in critical_zones:
            if zone in zone_coverage:
                attention_score += min(zone_coverage[zone] / 100.0, 0.3)  # Max 30% por zona
        
        # Penaliza tempo excessivo fora das zonas
        out_of_zone = zone_coverage.get('out_of_zone', 0)
        attention_score = max(0, attention_score - (out_of_zone / 100.0) * 0.5)
        scores['attention'] = min(1.0, attention_score)
        
        # 2. Score de Vigilância (baseado em sacadas e fixações)
        saccade_freq = saccade_data.get('saccade_frequency', 0)
        optimal_saccade_freq = 2.0  # Sacadas por segundo
        
        vigilance_score = 1.0 - abs(saccade_freq - optimal_saccade_freq) / optimal_saccade_freq
        vigilance_score = max(0.0, vigilance_score)
        
        # Considera estabilidade de fixações
        fixation_stability = fixation_data.get('fixation_stability', 0)
        vigilance_score = (vigilance_score * 0.7) + (fixation_stability * 0.3)
        scores['vigilance'] = vigilance_score
        
        # 3. Score de Fadiga (inverso do score de fadiga)
        fatigue_score = fatigue_data.get('fatigue_score', 0)
        scores['alertness'] = 1.0 - fatigue_score
        
        # 4. Score de Switching (capacidade de alternar atenção)
        switching_rate = attention_data.get('attention_switching_rate', 0)
        optimal_switching = 0.5  # Mudanças por segundo
        
        switching_score = 1.0 - abs(switching_rate - optimal_switching) / max(optimal_switching, switching_rate)
        switching_score = max(0.0, switching_score)
        scores['attention_switching'] = switching_score
        
        # Score final ponderado
        final_score = (
            scores['attention'] * 0.35 +
            scores['vigilance'] * 0.25 +
            scores['alertness'] * 0.25 +
            scores['attention_switching'] * 0.15
        )
        
        return {
            'overall_score': final_score,
            'component_scores': scores,
            'classification': self.classify_situational_awareness(final_score)
        }
    
    def classify_situational_awareness(self, score) -> str:
        """Classifica nível de consciência situacional"""
        if score >= 0.8:
            return 'excellent'
        elif score >= 0.6:
            return 'good'
        elif score >= 0.4:
            return 'moderate'
        elif score >= 0.2:
            return 'poor'
        else:
            return 'critical'
    
    def generate_session_report(self) -> Dict:
        """Gera relatório da sessão completa"""
        session_time = time.time() - self.session_metrics['start_time']
        
        # Análise de cobertura de zonas
        zone_coverage = self.calculate_zone_coverage()
        
        # Métricas de performance
        fixation_rate = self.session_metrics['total_fixations'] / max(session_time / 60, 1)  # por minuto
        saccade_rate = self.session_metrics['total_saccades'] / max(session_time / 60, 1)  # por minuto
        attention_switch_rate = self.session_metrics['attention_switches'] / max(session_time / 60, 1)
        
        return {
            'session_duration': session_time,
            'total_fixations': self.session_metrics['total_fixations'],
            'total_saccades': self.session_metrics['total_saccades'],
            'attention_switches': self.session_metrics['attention_switches'],
            'fixation_rate_per_minute': fixation_rate,
            'saccade_rate_per_minute': saccade_rate,
            'attention_switch_rate_per_minute': attention_switch_rate,
            'zone_coverage': zone_coverage,
            'performance_summary': self.generate_performance_summary()
        }
    
    def generate_performance_summary(self) -> Dict:
        """Gera resumo de performance"""
        if not self.gaze_history:
            return {'status': 'insufficient_data'}
        
        # Análise dos últimos dados disponíveis
        recent_analysis = []
        for gaze_data in list(self.gaze_history)[-10:]:
            if gaze_data['gaze_vectors']:
                analysis = self.analyze_gaze_data(
                    gaze_data['gaze_vectors'], 
                    gaze_data['screen_point'], 
                    gaze_data['timestamp']
                )
                recent_analysis.append(analysis)
        
        if not recent_analysis:
            return {'status': 'no_valid_data'}
        
        # Médias dos scores
        sa_scores = [a['situational_awareness_score']['overall_score'] for a in recent_analysis]
        avg_sa_score = np.mean(sa_scores)
        
        return {
            'status': 'complete',
            'average_sa_score': avg_sa_score,
            'sa_classification': self.classify_situational_awareness(avg_sa_score),
            'recommendations': self.generate_recommendations(avg_sa_score, recent_analysis[-1] if recent_analysis else {})
        }
    
    def generate_recommendations(self, sa_score, latest_analysis) -> List[str]:
        """Gera recomendações baseadas na performance"""
        recommendations = []
        
        if sa_score < 0.4:
            recommendations.append("⚠️ Consciência situacional crítica - considere pausa para descanso")
        
        if latest_analysis:
            fatigue = latest_analysis.get('fatigue_indicators', {})
            if fatigue.get('fatigue_level') == 'high':
                recommendations.append("😴 Alta fadiga detectada - recomenda-se intervalo")
            
            attention = latest_analysis.get('attention_distribution', {})
            out_of_zone = attention.get('zone_coverage', {}).get('out_of_zone', 0)
            if out_of_zone > 30:
                recommendations.append("👀 Atenção frequentemente fora das zonas críticas")
            
            if attention.get('attention_switching_rate', 0) > 1.0:
                recommendations.append("🔄 Taxa de mudança de atenção muito alta - manter foco")
            elif attention.get('attention_switching_rate', 0) < 0.2:
                recommendations.append("🎯 Baixa alternância de atenção - verificar espelhos e instrumentos")
        
        if not recommendations:
            recommendations.append("✅ Performance adequada - manter padrão atual")
        
        return recommendations
