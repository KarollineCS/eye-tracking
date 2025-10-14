import numpy as np
import time
import math
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

class AdaptiveCalibrationSystem:
    """Sistema de calibração inteligente que se adapta à qualidade dos dados"""
    
    def __init__(self, config):
        self.config = config
        self.quality_threshold = config.calibration.quality_threshold
        self.min_points = config.calibration.min_points
        self.max_points = config.calibration.max_points
        self.stability_samples = 10
        
        # Armazenamento de dados
        self.calibration_samples = defaultdict(list)
        self.point_qualities = {}
        
        # Estados
        self.current_point_samples = []
        self.points_completed = 0
        self.adaptive_points_added = 6
        
        # Configurações de coleta
        self.samples_per_point = config.calibration.samples_per_point
        self.max_samples_per_point = config.calibration.max_samples_per_point
        
        # Status
        self.is_active = False
        self.current_points = []
        self.current_point_index = 0
        
        # Sistema de intervalos
        self.interval_enabled = config.calibration.interval_enabled
        self.interval_duration = config.calibration.interval_duration
        self.countdown_duration = config.calibration.countdown_duration
        self.interval_start_time = None
        self.countdown_start_time = None
        self.is_in_interval = False
        self.is_in_countdown = False
        self.show_ready_message = False
        
        print("🎯 Sistema de Calibração Adaptativa inicializado")
    
    def start_adaptive_calibration(self, screen_width, screen_height) -> Dict:
        """Inicia calibração adaptativa"""
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.is_active = True
        
        # Reset
        self.calibration_samples.clear()
        self.point_qualities.clear()
        self.points_completed = 0
        self.adaptive_points_added = 0
        
        # Grid inicial (3x3)
        margin = 0.15
        self.initial_points = []
        for y in np.linspace(margin, 1-margin, 3):
            for x in np.linspace(margin, 1-margin, 3):
                screen_x = int(x * self.screen_width)
                screen_y = int(y * self.screen_height)
                self.initial_points.append((screen_x, screen_y))
        
        self.current_points = self.initial_points.copy()
        self.current_point_index = 0
        self.current_point_samples = []
        
        # Inicia contagem regressiva
        if self.interval_enabled:
            self.is_in_countdown = True
            self.countdown_start_time = time.time()
            print(f"Iniciando calibração adaptativa com {len(self.current_points)} pontos")
            print(f"Contagem regressiva de {self.countdown_duration:.0f}s iniciando...")
        else:
            print(f"Iniciando calibração adaptativa com {len(self.current_points)} pontos")
        
        return self.get_current_point()
    
    def add_gaze_sample(self, gaze_vectors) -> bool:
        """Adiciona amostra de olhar para o ponto atual"""
        if not self.is_active:
            return False
        
        # Gerencia intervalos e contagem regressiva
        current_time = time.time()
        
        # Contagem regressiva inicial
        if self.is_in_countdown:
            if current_time - self.countdown_start_time >= self.countdown_duration:
                self.is_in_countdown = False
                self.show_ready_message = True
                print(f"PRONTO! Iniciando coleta para o ponto 1/{len(self.current_points)}")
            return False
        
        # Intervalo entre pontos
        if self.is_in_interval:
            if current_time - self.interval_start_time >= self.interval_duration:
                self.is_in_interval = False
                self.show_ready_message = True
                print(f"PRONTO! Iniciando coleta para o ponto {self.current_point_index + 1}/{len(self.current_points)}")
            return False
        
        if not gaze_vectors or 'left' not in gaze_vectors or 'right' not in gaze_vectors:
            return False
        
        # Remove mensagem de "pronto" após primeira amostra
        if self.show_ready_message:
            self.show_ready_message = False
        
        # Calcula dados do olhar
        avg_yaw = (gaze_vectors['left']['yaw'] + gaze_vectors['right']['yaw']) / 2
        avg_pitch = (gaze_vectors['left']['pitch'] + gaze_vectors['right']['pitch']) / 2
        
        sample = {
            'yaw': avg_yaw,
            'pitch': avg_pitch,
            'timestamp': time.time(),
            'point': self.current_points[self.current_point_index]
        }
        
        self.current_point_samples.append(sample)
        
        # Verifica se coletou amostras suficientes
        if len(self.current_point_samples) >= self.samples_per_point:
            quality = self.evaluate_point_quality(self.current_point_samples)
            
            # Se qualidade boa OU atingiu máximo de amostras
            if quality >= self.quality_threshold or len(self.current_point_samples) >= self.max_samples_per_point:
                self.finish_current_point(quality)
                return self.move_to_next_point()
        
        return False
    
    def evaluate_point_quality(self, samples) -> float:
        """Avalia qualidade da coleta para um ponto"""
        if len(samples) < 3:
            return 0.0
        
        # Calcula variância dos ângulos
        yaw_values = [s['yaw'] for s in samples]
        pitch_values = [s['pitch'] for s in samples]
        
        yaw_variance = np.var(yaw_values)
        pitch_variance = np.var(pitch_values)
        
        # Normaliza variância (menor = melhor)
        stability_score = 1.0 / (1.0 + (yaw_variance + pitch_variance) * 100)
        
        # Considera também consistência temporal
        temporal_score = self.evaluate_temporal_consistency(samples)
        
        # Combina scores
        quality = (stability_score * 0.7) + (temporal_score * 0.3)
        return min(1.0, quality)
    
    def evaluate_temporal_consistency(self, samples) -> float:
        """Avalia consistência temporal das amostras"""
        if len(samples) < 5:
            return 0.5
        
        # Verifica se há drift ao longo do tempo
        timestamps = [s['timestamp'] for s in samples]
        yaw_values = [s['yaw'] for s in samples]
        
        # Normaliza timestamps
        min_time = min(timestamps)
        normalized_times = [(t - min_time) for t in timestamps]
        
        # Calcula correlação entre tempo e posição (menor = mais estável)
        if len(set(normalized_times)) > 1 and len(set(yaw_values)) > 1:
            correlation = abs(np.corrcoef(normalized_times, yaw_values)[0, 1])
            consistency_score = 1.0 - min(1.0, correlation)
        else:
            consistency_score = 1.0
        
        return consistency_score
    
    def finish_current_point(self, quality):
        """Finaliza coleta do ponto atual"""
        point = self.current_points[self.current_point_index]
        
        # Armazena dados do ponto
        self.calibration_samples[point] = self.current_point_samples.copy()
        self.point_qualities[point] = quality
        
        print(f"Ponto {self.current_point_index + 1}/{len(self.current_points)} concluído")
        print(f"Qualidade: {quality:.3f} ({len(self.current_point_samples)} amostras)")
        
        # Reset para próximo ponto
        self.current_point_samples = []
        self.points_completed += 1
    
    def move_to_next_point(self):
        """Move para próximo ponto ou finaliza calibração"""
        self.current_point_index += 1
        
        # Se terminou pontos iniciais, avalia se precisa de pontos adaptativos
        if self.current_point_index >= len(self.current_points):
            if self.points_completed < self.max_points and self.adaptive_points_added < 6:
                additional_points = self.identify_additional_points()
                if additional_points:
                    self.current_points.extend(additional_points)
                    self.adaptive_points_added += len(additional_points)
                    print(f"Adicionando {len(additional_points)} pontos adaptativos")
                    
                    # Inicia intervalo antes do próximo ponto
                    if self.interval_enabled and len(additional_points) > 0:
                        self.is_in_interval = True
                        self.interval_start_time = time.time()
                        print(f"Intervalo de {self.interval_duration:.0f}s...")
                    
                    return self.get_current_point()
            
            # Calibração concluída
            return self.finalize_calibration()
        
        # Inicia intervalo antes do próximo ponto
        if self.interval_enabled:
            self.is_in_interval = True
            self.interval_start_time = time.time()
            print(f"Ponto concluído! Intervalo de {self.interval_duration:.0f}s...")
        
        return self.get_current_point()
    
    def identify_additional_points(self) -> List[Tuple[int, int]]:
        """Identifica pontos adicionais baseado na qualidade"""
        additional_points = []
        
        # Encontra pontos com qualidade baixa
        low_quality_points = [
            point for point, quality in self.point_qualities.items() 
            if quality < self.quality_threshold
        ]
        
        # Se não há pontos de baixa qualidade, adiciona pontos nas bordas
        if not low_quality_points:
            border_points = [
                (int(0.05 * self.screen_width), int(0.5 * self.screen_height)),  # Esquerda
                (int(0.95 * self.screen_width), int(0.5 * self.screen_height)),  # Direita
                (int(0.5 * self.screen_width), int(0.05 * self.screen_height)),  # Topo
                (int(0.5 * self.screen_width), int(0.95 * self.screen_height))   # Baixo
            ]
            
            for point in border_points:
                if point not in self.current_points:
                    additional_points.append(point)
                    if len(additional_points) >= 2:
                        break
        else:
            # Adiciona pontos próximos aos de baixa qualidade
            for point in low_quality_points[:2]:
                x, y = point
                offset_x = 100 if x < self.screen_width / 2 else -100
                offset_y = 80 if y < self.screen_height / 2 else -80
                
                new_x = max(50, min(self.screen_width - 50, x + offset_x))
                new_y = max(50, min(self.screen_height - 50, y + offset_y))
                new_point = (new_x, new_y)
                
                if new_point not in self.current_points:
                    additional_points.append(new_point)
                    if len(additional_points) >= 3:
                        break
        
        return additional_points
    
    def get_current_point(self) -> Optional[Dict]:
        """Retorna ponto atual para calibração"""
        if self.current_point_index < len(self.current_points):
            point = self.current_points[self.current_point_index]
            samples_collected = len(self.current_point_samples)
            samples_needed = self.samples_per_point - samples_collected
            
            return {
                'point': point,
                'index': self.current_point_index,
                'total_points': len(self.current_points),
                'samples_collected': samples_collected,
                'samples_needed': max(0, samples_needed),
                'progress': min(1.0, samples_collected / self.samples_per_point),
                'is_in_interval': self.is_in_interval,
                'is_in_countdown': self.is_in_countdown,
                'show_ready_message': self.show_ready_message,
                'countdown_remaining': max(0, self.countdown_duration - (time.time() - self.countdown_start_time)) if self.is_in_countdown else 0,
                'interval_remaining': max(0, self.interval_duration - (time.time() - self.interval_start_time)) if self.is_in_interval else 0
            }
        return None
    
    def finalize_calibration(self):
        """Finaliza calibração e prepara dados para treinamento"""
        self.is_active = False
        
        print("Finalizando calibração adaptativa...")
        
        # Converte dados para formato compatível
        training_data = []
        for point, samples in self.calibration_samples.items():
            if not samples:
                continue
            
            # Usa mediana das amostras (mais robusta que média)
            yaw_values = [s['yaw'] for s in samples]
            pitch_values = [s['pitch'] for s in samples]
            
            yaw_median = np.median(yaw_values)
            pitch_median = np.median(pitch_values)
            
            training_data.append({
                'yaw': yaw_median,
                'pitch': pitch_median,
                'screen_x': point[0],
                'screen_y': point[1],
                'quality': self.point_qualities.get(point, 0.0),
                'samples_count': len(samples)
            })
        
        # Gera relatório de qualidade
        self.generate_quality_report()
        
        return training_data
    
    def generate_quality_report(self):
        """Gera relatório da qualidade da calibração"""
        if not self.point_qualities:
            return
        
        qualities = list(self.point_qualities.values())
        avg_quality = np.mean(qualities)
        min_quality = np.min(qualities)
        max_quality = np.max(qualities)
        
        print("\n=== RELATÓRIO DE CALIBRAÇÃO ADAPTATIVA ===")
        print(f"Pontos coletados: {len(self.point_qualities)}")
        print(f"Pontos adaptativos adicionados: {self.adaptive_points_added}")
        print(f"Qualidade média: {avg_quality:.3f}")
        print(f"Qualidade mínima: {min_quality:.3f}")
        print(f"Qualidade máxima: {max_quality:.3f}")
        print(f"Pontos com alta qualidade (>0.75): {sum(1 for q in qualities if q > 0.75)}")
        
        # Estimativa de precisão
        estimated_error = (1.0 - avg_quality) * 80 + 15  # 15-95px
        print(f"Precisão estimada: ±{estimated_error:.0f} pixels")
        print("==========================================\n")
    
    def get_calibration_status(self) -> Dict:
        """Retorna status atual da calibração"""
        if self.is_active and self.current_point_index < len(self.current_points):
            current_info = self.get_current_point()
            return {
                'active': True,
                'current_point': current_info,
                'total_completed': self.points_completed,
            }
        return {'active': False, 'completed': True}
    
    def set_interval_settings(self, enabled=True, interval_duration=2.0, countdown_duration=3.0):
        """Configura o sistema de intervalos"""
        self.interval_enabled = enabled
        self.interval_duration = interval_duration
        self.countdown_duration = countdown_duration
        
        status = "ativados" if enabled else "desativados"
        print(f"Intervalos {status}. Duração: {interval_duration}s entre pontos, contagem regressiva: {countdown_duration}s")


class ScreenCalibrationSystem:
    """Sistema de calibração de tela com múltiplos métodos"""
    
    def __init__(self, config):
        self.config = config
        self.calibration_method = "hybrid"
        self.is_trained = False
        self.ml_model_x = Ridge(alpha=1.0)
        self.ml_model_y = Ridge(alpha=1.0)
        self.scaler = StandardScaler()
        self.geometric_params = None
        self.calibration_data = []
    
    def train_calibration_models(self, calibration_data) -> bool:
        """Treina modelos de calibração"""
        self.calibration_data = calibration_data
        if len(calibration_data) < 4:
            print("Dados insuficientes para calibração")
            return False
        
        X = np.array([[d['yaw'], d['pitch']] for d in calibration_data])
        y_screen_x = np.array([d['screen_x'] for d in calibration_data])
        y_screen_y = np.array([d['screen_y'] for d in calibration_data])
        
        if self.calibration_method == "ml":
            self._train_ml_model(X, y_screen_x, y_screen_y)
        elif self.calibration_method == "geometric":
            self._train_geometric_model(calibration_data)
        elif self.calibration_method == "hybrid":
            self._train_hybrid_model(X, y_screen_x, y_screen_y)
        
        self.is_trained = True
        return True
    
    def _train_ml_model(self, X, y_screen_x, y_screen_y):
        """Treina modelo de machine learning"""
        X_scaled = self.scaler.fit_transform(X)
        self.ml_model_x.fit(X_scaled, y_screen_x)
        self.ml_model_y.fit(X_scaled, y_screen_y)
        print("Modelo ML treinado")
    
    def _train_geometric_model(self, calibration_data):
        """Treina modelo geométrico"""
        X = np.array([[d['yaw'], d['pitch'], 1] for d in calibration_data])
        y_x = np.array([d['screen_x'] for d in calibration_data])
        y_y = np.array([d['screen_y'] for d in calibration_data])
        
        params_x = np.linalg.lstsq(X, y_x, rcond=None)[0]
        params_y = np.linalg.lstsq(X, y_y, rcond=None)[0]
        
        self.geometric_params = {'x_params': params_x, 'y_params': params_y}
        print("Modelo Geométrico treinado")
    
    def _train_hybrid_model(self, X, y_screen_x, y_screen_y):
        """Treina modelo híbrido"""
        self._train_geometric_model(self.calibration_data)
        
        if self.geometric_params:
            X_geo = np.column_stack([X, np.ones(X.shape[0])])
            pred_x = X_geo @ self.geometric_params['x_params']
            pred_y = X_geo @ self.geometric_params['y_params']
            
            residual_x = y_screen_x - pred_x
            residual_y = y_screen_y - pred_y
            
            X_scaled = self.scaler.fit_transform(X)
            self.ml_model_x.fit(X_scaled, residual_x)
            self.ml_model_y.fit(X_scaled, residual_y)
        
        print("Modelo Híbrido treinado")
    
    def predict_screen_point(self, yaw, pitch) -> Optional[Tuple[float, float]]:
        """Prediz ponto na tela baseado no gaze"""
        if not self.is_trained:
            return None
        
        X_input = np.array([[yaw, pitch]])
        
        if self.calibration_method == "ml":
            X_scaled = self.scaler.transform(X_input)
            screen_x = self.ml_model_x.predict(X_scaled)[0]
            screen_y = self.ml_model_y.predict(X_scaled)[0]
        
        elif self.calibration_method == "geometric":
            X_geo = np.array([[yaw, pitch, 1]])
            screen_x = X_geo @ self.geometric_params['x_params']
            screen_y = X_geo @ self.geometric_params['y_params']
            screen_x, screen_y = screen_x[0], screen_y[0]
        
        elif self.calibration_method == "hybrid":
            X_geo = np.array([[yaw, pitch, 1]])
            base_x = X_geo @ self.geometric_params['x_params']
            base_y = X_geo @ self.geometric_params['y_params']
            
            X_scaled = self.scaler.transform(X_input)
            correction_x = self.ml_model_x.predict(X_scaled)[0]
            correction_y = self.ml_model_y.predict(X_scaled)[0]
            
            screen_x = base_x[0] + correction_x
            screen_y = base_y[0] + correction_y
        
        return (screen_x, screen_y)
    
    def evaluate_calibration_accuracy(self) -> Optional[Dict]:
        """Avalia precisão da calibração"""
        if not self.is_trained or not self.calibration_data:
            return None
        
        errors = []
        for data in self.calibration_data:
            pred = self.predict_screen_point(data['yaw'], data['pitch'])
            if pred:
                error = np.sqrt((pred[0] - data['screen_x'])**2 + (pred[1] - data['screen_y'])**2)
                errors.append(error)
        
        if errors:
            return {
                'mean_error': np.mean(errors),
                'std_error': np.std(errors),
                'max_error': np.max(errors),
                'min_error': np.min(errors)
            }
        return None
    
    def switch_calibration_method(self, method) -> bool:
        """Altera método de calibração"""
        if method in ["geometric", "ml", "hybrid"]:
            self.calibration_method = method
            if self.calibration_data:
                self.train_calibration_models(self.calibration_data)
            print(f"Método alterado para: {method}")
            return True
        return False
    
    def get_current_method(self) -> str:
        return self.calibration_method.upper()
    
    def is_calibrated(self) -> bool:
        return self.is_trained
    
    def reset(self):
        """Reseta sistema de calibração"""
        self.is_trained = False
        self.calibration_data = []
        self.geometric_params = None
        print("Sistema de calibração resetado")
