import cv2
import numpy as np
import time
from typing import Tuple, List, Dict, Optional
import yaml
import os
from collections import deque
import threading
import keyboard


class AdvancedScreenCalibration:
    """
    Sistema de calibração avançado com modelo polinomial
    """
    
    def __init__(self, screen_width: int, screen_height: int):
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        # Pontos de calibração
        self.calibration_points = self._generate_calibration_points()
        self.current_point_idx = 0
        
        # Dados coletados
        self.collected_data = []
        self.calibration_data = []  
        self.current_samples = []
        self.gaze_samples = deque(maxlen=30)  
        
        # Modelos
        self.model = None 
        self.model_x = None
        self.model_y = None
        self.poly_features = None
        self.transformation_matrix = None
        self.use_polynomial_model = False
        
        # Estado
        self.calibration_complete = False
        self.collection_started = False  
        self.space_pressed = False
        self.enter_pressed = False
        self.calibration_aborted = False
        
        # Parâmetros
        self.min_samples_per_point = 20
        self.max_samples_per_point = 50
        self.point_radius = 25
        self.point_display_time = 2.0  
        
        # Monitor
        self.monitor_mm = (500, 300)
        self.monitor_distance = 600
        
        # Estatísticas
        self.calibration_stats = {}
        self.last_gaze_data = None
        
    
    def _generate_calibration_points(self) -> List[Tuple[int, int]]:
        """Gera pontos de calibração"""
        points = []
        margins = 0.1
        
        # Grade 3x3 padrão para compatibilidade
        for y in [margins, 0.5, 1 - margins]:
            for x in [margins, 0.5, 1 - margins]:
                px = int(x * self.screen_width)
                py = int(y * self.screen_height)
                points.append((px, py))  # Tupla para compatibilidade
        
        return points
    
    def is_calibrated(self) -> bool:
        """Verifica se está calibrado"""
        return self.calibration_complete and (
            self.transformation_matrix is not None or 
            self.model is not None
        )
    
    def start_calibration(self):
        self.current_point_idx = 0
        self.collected_data = []
        self.calibration_data = []
        self.calibration_complete = False
        self.collection_started = True
        time.sleep(2)
        
        # Criar janela
        cv2.namedWindow('Calibration', cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty('Calibration', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    def display_calibration_point(self) -> np.ndarray:
        """Exibe ponto de calibração (compatível)"""
        screen = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)
        
        if self.current_point_idx < len(self.calibration_points):
            point = self.calibration_points[self.current_point_idx]
            
            # Converter tupla para uso
            if isinstance(point, tuple):
                point = list(point)
            
            # Cor baseada no progresso
            samples_collected = len(self.gaze_samples)
            if samples_collected < 15:
                color = (0, 0, 255)  # Vermelho
            elif samples_collected < 25:
                color = (0, 165, 255)  # Laranja
            else:
                color = (0, 255, 0)  # Verde
            
            # Desenhar ponto
            cv2.circle(screen, tuple(point), self.point_radius, color, -1)
            cv2.circle(screen, tuple(point), self.point_radius + 5, color, 2)
            
            # Crosshair
            cv2.line(screen, 
                    (point[0] - 10, point[1]), 
                    (point[0] + 10, point[1]), 
                    (255, 255, 255), 1)
            cv2.line(screen, 
                    (point[0], point[1] - 10), 
                    (point[0], point[1] + 10), 
                    (255, 255, 255), 1)
            
            # Progresso
            progress_text = f"Ponto {self.current_point_idx + 1}/{len(self.calibration_points)}"
            cv2.putText(screen, progress_text, (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Barra de progresso
            progress_width = int((samples_collected / 30) * 200)
            cv2.rectangle(screen, (50, 70), (250, 90), (100, 100, 100), -1)
            cv2.rectangle(screen, (50, 70), (50 + progress_width, 90), color, -1)
        
        return screen
    
    def collect_gaze_sample(self, gaze_data: Dict):
        """Coleta amostra de gaze (compatível com código existente)"""
        if gaze_data and self.collection_started:
            # Adicionar à deque para compatibilidade
            self.gaze_samples.append({
                'gaze': (gaze_data.get('yaw', 0), gaze_data.get('pitch', 0)),
                'timestamp': time.time()
            })
            
            # Verificar se coletou suficiente
            if len(self.gaze_samples) >= 30:
                self._process_calibration_point()
    
    def _process_calibration_point(self):
        """Processa ponto atual (compatível)"""
        if len(self.gaze_samples) > 0:
            # Calcular média
            gazes = [s['gaze'] for s in self.gaze_samples]
            avg_yaw = np.mean([g[0] for g in gazes])
            avg_pitch = np.mean([g[1] for g in gazes])
            std_yaw = np.std([g[0] for g in gazes])
            std_pitch = np.std([g[1] for g in gazes])
            
            # Armazenar
            screen_point = self.calibration_points[self.current_point_idx]
            self.collected_data.append({
                'screen_point': screen_point,
                'gaze_point': (avg_yaw, avg_pitch),
                'std': (std_yaw, std_pitch),
                'samples': len(self.gaze_samples)
            })
            
            # Para compatibilidade com train_model
            self.calibration_data.append({
                'screen_x': screen_point[0],
                'screen_y': screen_point[1],
                'samples': [{'yaw': g[0], 'pitch': g[1]} for g in gazes]
            })
            
            # Limpar e avançar
            self.gaze_samples.clear()
            self.current_point_idx += 1
            self.collection_started = False
            
            time.sleep(0.5)
            self.collection_started = True
            
            # Verificar se terminou
            if self.current_point_idx >= len(self.calibration_points):
                self._finalize_calibration()
    
    def _finalize_calibration(self):
        """Finaliza calibração"""
        
        if len(self.collected_data) >= 4:
            # Tentar modelo avançado primeiro
            success = False
            
            # Tentar sklearn se disponível
            try:
                success = self._train_advanced_model()
            except ImportError:
                success = False
            
            # Fallback para homografia
            if not success:
                success = self._train_homography_model()
            
            if success:
                # Calcular erro
                self._calculate_error()
                
                # Salvar
                self._save_calibration()
                
                self.calibration_complete = True
            else:
                self.calibration_complete = False
        else:
            self.calibration_complete = False
        
        cv2.destroyWindow('Calibration')
    
    def _train_advanced_model(self) -> bool:
        """Treina modelo avançado se sklearn disponível"""
        try:
            from sklearn.preprocessing import PolynomialFeatures
            from sklearn.linear_model import Ridge
            
            # Preparar dados
            X = []
            y_x = []
            y_y = []
            
            for data in self.collected_data:
                X.append(list(data['gaze_point']))
                y_x.append(data['screen_point'][0])
                y_y.append(data['screen_point'][1])
            
            X = np.array(X)
            y_x = np.array(y_x)
            y_y = np.array(y_y)
            
            # Criar features polinomiais
            self.poly_features = PolynomialFeatures(degree=2, include_bias=True)
            X_poly = self.poly_features.fit_transform(X)
            
            # Treinar modelos
            self.model_x = Ridge(alpha=0.1)
            self.model_y = Ridge(alpha=0.1)
            
            self.model_x.fit(X_poly, y_x)
            self.model_y.fit(X_poly, y_y)
            
            self.use_polynomial_model = True
            return True
            
        except Exception:
            return False
    
    def _train_homography_model(self) -> bool:
        """Treina modelo de homografia"""
        try:
            src_points = np.array([d['gaze_point'] for d in self.collected_data], 
                                 dtype=np.float32)
            dst_points = np.array([d['screen_point'] for d in self.collected_data], 
                                 dtype=np.float32)
            
            self.transformation_matrix, _ = cv2.findHomography(
                src_points, dst_points, cv2.RANSAC, 5.0
            )
            
            return self.transformation_matrix is not None
            
        except Exception:
            return False
    
    def _calculate_error(self):
        """Calcula erro de calibração"""
        errors = []
        
        for data in self.collected_data:
            gaze = data['gaze_point']
            actual = data['screen_point']
            
            # Mapear
            predicted = self.map_gaze_to_screen(gaze[0], gaze[1])
            
            if predicted:
                error = np.sqrt((predicted[0] - actual[0])**2 + 
                              (predicted[1] - actual[1])**2)
                errors.append(error)
        
        if errors:
            avg_error = np.mean(errors)
            std_error = np.std(errors)
            
            self.calibration_stats = {
                'avg_error': avg_error,
                'std_error': std_error,
                'errors': errors
            }
    
    def map_gaze_to_screen(self, gaze_yaw: float, gaze_pitch: float) -> Optional[Tuple[int, int]]:
        """Mapeia gaze para tela"""
        # Tentar modelo polinomial primeiro
        if self.use_polynomial_model and self.model_x and self.model_y:
            try:
                X = np.array([[gaze_yaw, gaze_pitch]])
                X_poly = self.poly_features.transform(X)
                
                x = int(self.model_x.predict(X_poly)[0])
                y = int(self.model_y.predict(X_poly)[0])
                
                x = np.clip(x, 0, self.screen_width - 1)
                y = np.clip(y, 0, self.screen_height - 1)
                
                return (x, y)
            except:
                pass
        
        # Fallback para homografia
        if self.transformation_matrix is not None:
            try:
                gaze_point = np.array([gaze_yaw, gaze_pitch, 1])
                screen_point = self.transformation_matrix @ gaze_point
                
                if screen_point[2] != 0:
                    x = int(screen_point[0] / screen_point[2])
                    y = int(screen_point[1] / screen_point[2])
                    
                    x = np.clip(x, 0, self.screen_width - 1)
                    y = np.clip(y, 0, self.screen_height - 1)
                    
                    return (x, y)
            except:
                pass
        
        return None
    
    def _save_calibration(self):
        """Salva calibração"""
        calibration_data = {
            'screen_dimensions': [self.screen_width, self.screen_height],
            'calibration_points': self.calibration_points,
            'collected_data': self.collected_data,
            'transformation_matrix': self.transformation_matrix.tolist() if self.transformation_matrix is not None else None,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'monitor_info': {
                'size_mm': self.monitor_mm,
                'distance_mm': self.monitor_distance
            },
            'use_polynomial_model': self.use_polynomial_model,
            'calibration_stats': self.calibration_stats
        }
        
        os.makedirs('calibration', exist_ok=True)
        
        with open('calibration/screen_calibration.yaml', 'w') as f:
            yaml.dump(calibration_data, f)
        
        # Salvar modelos sklearn se existirem
        if self.use_polynomial_model and self.model_x and self.model_y:
            try:
                import pickle
                models = {
                    'model_x': self.model_x,
                    'model_y': self.model_y,
                    'poly_features': self.poly_features
                }
                with open('calibration/models.pkl', 'wb') as f:
                    pickle.dump(models, f)
            except:
                pass
    
    def load_calibration(self, filepath: str = 'calibration/screen_calibration.yaml') -> bool:
        """Carrega calibração salva"""
        try:
            with open(filepath, 'r') as f:
                data = yaml.safe_load(f)
            
            if data:
                self.transformation_matrix = np.array(data['transformation_matrix']) if data.get('transformation_matrix') else None
                self.calibration_points = data.get('calibration_points', self.calibration_points)
                self.collected_data = data.get('collected_data', [])
                self.calibration_stats = data.get('calibration_stats', {})
                self.use_polynomial_model = data.get('use_polynomial_model', False)
                self.calibration_complete = True
                
                # Tentar carregar modelos sklearn
                if self.use_polynomial_model:
                    try:
                        import pickle
                        with open('calibration/models.pkl', 'rb') as f:
                            models = pickle.load(f)
                        self.model_x = models['model_x']
                        self.model_y = models['model_y']
                        self.poly_features = models['poly_features']
                    except:
                        # Re-treinar se necessário
                        if len(self.collected_data) >= 4:
                            self._train_advanced_model()
                
                return True
                
        except Exception:
            return False
    
    def reset_calibration(self) -> bool:
        """Reset da calibração"""
        self.transformation_matrix = None
        self.model_x = None
        self.model_y = None
        self.model = None
        self.collected_data = []
        self.calibration_data = []
        self.calibration_complete = False
        self.current_point_idx = 0
        self.gaze_samples.clear()
        return True
    
    def train_model(self) -> bool:
        """Treina modelo"""
        if len(self.calibration_data) < 5:
            return False
        
        try:
            # Tentar sklearn primeiro
            from sklearn.ensemble import RandomForestRegressor
            
            X = []
            y = []
            
            for point_data in self.calibration_data:
                screen_x = point_data['screen_x']
                screen_y = point_data['screen_y']
                
                for sample in point_data['samples']:
                    X.append([sample['yaw'], sample['pitch']])
                    y.append([screen_x, screen_y])
            
            X = np.array(X)
            y = np.array(y)
            
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
            self.model.fit(X, y)
            
            return True
            
        except ImportError:
            return False
        except Exception:
            return False
    
    def get_calibration_quality(self) -> Dict:
        """Retorna qualidade da calibração"""
        if not self.calibration_complete or not self.collected_data:
            return {
                'status': 'not_calibrated',
                'points_collected': 0,
                'quality_score': 0
            }
        
        # Calcular métricas
        stds = [d.get('std', (0, 0)) for d in self.collected_data]
        avg_std = np.mean([np.mean(s) for s in stds if s])
        
        # Score baseado no erro e desvio
        quality_score = 100.0
        
        if self.calibration_stats:
            avg_error = self.calibration_stats.get('avg_error', 0)
            if avg_error > 0:
                # Penalizar por erro alto
                quality_score -= min(50, avg_error / 5)
        
        # Penalizar por alta variância
        quality_score -= min(30, avg_std * 500)
        
        # Penalizar por poucos pontos
        if len(self.collected_data) < 9:
            quality_score -= (9 - len(self.collected_data)) * 5
        
        quality_score = max(0, min(100, quality_score))
        
        # Determinar qualidade textual
        if quality_score >= 80:
            quality = 'excellent'
        elif quality_score >= 60:
            quality = 'good'
        elif quality_score >= 40:
            quality = 'regular'
        else:
            quality = 'poor'
        
        return {
            'status': 'calibrated',
            'points_collected': len(self.collected_data),
            'quality_score': quality_score,
            'quality': quality,
            'avg_error': self.calibration_stats.get('avg_error', 0) if self.calibration_stats else 0,
            'std_error': self.calibration_stats.get('std_error', 0) if self.calibration_stats else 0
        }
    
    def get_calibration_report(self) -> Dict:
        """Gera relatório de calibração"""
        quality = self.get_calibration_quality()
        
        return {
            'status': quality['status'],
            'points_collected': quality['points_collected'],
            'quality_score': quality['quality_score'],
            'quality': quality.get('quality', 'unknown'),
            'avg_error': quality.get('avg_error', 0),
            'std_error': quality.get('std_error', 0),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }


class ScreenCalibration(AdvancedScreenCalibration):
    """Alias para compatibilidade total"""
    pass


class AdaptiveCalibration:
    """Sistema de calibração adaptativa"""
    
    def __init__(self, base_calibration: ScreenCalibration):
        self.base_calibration = base_calibration
        self.refinement_data = deque(maxlen=100)
        self.correction_matrix = np.eye(3)
        self.adaptation_enabled = True
    
    def add_refinement_point(self, predicted: Tuple[int, int], 
                            actual: Tuple[int, int], confidence: float = 1.0):
        """Adiciona ponto de refinamento"""
        if not self.adaptation_enabled:
            return
        
        self.refinement_data.append({
            'predicted': predicted,
            'actual': actual,
            'confidence': confidence,
            'timestamp': time.time()
        })
        
        if len(self.refinement_data) % 10 == 0:
            self._update_correction_matrix()
    
    def _update_correction_matrix(self):
        """Atualiza matriz de correção"""
        if len(self.refinement_data) < 5:
            return
        
        current_time = time.time()
        weighted_data = []
        
        for data in self.refinement_data:
            age = current_time - data['timestamp']
            weight = np.exp(-age / 3600)
            weighted_data.append((data, weight * data['confidence']))
        
        total_weight = sum(w for _, w in weighted_data)
        if total_weight > 2.0:
            src_points = np.array([d['predicted'] for d, _ in weighted_data], 
                                 dtype=np.float32)
            dst_points = np.array([d['actual'] for d, _ in weighted_data], 
                                 dtype=np.float32)
            
            correction, _ = cv2.findHomography(src_points, dst_points, 
                                             cv2.RANSAC, 5.0)
            
            if correction is not None:
                alpha = 0.3
                self.correction_matrix = (1 - alpha) * self.correction_matrix + alpha * correction
    
    def map_gaze_to_screen(self, gaze_yaw: float, gaze_pitch: float) -> Optional[Tuple[int, int]]:
        """Mapeia com correção adaptativa"""
        base_point = self.base_calibration.map_gaze_to_screen(gaze_yaw, gaze_pitch)
        
        if base_point is None:
            return None
        
        try:
            point = np.array([base_point[0], base_point[1], 1])
            corrected = self.correction_matrix @ point
            
            x = int(corrected[0] / corrected[2])
            y = int(corrected[1] / corrected[2])
            
            x = np.clip(x, 0, self.base_calibration.screen_width - 1)
            y = np.clip(y, 0, self.base_calibration.screen_height - 1)
            
            return (x, y)
            
        except Exception:
            return base_point


if __name__ == "__main__":
    
    # Criar calibrador
    cal = AdvancedScreenCalibration(1920, 1080)
    
    # Testar o método
    quality = cal.get_calibration_quality()
    
    # Testar carregamento
    if cal.load_calibration():
        quality = cal.get_calibration_quality()
