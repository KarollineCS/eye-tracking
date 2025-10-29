import cv2
import numpy as np
import time
from typing import Tuple, List, Dict, Optional
import yaml
import threading
from collections import deque
import os

class ScreenCalibration:
    """Sistema de calibração screen-to-gaze com múltiplos pontos"""
    
    def __init__(self, screen_width: int, screen_height: int):
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        # Pontos de calibração (9 pontos)
        self.calibration_points = self._generate_calibration_points()
        self.current_point_idx = 0
        
        # Dados coletados durante calibração
        self.collected_data = []
        self.gaze_samples = deque(maxlen=30)  # Coletar 30 amostras por ponto
        
        # Matriz de transformação
        self.transformation_matrix = None
        self.calibration_complete = False
        
        # Parâmetros de calibração
        self.point_display_time = 2.0  # segundos por ponto
        self.point_radius = 20
        self.collection_started = False
        
        # Monitor virtual para calibração
        self.monitor_mm = (500, 300)  # Dimensões físicas do monitor em mm
        self.monitor_distance = 600  # Distância do usuário ao monitor em mm

    def is_calibrated(self):
        """Verifica se o sistema está calibrado"""
        return hasattr(self, 'model') and self.model is not None
    
    def train_model(self):
        """Treina o modelo de calibração"""
        if len(self.calibration_data) < 5:
            print("❌ Dados insuficientes para treinar")
            return False
        
        # Preparar dados
        X = []  # features (yaw, pitch)
        y = []  # targets (screen_x, screen_y)
        
        for point_data in self.calibration_data:
            screen_x = point_data['screen_x']
            screen_y = point_data['screen_y']
            
            for sample in point_data['samples']:
                X.append([sample['yaw'], sample['pitch']])
                y.append([screen_x, screen_y])
        
        X = np.array(X)
        y = np.array(y)
        
        # Treinar modelo (pode ser SVR, Ridge, etc)
        from sklearn.ensemble import RandomForestRegressor
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X, y)
        
        print(f"✅ Modelo treinado com {len(X)} amostras")
        return True
        
    def _generate_calibration_points(self) -> List[Tuple[int, int]]:
        """Gera os 9 pontos de calibração em grade 3x3"""
        points = []
        margins = 0.1  # 10% de margem das bordas
        
        for y in [margins, 0.5, 1 - margins]:
            for x in [margins, 0.5, 1 - margins]:
                px = int(x * self.screen_width)
                py = int(y * self.screen_height)
                points.append((px, py))
        
        return points
    
    def start_calibration(self):
        """Inicia o processo de calibração"""
        print("\n🎯 INICIANDO CALIBRAÇÃO DE TELA")
        print("Instruções:")
        print("1. Olhe fixamente para cada ponto vermelho")
        print("2. Mantenha o olhar até o ponto ficar verde")
        print("3. A calibração será automática\n")
        
        self.current_point_idx = 0
        self.collected_data = []
        self.calibration_complete = False
        time.sleep(2)
        
        # Criar janela de calibração em tela cheia
        cv2.namedWindow('Calibration', cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty('Calibration', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        self.collection_started = True
        
    def display_calibration_point(self) -> np.ndarray:
        """Exibe o ponto de calibração atual"""
        # Criar tela preta
        screen = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)
        
        if self.current_point_idx < len(self.calibration_points):
            point = self.calibration_points[self.current_point_idx]
            
            # Cor do ponto baseada no progresso da coleta
            samples_collected = len(self.gaze_samples)
            if samples_collected < 15:
                color = (0, 0, 255)  # Vermelho
            elif samples_collected < 25:
                color = (0, 165, 255)  # Laranja
            else:
                color = (0, 255, 0)  # Verde
            
            # Desenhar ponto de calibração
            cv2.circle(screen, point, self.point_radius, color, -1)
            cv2.circle(screen, point, self.point_radius + 5, color, 2)
            
            # Crosshair no centro do ponto
            cv2.line(screen, 
                    (point[0] - 10, point[1]), 
                    (point[0] + 10, point[1]), 
                    (255, 255, 255), 1)
            cv2.line(screen, 
                    (point[0], point[1] - 10), 
                    (point[0], point[1] + 10), 
                    (255, 255, 255), 1)
            
            # Mostrar progresso
            progress_text = f"Ponto {self.current_point_idx + 1}/{len(self.calibration_points)}"
            cv2.putText(screen, progress_text, (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Barra de progresso
            progress_width = int((samples_collected / 30) * 200)
            cv2.rectangle(screen, (50, 70), (250, 90), (100, 100, 100), -1)
            cv2.rectangle(screen, (50, 70), (50 + progress_width, 90), color, -1)
        
        return screen
    
    def collect_gaze_sample(self, gaze_data: Dict):
        """Coleta uma amostra de gaze para o ponto atual"""
        if gaze_data and self.collection_started:
            self.gaze_samples.append({
                'gaze': (gaze_data.get('yaw', 0), gaze_data.get('pitch', 0)),
                'timestamp': time.time()
            })
            
            # Verificar se coletou amostras suficientes
            if len(self.gaze_samples) >= 30:
                self._process_calibration_point()
    
    def _process_calibration_point(self):
        """Processa os dados coletados para o ponto atual"""
        if len(self.gaze_samples) > 0:
            # Calcular média do gaze para este ponto
            gazes = [s['gaze'] for s in self.gaze_samples]
            avg_yaw = np.mean([g[0] for g in gazes])
            avg_pitch = np.mean([g[1] for g in gazes])
            std_yaw = np.std([g[0] for g in gazes])
            std_pitch = np.std([g[1] for g in gazes])
            
            # Armazenar dados
            screen_point = self.calibration_points[self.current_point_idx]
            self.collected_data.append({
                'screen_point': screen_point,
                'gaze_point': (avg_yaw, avg_pitch),
                'std': (std_yaw, std_pitch),
                'samples': len(self.gaze_samples)
            })
            
            print(f"✓ Ponto {self.current_point_idx + 1} calibrado: "
                  f"Tela({screen_point}) -> Gaze({avg_yaw:.3f}, {avg_pitch:.3f})")
            
            # Limpar e avançar
            self.gaze_samples.clear()
            self.current_point_idx += 1
            self.collection_started = False
            
            # Delay antes do próximo ponto
            time.sleep(0.5)
            self.collection_started = True
            
            # Verificar se terminou
            if self.current_point_idx >= len(self.calibration_points):
                self._finalize_calibration()
    
    def _finalize_calibration(self):
        """Finaliza a calibração e calcula a matriz de transformação"""
        print("\n📊 Processando dados de calibração...")
        
        if len(self.collected_data) >= 4:  # Mínimo 4 pontos para homografia
            # Preparar pontos para homografia
            src_points = np.array([d['gaze_point'] for d in self.collected_data], 
                                 dtype=np.float32)
            dst_points = np.array([d['screen_point'] for d in self.collected_data], 
                                 dtype=np.float32)
            
            # Calcular matriz de transformação (homografia)
            self.transformation_matrix, _ = cv2.findHomography(src_points, dst_points, 
                                                              cv2.RANSAC, 5.0)
            
            # Calcular erro médio
            errors = []
            for data in self.collected_data:
                gaze = np.array([data['gaze_point'][0], data['gaze_point'][1], 1])
                predicted = self.transformation_matrix @ gaze
                predicted = predicted[:2] / predicted[2]
                
                actual = data['screen_point']
                error = np.linalg.norm(predicted - actual)
                errors.append(error)
            
            avg_error = np.mean(errors)
            print(f"✅ Calibração concluída!")
            print(f"📏 Erro médio: {avg_error:.1f} pixels")
            print(f"📊 Desvio padrão: {np.std(errors):.1f} pixels")
            
            # Salvar calibração
            self._save_calibration()
            self.calibration_complete = True
            
            # Análise de qualidade
            if avg_error < 50:
                print("🎯 Qualidade: EXCELENTE")
            elif avg_error < 100:
                print("✓ Qualidade: BOA")
            else:
                print("⚠️ Qualidade: REGULAR - Considere recalibrar")
        else:
            print("❌ Dados insuficientes para calibração")
            self.calibration_complete = False
        
        cv2.destroyWindow('Calibration')
    
    def map_gaze_to_screen(self, gaze_yaw: float, gaze_pitch: float) -> Optional[Tuple[int, int]]:
        """Mapeia coordenadas de gaze para posição na tela"""
        if self.transformation_matrix is None:
            return None
        
        try:
            # Aplicar transformação
            gaze_point = np.array([gaze_yaw, gaze_pitch, 1])
            screen_point = self.transformation_matrix @ gaze_point
            
            # Normalizar coordenadas homogêneas
            if screen_point[2] != 0:
                x = int(screen_point[0] / screen_point[2])
                y = int(screen_point[1] / screen_point[2])
                
                # Limitar às dimensões da tela
                x = np.clip(x, 0, self.screen_width - 1)
                y = np.clip(y, 0, self.screen_height - 1)
                
                return (x, y)
        except Exception as e:
            print(f"Erro no mapeamento: {e}")
        
        return None
    
    def _save_calibration(self):
        """Salva os dados de calibração em arquivo"""
        calibration_data = {
            'screen_dimensions': [self.screen_width, self.screen_height],
            'calibration_points': self.calibration_points,
            'collected_data': [
                {
                    'screen_point': d['screen_point'],
                    'gaze_point': d['gaze_point'],
                    'std': d['std']
                } for d in self.collected_data
            ],
            'transformation_matrix': self.transformation_matrix.tolist() if self.transformation_matrix is not None else None,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'monitor_info': {
                'size_mm': self.monitor_mm,
                'distance_mm': self.monitor_distance
            }
        }

        # Criar diretório se não existir
        os.makedirs('calibration', exist_ok=True)

        with open('calibration/screen_calibration.yaml', 'w') as f:
            yaml.dump(calibration_data, f)

        print(f"💾 Calibração salva em 'calibration/screen_calibration.yaml'")
    
    def load_calibration(self, filepath: str = 'calibration/screen_calibration.yaml') -> bool:
        """Carrega calibração salva"""
        try:
            with open(filepath, 'r') as f:
                data = yaml.safe_load(f)
            
            self.transformation_matrix = np.array(data['transformation_matrix'])
            self.calibration_points = data['calibration_points']
            self.collected_data = data['collected_data']
            self.calibration_complete = True
            
            print(f"✅ Calibração carregada de {filepath}")
            print(f"📅 Data da calibração: {data['timestamp']}")
            return True
            
        except Exception as e:
            print(f"❌ Erro ao carregar calibração: {e}")
            return False
    
    def get_calibration_quality(self) -> Dict:
        """Retorna métricas de qualidade da calibração"""
        if not self.calibration_complete or not self.collected_data:
            return {'status': 'not_calibrated'}
        
        # Calcular métricas
        stds = [d['std'] for d in self.collected_data]
        avg_std_yaw = np.mean([s[0] for s in stds])
        avg_std_pitch = np.mean([s[1] for s in stds])
        
        # Calcular cobertura da tela
        screen_points = [d['screen_point'] for d in self.collected_data]
        x_coverage = (max(p[0] for p in screen_points) - 
                     min(p[0] for p in screen_points)) / self.screen_width
        y_coverage = (max(p[1] for p in screen_points) - 
                     min(p[1] for p in screen_points)) / self.screen_height
        
        return {
            'status': 'calibrated',
            'points_collected': len(self.collected_data),
            'avg_std_yaw': avg_std_yaw,
            'avg_std_pitch': avg_std_pitch,
            'screen_coverage_x': x_coverage,
            'screen_coverage_y': y_coverage,
            'quality_score': self._calculate_quality_score()
        }
    
    def _calculate_quality_score(self) -> float:
        """Calcula score de qualidade da calibração (0-100)"""
        if not self.collected_data:
            return 0.0
        
        score = 100.0
        
        # Penalizar por alta variância
        stds = [d['std'] for d in self.collected_data]
        avg_std = np.mean([s[0] + s[1] for s in stds])
        score -= min(avg_std * 50, 30)  # Máximo -30 pontos
        
        # Penalizar por poucos pontos
        if len(self.collected_data) < 9:
            score -= (9 - len(self.collected_data)) * 5
        
        return max(0, min(100, score))


class AdaptiveCalibration:
    """Sistema de calibração adaptativa que melhora com o uso"""
    
    def __init__(self, base_calibration: ScreenCalibration):
        self.base_calibration = base_calibration
        self.refinement_data = deque(maxlen=100)  # Últimas 100 correções
        self.correction_matrix = np.eye(3)
        
    def add_correction(self, predicted_point: Tuple[int, int], 
                       actual_point: Tuple[int, int], confidence: float = 1.0):
        """Adiciona uma correção baseada em feedback do usuário"""
        self.refinement_data.append({
            'predicted': predicted_point,
            'actual': actual_point,
            'confidence': confidence,
            'timestamp': time.time()
        })
        
        # Recalcular correção se tiver dados suficientes
        if len(self.refinement_data) >= 10:
            self._update_correction_matrix()
    
    def _update_correction_matrix(self):
        """Atualiza matriz de correção com base nos dados recentes"""
        if len(self.refinement_data) < 4:
            return
        
        # Pegar dados recentes com peso baseado no tempo
        current_time = time.time()
        weighted_data = []
        
        for data in self.refinement_data:
            age = current_time - data['timestamp']
            weight = np.exp(-age / 3600)  # Decaimento exponencial (1 hora)
            weighted_data.append((data, weight * data['confidence']))
        
        # Calcular correção se tiver peso suficiente
        total_weight = sum(w for _, w in weighted_data)
        if total_weight > 2.0:
            src_points = np.array([d['predicted'] for d, _ in weighted_data], 
                                 dtype=np.float32)
            dst_points = np.array([d['actual'] for d, _ in weighted_data], 
                                 dtype=np.float32)
            
            # Calcular correção incremental
            correction, _ = cv2.findHomography(src_points, dst_points, 
                                             cv2.RANSAC, 5.0)
            
            if correction is not None:
                # Aplicar correção suave (blend com matriz anterior)
                alpha = 0.3  # Taxa de aprendizado
                self.correction_matrix = (1 - alpha) * self.correction_matrix + alpha * correction
    
    def map_gaze_to_screen(self, gaze_yaw: float, gaze_pitch: float) -> Optional[Tuple[int, int]]:
        """Mapeia gaze com correção adaptativa"""
        # Primeiro, usar calibração base
        base_point = self.base_calibration.map_gaze_to_screen(gaze_yaw, gaze_pitch)
        
        if base_point is None:
            return None
        
        # Aplicar correção adaptativa
        try:
            point = np.array([base_point[0], base_point[1], 1])
            corrected = self.correction_matrix @ point
            
            x = int(corrected[0] / corrected[2])
            y = int(corrected[1] / corrected[2])
            
            # Limitar às dimensões da tela
            x = np.clip(x, 0, self.base_calibration.screen_width - 1)
            y = np.clip(y, 0, self.base_calibration.screen_height - 1)
            
            return (x, y)
            
        except Exception:
            return base_point
