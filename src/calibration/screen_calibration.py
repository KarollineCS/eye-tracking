import cv2
import numpy as np
import time
from typing import Tuple, List, Dict, Optional
import yaml
import os
from collections import deque
import threading
import keyboard


class ManualScreenCalibration:
    """
    Sistema de calibração manual onde o usuário confirma quando está olhando
    Baseado no MonitorTracking.py
    """
    
    def __init__(self, screen_width: int, screen_height: int):
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        # Pontos de calibração (9 pontos em grade 3x3)
        self.calibration_points = self._generate_calibration_points()
        self.current_point_idx = 0
        
        # Dados coletados
        self.collected_data = []
        self.current_samples = []  # Amostras do ponto atual
        self.is_collecting = False
        self.collection_thread = None
        
        # Estado da calibração
        self.transformation_matrix = None
        self.calibration_complete = False
        
        # Parâmetros ajustáveis
        self.min_samples_per_point = 20  # Mínimo de amostras por ponto
        self.max_samples_per_point = 50  # Máximo de amostras
        self.point_radius = 25
        self.collection_rate = 30  # Hz de coleta quando pressionado
        
        # Feedback visual
        self.last_gaze_data = None
        self.space_pressed = False
        self.enter_pressed = False
        self.calibration_aborted = False
        
        # Estatísticas
        self.point_start_time = 0
        self.total_calibration_time = 0
        self.calibration_start_time = 0
        
        # Monitor físico
        self.monitor_mm = (500, 300)
        self.monitor_distance = 600
        
    
    def _generate_calibration_points(self) -> List[List[int]]:
        """Gera pontos de calibração otimizados"""
        points = []
        margins = 0.15
        
        # Ordem otimizada: centro primeiro, depois cantos, depois meios
        positions = [
            (0.5, 0.5),    # Centro
            (margins, margins),         # Canto superior esquerdo
            (1-margins, margins),       # Canto superior direito
            (margins, 1-margins),       # Canto inferior esquerdo
            (1-margins, 1-margins),     # Canto inferior direito
            (0.5, margins),             # Meio superior
            (0.5, 1-margins),           # Meio inferior
            (margins, 0.5),             # Meio esquerdo
            (1-margins, 0.5),           # Meio direito
        ]
        
        for x, y in positions:
            px = int(x * self.screen_width)
            py = int(y * self.screen_height)
            points.append([px, py])
        
        return points
    
    def start_calibration(self) -> bool:
        
        cv2.waitKey(0)
        
        # Reset do estado
        self.current_point_idx = 0
        self.collected_data = []
        self.calibration_complete = False
        self.calibration_aborted = False
        self.calibration_start_time = time.time()
        
        # Criar janela fullscreen
        cv2.namedWindow('Manual Calibration', cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty('Manual Calibration', cv2.WND_PROP_FULLSCREEN, 
                            cv2.WINDOW_FULLSCREEN)
        
        # Iniciar thread de monitoramento de teclas
        self.start_keyboard_monitoring()
        
        return True
    
    def start_keyboard_monitoring(self):
        """Inicia monitoramento de teclado em thread separada"""
        def monitor_keys():
            while not self.calibration_complete and not self.calibration_aborted:
                try:
                    self.space_pressed = keyboard.is_pressed('space')
                    
                    if keyboard.is_pressed('enter'):
                        self.enter_pressed = True
                        time.sleep(0.2)  # Debounce
                    
                    if keyboard.is_pressed('escape'):
                        self.calibration_aborted = True
                        break
                        
                    time.sleep(0.01)  # 100Hz de verificação
                    
                except Exception:
                    break
        
        self.keyboard_thread = threading.Thread(target=monitor_keys, daemon=True)
        self.keyboard_thread.start()
    
    def display_calibration_point(self) -> np.ndarray:
        """Exibe ponto de calibração com feedback visual rico"""
        # Tela base
        screen = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)
        
        if self.current_point_idx >= len(self.calibration_points):
            # Tela de conclusão
            cv2.putText(screen, "Calibracao Concluida!", 
                       (self.screen_width//2 - 200, self.screen_height//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
            return screen
        
        point = self.calibration_points[self.current_point_idx]
        samples_count = len(self.current_samples)
        
        # Cor do ponto baseada no estado e progresso
        if self.space_pressed:
            # Coletando ativamente
            if samples_count < self.min_samples_per_point:
                color = (0, 100, 255)  # Vermelho escuro -> coletando
                pulse = int(20 * np.sin(time.time() * 10))  # Pulsação rápida
            else:
                color = (0, 255, 100)  # Verde claro -> suficiente
                pulse = 5
        else:
            # Esperando usuário
            if samples_count == 0:
                color = (0, 0, 255)  # Vermelho
                pulse = int(10 * np.sin(time.time() * 2))  # Pulsação lenta
            elif samples_count < self.min_samples_per_point:
                color = (0, 165, 255)  # Laranja
                pulse = 5
            else:
                color = (0, 255, 0)  # Verde
                pulse = 0
        
        # Círculo externo pulsante
        cv2.circle(screen, point, self.point_radius + 10 + pulse, 
                  (color[0]//3, color[1]//3, color[2]//3), -1)
        
        # Círculo principal
        cv2.circle(screen, point, self.point_radius, color, -1)
        
        # Círculo interno (alvo)
        cv2.circle(screen, point, 5, (255, 255, 255), -1)
        
        # Crosshair
        cv2.line(screen, (point[0]-15, point[1]), (point[0]+15, point[1]), 
                (255, 255, 255), 1)
        cv2.line(screen, (point[0], point[1]-15), (point[0], point[1]+15), 
                (255, 255, 255), 1)
        
        # === PAINEL DE INFORMAÇÕES ===
        info_y = 50
        
        # Progresso geral
        cv2.putText(screen, f"Ponto {self.current_point_idx + 1}/{len(self.calibration_points)}", 
                   (50, info_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Status de coleta
        status_text = "COLETANDO..." if self.space_pressed else "Pressione ESPACO"
        status_color = (0, 255, 0) if self.space_pressed else (255, 255, 255)
        cv2.putText(screen, status_text, (50, info_y + 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        
        # Contador de amostras com barra de progresso
        cv2.putText(screen, f"Amostras: {samples_count}/{self.min_samples_per_point}", 
                   (50, info_y + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Barra de progresso
        bar_width = 300
        bar_height = 20
        bar_x = 50
        bar_y = info_y + 100
        
        # Fundo da barra
        cv2.rectangle(screen, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height),
                     (100, 100, 100), -1)
        
        # Progresso
        progress = min(1.0, samples_count / self.min_samples_per_point)
        progress_width = int(bar_width * progress)
        progress_color = (0, 255, 0) if progress >= 1.0 else (0, 165, 255)
        cv2.rectangle(screen, (bar_x, bar_y), (bar_x + progress_width, bar_y + bar_height),
                     progress_color, -1)
        
        # === INSTRUÇÕES CONTEXTUAIS ===
        instruction_y = self.screen_height - 150
        
        if samples_count == 0:
            instruction = "Olhe para o ponto e SEGURE ESPACO"
        elif samples_count < self.min_samples_per_point:
            instruction = f"Continue segurando... ({self.min_samples_per_point - samples_count} restantes)"
        else:
            instruction = "Otimo! Pressione ENTER ou continue para mais precisao"
        
        cv2.putText(screen, instruction, (50, instruction_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)
        
        # Atalhos
        cv2.putText(screen, "[ESPACO] Coletar  [ENTER] Proximo  [ESC] Cancelar",
                   (50, instruction_y + 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 1)
        
        # === VISUALIZAÇÃO DO GAZE ATUAL (se disponível) ===
        if self.last_gaze_data and self.space_pressed:
            # Mostrar onde o sistema está detectando o olhar
            gaze_x = self.screen_width - 350
            gaze_y = 100
            
            cv2.putText(screen, "Gaze Detectado:", (gaze_x, gaze_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 1)
            
            yaw = self.last_gaze_data.get('yaw', 0)
            pitch = self.last_gaze_data.get('pitch', 0)
            
            cv2.putText(screen, f"Yaw: {yaw:.3f}", (gaze_x, gaze_y + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            cv2.putText(screen, f"Pitch: {pitch:.3f}", (gaze_x, gaze_y + 55),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
            # Indicador de estabilidade (baseado no desvio padrão das últimas amostras)
            if len(self.current_samples) > 5:
                recent = self.current_samples[-5:]
                std_yaw = np.std([s['yaw'] for s in recent])
                std_pitch = np.std([s['pitch'] for s in recent])
                stability = 1.0 - min(1.0, (std_yaw + std_pitch) * 10)
                
                stability_color = (0, int(255 * stability), int(255 * (1-stability)))
                cv2.putText(screen, f"Estabilidade: {stability*100:.0f}%", 
                           (gaze_x, gaze_y + 80),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, stability_color, 1)
        
        return screen
    
    def collect_gaze_sample(self, gaze_data: Dict) -> bool:
        """
        Coleta amostra apenas quando ESPAÇO está pressionado
        Retorna True se deve avançar para próximo ponto
        """
        if not gaze_data:
            return False
        
        self.last_gaze_data = gaze_data
        
        # Coletar apenas se ESPAÇO pressionado
        if self.space_pressed and self.current_point_idx < len(self.calibration_points):
            # Adicionar amostra com timestamp
            self.current_samples.append({
                'yaw': gaze_data.get('yaw', 0),
                'pitch': gaze_data.get('pitch', 0),
                'timestamp': time.time()
            })
            
            # Limitar número máximo de amostras
            if len(self.current_samples) > self.max_samples_per_point:
                self.current_samples = self.current_samples[-self.max_samples_per_point:]
        
        # Verificar se deve avançar (ENTER pressionado ou ESC)
        if self.enter_pressed and len(self.current_samples) >= self.min_samples_per_point:
            self.enter_pressed = False
            self._save_current_point()
            return True
        
        if self.calibration_aborted:
            return False
        
        return False
    
    def _save_current_point(self):
        """Salva dados do ponto atual e avança"""
        if len(self.current_samples) > 0:
            # Calcular estatísticas
            yaws = [s['yaw'] for s in self.current_samples]
            pitches = [s['pitch'] for s in self.current_samples]
            
            avg_yaw = float(np.mean(yaws))
            avg_pitch = float(np.mean(pitches))
            std_yaw = float(np.std(yaws))
            std_pitch = float(np.std(pitches))
            
            # Salvar dados
            point = self.calibration_points[self.current_point_idx]
            self.collected_data.append({
                'screen_point': list(point),  # Converter tupla para lista
                'gaze_point': [avg_yaw, avg_pitch],
                'std': [std_yaw, std_pitch],
                'samples_count': len(self.current_samples),
                'collection_time': time.time() - self.point_start_time
            })
        
        # Limpar e avançar
        self.current_samples = []
        self.current_point_idx += 1
        self.point_start_time = time.time()
        
        # Verificar se terminou
        if self.current_point_idx >= len(self.calibration_points):
            self.finalize_calibration()
    
    def finalize_calibration(self):
        """Finaliza calibração e calcula transformação"""
        self.calibration_complete = True
        self.total_calibration_time = time.time() - self.calibration_start_time
        
        if len(self.collected_data) >= 4:
            try:
                # Preparar pontos
                src_points = np.array([d['gaze_point'] for d in self.collected_data], 
                                     dtype=np.float32)
                dst_points = np.array([d['screen_point'] for d in self.collected_data], 
                                     dtype=np.float32)
                
                # Calcular homografia
                self.transformation_matrix, mask = cv2.findHomography(
                    src_points, dst_points, cv2.RANSAC, 5.0
                )
                
                # Calcular erro de reprojeção
                errors = []
                for i, data in enumerate(self.collected_data):
                    gaze = np.array([data['gaze_point'][0], data['gaze_point'][1], 1])
                    predicted = self.transformation_matrix @ gaze
                    predicted = predicted[:2] / predicted[2]
                    
                    actual = np.array(data['screen_point'])
                    error = np.linalg.norm(predicted - actual)
                    errors.append(error)
                    
                    # Mostrar erro por ponto
                    quality = "✅" if error < 50 else "⚠️" if error < 100 else "❌"
                    print(f"  Ponto {i+1}: Erro = {error:.1f}px {quality}")
                
                avg_error = np.mean(errors)
                std_error = np.std(errors)
                
                # Salvar calibração
                self._save_calibration()
                
            except Exception:
                self.calibration_complete = False
        else:
            self.calibration_complete = False
        
        # Fechar janela
        cv2.destroyWindow('Manual Calibration')
    
    def _save_calibration(self):
        """Salva calibração em arquivo"""
        try:
            calibration_data = {
                'screen_dimensions': [self.screen_width, self.screen_height],
                'calibration_points': self.calibration_points,
                'collected_data': self.collected_data,
                'transformation_matrix': self.transformation_matrix.tolist() if self.transformation_matrix is not None else None,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'calibration_time': self.total_calibration_time,
                'method': 'manual_with_confirmation',
                'monitor_info': {
                    'size_mm': list(self.monitor_mm),
                    'distance_mm': self.monitor_distance
                }
            }
            
            os.makedirs('calibration', exist_ok=True)
            
            # Salvar em YAML
            with open('calibration/manual_calibration.yaml', 'w') as f:
                yaml.dump(calibration_data, f, default_flow_style=False)
            
            
        except Exception as e:
            print(f"❌ Erro ao salvar calibração: {e}")
    
    def load_calibration(self, filepath: str = 'calibration/manual_calibration.yaml') -> bool:
        """Carrega calibração salva"""
        try:
            if not os.path.exists(filepath):
                # Tentar arquivo padrão antigo
                old_file = 'calibration/screen_calibration.yaml'
                if os.path.exists(old_file):
                    filepath = old_file
                else:
                    return False
            
            with open(filepath, 'r') as f:
                data = yaml.safe_load(f)
            
            if data and 'transformation_matrix' in data:
                self.transformation_matrix = np.array(data['transformation_matrix'])
                self.calibration_points = data.get('calibration_points', self.calibration_points)
                self.collected_data = data.get('collected_data', [])
                self.calibration_complete = True
                
                return True
                
        except Exception:
            pass
        
        return False
    
    def map_gaze_to_screen(self, gaze_yaw: float, gaze_pitch: float) -> Optional[Tuple[int, int]]:
        """Mapeia gaze para coordenadas da tela"""
        if self.transformation_matrix is None:
            return None
        
        try:
            gaze_point = np.array([gaze_yaw, gaze_pitch, 1])
            screen_point = self.transformation_matrix @ gaze_point
            
            if screen_point[2] != 0:
                x = int(screen_point[0] / screen_point[2])
                y = int(screen_point[1] / screen_point[2])
                
                # Limitar às dimensões da tela
                x = np.clip(x, 0, self.screen_width - 1)
                y = np.clip(y, 0, self.screen_height - 1)
                
                return (x, y)
                
        except:
            pass
        
        return None
    
    def is_calibrated(self) -> bool:
        """Verifica se está calibrado"""
        return self.calibration_complete and self.transformation_matrix is not None
    
    def reset_calibration(self) -> bool:
        """Reset da calibração"""
        self.transformation_matrix = None
        self.collected_data = []
        self.calibration_complete = False
        self.current_point_idx = 0
        self.current_samples = []
        
        return True
    
    def get_calibration_quality(self) -> Dict:
        """Retorna métricas de qualidade"""
        if not self.calibration_complete:
            return {'status': 'not_calibrated'}
        
        total_samples = sum(d.get('samples_count', 0) for d in self.collected_data)
        avg_std = np.mean([np.mean(d.get('std', [0, 0])) for d in self.collected_data])
        
        return {
            'status': 'calibrated',
            'points_collected': len(self.collected_data),
            'total_samples': total_samples,
            'avg_std': avg_std,
            'quality': 'excellent' if avg_std < 0.01 else 'good' if avg_std < 0.02 else 'regular'
        }


# Para compatibilidade com código existente
class ScreenCalibration(ManualScreenCalibration):
    """Alias para compatibilidade"""
    pass


class AdaptiveCalibration:
    """Calibração adaptativa que melhora com o tempo"""
    
    def __init__(self, base_calibration):
        self.base_calibration = base_calibration
        self.refinement_data = deque(maxlen=100)
        self.adaptation_enabled = True
    
    def map_gaze_to_screen(self, gaze_yaw: float, gaze_pitch: float) -> Optional[Tuple[int, int]]:
        """Mapeia com refinamentos adaptativos"""
        return self.base_calibration.map_gaze_to_screen(gaze_yaw, gaze_pitch)


if __name__ == "__main__":
    
    # Criar calibrador
    calibrator = ManualScreenCalibration(1920, 1080)
    
    # Verificar se já está calibrado
    if calibrator.load_calibration():
        
        quality = calibrator.get_calibration_quality()
    else:
        print("❌ Nenhuma calibração encontrada")
