import cv2
import numpy as np
import time
import math
from typing import Dict, List, Tuple, Optional

class VisualizationManager:
    """Gerenciador de visualização otimizado"""
    
    def __init__(self, config):
        self.config = config
        self.show_landmarks = True
        self.show_iris_detection = True
        self.show_gaze_vectors = True
        self.show_attention_zones = False
        self.show_sa_metrics = False
        
        # Cores definidas
        self.colors = {
            'face_box': (0, 255, 0),
            'left_eye': (0, 255, 0),
            'right_eye': (255, 0, 0),
            'iris_left': (0, 255, 0),
            'iris_right': (255, 0, 0),
            'gaze_vector': (0, 255, 255),
            'calibration_point': (0, 0, 255),
            'predicted_gaze': (255, 0, 255),
            'attention_zone': (255, 255, 0),
            'sa_good': (0, 255, 0),
            'sa_moderate': (0, 165, 255),
            'sa_poor': (0, 0, 255)
        }
        
        # Índices dos landmarks dos olhos
        self.LEFT_EYE_POINTS = list(range(36, 42))
        self.RIGHT_EYE_POINTS = list(range(42, 48))
        
        print("🎨 Gerenciador de Visualização inicializado")
    
    def render_frame(self, frame, results, is_calibrating, calibration_system, 
                    screen_calibration, current_fps):
        """Renderiza frame principal com todas as visualizações"""
        display_frame = frame.copy()
        
        # Desenha detecções faciais
        self.draw_face_detection(display_frame, results)
        
        # Desenha landmarks dos olhos
        if self.show_landmarks and results.get('landmarks') is not None:
            self.draw_eye_landmarks(display_frame, results['landmarks'])
        
        # Desenha detecção de íris
        if self.show_iris_detection and results.get('iris_data'):
            self.draw_iris_detection(display_frame, results['iris_data'])
        
        # Desenha vetores de gaze
        if self.show_gaze_vectors and results.get('gaze_vectors'):
            self.draw_gaze_vectors(display_frame, results['gaze_vectors'])
        
        # Interface de calibração
        if is_calibrating:
            display_frame = self.draw_calibration_ui(
                display_frame, calibration_system, screen_calibration
            )
        else:
            # Desenha predição do olhar se calibrado
            if (screen_calibration.is_calibrated() and 
                results.get('screen_point') is not None):
                self.draw_gaze_prediction(display_frame, results['screen_point'])
        
        # Desenha zonas de atenção
        if self.show_attention_zones and not is_calibrating:
            self.draw_attention_zones(display_frame)
        
        # Desenha métricas de consciência situacional
        if self.show_sa_metrics and results.get('sa_analysis'):
            self.draw_sa_metrics(display_frame, results['sa_analysis'])
        
        # Desenha informações de performance
        self.draw_performance_info(display_frame, results, current_fps)
        
        # Desenha controles
        self.draw_controls_info(display_frame, is_calibrating, screen_calibration.is_calibrated())
        
        return display_frame
    
    def draw_face_detection(self, frame, results):
        """Desenha detecção facial"""
        faces = results.get('faces', [])
        face_confidence = results.get('face_confidence')
        
        for i, face in enumerate(faces):
            if face is not None:
                x, y, w, h = face
                cv2.rectangle(frame, (x, y), (x + w, y + h), self.colors['face_box'], 2)
                
                # Mostra confiança se disponível
                if face_confidence is not None:
                    cv2.putText(frame, f"Conf: {face_confidence:.2f}", 
                              (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                              self.colors['face_box'], 1)
    
    def draw_eye_landmarks(self, frame, landmarks):
        """Desenha landmarks dos olhos"""
        for i, (x, y) in enumerate(landmarks):
            color = self.colors['face_box']  # Cor padrão
            
            if i in self.LEFT_EYE_POINTS:
                color = self.colors['left_eye']
            elif i in self.RIGHT_EYE_POINTS:
                color = self.colors['right_eye']
            
            cv2.circle(frame, (int(x), int(y)), 1, color, -1)
    
    def draw_iris_detection(self, frame, iris_data):
        """Desenha detecção de íris"""
        for eye_side, data in iris_data.items():
            center = data['center']
            radius = data['radius']
            
            color = self.colors['iris_left'] if eye_side == 'left' else self.colors['iris_right']
            
            # Círculo da íris
            cv2.circle(frame, tuple(map(int, center)), int(radius), color, 2)
            
            # Centro da íris
            cv2.circle(frame, tuple(map(int, center)), 3, color, -1)
    
    def draw_gaze_vectors(self, frame, gaze_vectors):
        """Desenha vetores de gaze"""
        for eye_side, data in gaze_vectors.items():
            eye_center = data['eye_center']
            iris_center = data['iris_center']
            
            color = self.colors['left_eye'] if eye_side == 'left' else self.colors['right_eye']
            
            # Linha do centro do olho para íris
            cv2.line(frame, tuple(map(int, eye_center)), 
                    tuple(map(int, iris_center)), color, 2)
            
            # Vetor de direção do olhar
            scale = 50
            vector_3d = data['vector_3d']
            end_point = (
                iris_center[0] + int(vector_3d[0] * scale),
                iris_center[1] + int(vector_3d[1] * scale)
            )
            
            cv2.arrowedLine(frame, tuple(map(int, iris_center)), 
                           tuple(map(int, end_point)), self.colors['gaze_vector'], 2)
    
    def draw_gaze_prediction(self, frame, screen_point):
        """Desenha predição do ponto de olhar"""
        frame_x = int(screen_point[0] * frame.shape[1] / 1920)  # Ajusta para resolução do frame
        frame_y = int(screen_point[1] * frame.shape[0] / 1080)
        
        frame_x = np.clip(frame_x, 0, frame.shape[1] - 1)
        frame_y = np.clip(frame_y, 0, frame.shape[0] - 1)
        
        cv2.circle(frame, (frame_x, frame_y), 10, self.colors['predicted_gaze'], -1)
        cv2.circle(frame, (frame_x, frame_y), 15, self.colors['predicted_gaze'], 2)
        cv2.putText(frame, "Olhar", (frame_x + 20, frame_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['predicted_gaze'], 1)
    
    def draw_attention_zones(self, frame):
        """Desenha zonas de atenção"""
        if not hasattr(self.config.situational_awareness, 'attention_zones'):
            return
        
        zones = self.config.situational_awareness.attention_zones
        frame_h, frame_w = frame.shape[:2]
        
        for zone_name, rectangles in zones.items():
            for rect in rectangles:
                # Ajusta coordenadas para o frame atual
                x, y, w, h = rect
                
                # Usar resolução detectada dinamicamente
                if not hasattr(self, '_screen_resolution'):
                    import tkinter as tk
                    root = tk.Tk()
                    self._screen_resolution = (root.winfo_screenwidth(), root.winfo_screenheight())
                    root.destroy()

                screen_w, screen_h = self._screen_resolution
                x = int(x * frame_w / screen_w)
                y = int(y * frame_h / screen_h)
                w = int(w * frame_w / screen_w)
                h = int(h * frame_h / screen_h)
                
                # Desenha retângulo da zona
                cv2.rectangle(frame, (x, y), (x + w, y + h), 
                            self.colors['attention_zone'], 1)
                
                # Label da zona
                cv2.putText(frame, zone_name, (x, y - 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, 
                          self.colors['attention_zone'], 1)

    def draw_gaze_prediction(self, frame, screen_point):
        """Desenha predição do ponto de olhar"""
        frame_h, frame_w = frame.shape[:2]
        
        # --- Lógica de Resolução da Tela Real ---
        # Certifica-se de que a resolução da tela real (treinamento) é conhecida
        if not hasattr(self, '_screen_resolution'):
            import tkinter as tk
            root = tk.Tk()
            self._screen_resolution = (root.winfo_screenwidth(), root.winfo_screenheight())
            root.destroy()

        screen_w, screen_h = self._screen_resolution
        # --- Fim da Lógica de Resolução ---
        
        # O screen_point já está na escala (screen_w, screen_h)
        # É escalado para o tamanho atual do frame (frame_w, frame_h)
        frame_x = int(screen_point[0] * frame_w / screen_w) # CORRIGIDO: usa screen_w
        frame_y = int(screen_point[1] * frame_h / screen_h) # CORRIGIDO: usa screen_h
        
        frame_x = np.clip(frame_x, 0, frame_w - 1)
        frame_y = np.clip(frame_y, 0, frame_h - 1)
        
        cv2.circle(frame, (frame_x, frame_y), 10, self.colors['predicted_gaze'], -1)
        cv2.circle(frame, (frame_x, frame_y), 15, self.colors['predicted_gaze'], 2)
        cv2.putText(frame, "Olhar", (frame_x + 20, frame_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['predicted_gaze'], 1)

    def draw_calibration_ui(self, frame, calibration_system, screen_calibration):
        """Desenha interface de calibração"""
        if not calibration_system.is_active:
            return frame
        
        status = calibration_system.get_calibration_status()
        if not status.get('active'):
            return frame
        
        current_point = status['current_point']
        point = current_point['point']
        
        # Ajusta coordenadas para o frame
        # Ajusta coordenadas para o frame - usar resolução REAL da tela
        frame_h, frame_w = frame.shape[:2]
        # Detecta resolução real da tela (uma vez só)
        if not hasattr(self, '_screen_resolution'):
            import tkinter as tk
            root = tk.Tk()
            self._screen_resolution = (root.winfo_screenwidth(), root.winfo_screenheight())
            root.destroy()

        screen_w, screen_h = self._screen_resolution
        frame_x = int(point[0] * frame_w / screen_w)
        frame_y = int(point[1] * frame_h / screen_h)

        #print(f"🔧 DEBUG: Ponto original: {point}")
        #print(f"🔧 DEBUG: Frame shape: {frame_w}x{frame_h}")
        #print(f"🔧 DEBUG: Config resolution: {self.config.hardware.resolution}")

        #print(f"🔧 DEBUG: Ponto convertido: ({frame_x}, {frame_y})")
        #print("=" * 50)
        
        # Interface para contagem regressiva
        if current_point.get('is_in_countdown', False):
            countdown = current_point.get('countdown_remaining', 0)
            countdown_text = f"PREPARANDO... {countdown:.1f}s"
            
            # Círculo pulsante
            pulse = int(abs(math.sin(time.time() * 3) * 20) + 10)
            cv2.circle(frame, (frame_x, frame_y), 30 + pulse, (0, 255, 255), 3)
            
            # Texto centralizado
            text_size = cv2.getTextSize(countdown_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            text_x = frame_x - text_size[0] // 2
            text_y = frame_y - 40
            cv2.putText(frame, countdown_text, (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            return frame
        
        # Interface para intervalo
        if current_point.get('is_in_interval', False):
            interval_remaining = current_point.get('interval_remaining', 0)
            interval_text = f"INTERVALO... {interval_remaining:.1f}s"
            
            pulse = int(abs(math.sin(time.time() * 2) * 15) + 10)
            cv2.circle(frame, (frame_x, frame_y), 25 + pulse, (255, 165, 0), 3)
            
            text_size = cv2.getTextSize(interval_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            text_x = frame_x - text_size[0] // 2
            text_y = frame_y - 35
            cv2.putText(frame, interval_text, (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
            return frame
        
        # Desenha ponto de calibração ativo
        progress = current_point['progress']
        
        # Círculos de progresso
        radius_outer = 25
        radius_inner = int(15 + (progress * 10))
        
        # Cor baseada no progresso
        if progress < 0.3:
            color = (0, 0, 255)  # Vermelho
        elif progress < 0.7:
            color = (0, 165, 255)  # Laranja
        else:
            color = (0, 255, 0)  # Verde
        
        # Animação especial para "PRONTO"
        if current_point.get('show_ready_message', False):
            pulse = int(abs(math.sin(time.time() * 4) * 10) + 5)
            cv2.circle(frame, (frame_x, frame_y), radius_outer + pulse, (0, 255, 0), 3)
            cv2.putText(frame, "PRONTO!", (frame_x - 35, frame_y - 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.circle(frame, (frame_x, frame_y), radius_outer, color, 2)
        cv2.circle(frame, (frame_x, frame_y), radius_inner, color, -1)
        
        # Informações do ponto
        info_text = f"Ponto {current_point['index'] + 1}/{current_point['total_points']}"
        cv2.putText(frame, info_text, (frame_x - 50, frame_y - 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Progresso das amostras
        samples_text = f"{current_point['samples_collected']}/{current_point['samples_collected'] + current_point['samples_needed']}"
        cv2.putText(frame, samples_text, (frame_x - 30, frame_y + 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Barra de progresso
        bar_width = 80
        bar_height = 6
        bar_x = frame_x - bar_width // 2
        bar_y = frame_y + 50
        
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                     (128, 128, 128), -1)
        
        progress_width = int(bar_width * progress)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + progress_width, bar_y + bar_height), 
                     color, -1)
        
        return frame
    
    def draw_sa_metrics(self, frame, sa_analysis):
        """Desenha métricas de consciência situacional"""
        if not sa_analysis:
            return
        
        y_pos = 30
        
        # Score geral
        sa_score = sa_analysis.get('situational_awareness_score', {})
        overall_score = sa_score.get('overall_score', 0)
        classification = sa_score.get('classification', 'unknown')
        
        # Cor baseada na classificação
        if classification in ['excellent', 'good']:
            color = self.colors['sa_good']
        elif classification == 'moderate':
            color = self.colors['sa_moderate']
        else:
            color = self.colors['sa_poor']
        
        # Desenha score principal
        score_text = f"SA Score: {overall_score:.2f} ({classification})"
        cv2.putText(frame, score_text, (10, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        y_pos += 25
        
        # Zona de atenção atual
        attention = sa_analysis.get('attention_distribution', {})
        current_zone = attention.get('current_zone', 'unknown')
        zone_confidence = attention.get('zone_confidence', 0)
        
        zone_text = f"Zona: {current_zone} ({zone_confidence:.1f})"
        cv2.putText(frame, zone_text, (10, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_pos += 20
        
        # Indicadores de fadiga
        fatigue = sa_analysis.get('fatigue_indicators', {})
        fatigue_level = fatigue.get('fatigue_level', 'unknown')
        fatigue_score = fatigue.get('fatigue_score', 0)
        
        fatigue_color = self.colors['sa_good'] if fatigue_level == 'low' else \
                       self.colors['sa_moderate'] if fatigue_level == 'moderate' else \
                       self.colors['sa_poor']
        
        fatigue_text = f"Fadiga: {fatigue_level} ({fatigue_score:.2f})"
        cv2.putText(frame, fatigue_text, (10, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, fatigue_color, 1)
        y_pos += 20
        
        # Taxa de sacadas
        saccade = sa_analysis.get('saccade_patterns', {})
        saccade_freq = saccade.get('saccade_frequency', 0)
        is_saccade = saccade.get('is_saccade', False)
        
        saccade_text = f"Sacadas/s: {saccade_freq:.1f}"
        if is_saccade:
            saccade_text += " [ATIVA]"
        
        cv2.putText(frame, saccade_text, (10, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def draw_performance_info(self, frame, results, current_fps):
        """Desenha informações de performance"""
        frame_height = frame.shape[0]
        
        # FPS
        fps_text = f"FPS: {current_fps:.1f}"
        cv2.putText(frame, fps_text, (10, frame_height - 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Tempo de processamento
        processing_time = results.get('processing_time', 0)
        if processing_time > 0:
            time_text = f"Proc: {processing_time*1000:.1f}ms"
            cv2.putText(frame, time_text, (10, frame_height - 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Status de detecções
        detections = []
        if results.get('faces'):
            detections.append("Face")
        if results.get('iris_data'):
            detections.append("Íris")
        if results.get('gaze_vectors'):
            detections.append("Gaze")
        
        if detections:
            status_text = f"Detectado: {', '.join(detections)}"
            cv2.putText(frame, status_text, (10, frame_height - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    
    def draw_controls_info(self, frame, is_calibrating, is_calibrated):
        """Desenha informações de controles"""
        frame_width = frame.shape[1]
        y_start = 30
        
        controls = [
            "'c' - Calibrar",
            "'r' - Reset",
            "'s' - Relatório",
            "'q' - Sair"
        ]
        
        if is_calibrating:
            controls = ["'ESC' - Cancelar", "'q' - Sair"]
        elif is_calibrated:
            controls.insert(2, "'d' - Debug")
        
        for i, control in enumerate(controls):
            text_size = cv2.getTextSize(control, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
            x_pos = frame_width - text_size[0] - 10
            y_pos = y_start + i * 20
            
            cv2.putText(frame, control, (x_pos, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    def draw_heatmap_overlay(self, frame, gaze_points, alpha=0.3):
        """Desenha overlay de mapa de calor dos pontos de olhar"""
        if not gaze_points:
            return frame
        
        # Cria máscara de calor
        heatmap = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.float32)
        
        for point in gaze_points:
            x, y = int(point[0]), int(point[1])
            if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
                # Adiciona gaussian blur ao redor do ponto
                cv2.circle(heatmap, (x, y), 30, 1.0, -1)
        
        # Aplica blur para suavizar
        heatmap = cv2.GaussianBlur(heatmap, (61, 61), 0)
        
        # Normaliza
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        
        # Converte para colormap
        heatmap_color = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
        
        # Mistura com frame original
        overlay = cv2.addWeighted(frame, 1 - alpha, heatmap_color, alpha, 0)
        
        return overlay
    
    def draw_trajectory(self, frame, gaze_trajectory, max_points=50):
        """Desenha trajetória do olhar"""
        if len(gaze_trajectory) < 2:
            return
        
        # Limita número de pontos
        recent_trajectory = list(gaze_trajectory)[-max_points:]
        
        # Desenha linhas conectando os pontos
        for i in range(1, len(recent_trajectory)):
            pt1 = tuple(map(int, recent_trajectory[i-1]))
            pt2 = tuple(map(int, recent_trajectory[i]))
            
            # Cor mais intensa para pontos mais recentes
            alpha = i / len(recent_trajectory)
            color = tuple(int(c * alpha) for c in self.colors['gaze_vector'])
            
            cv2.line(frame, pt1, pt2, color, 2)
        
        # Marca o ponto mais recente
        if recent_trajectory:
            last_point = tuple(map(int, recent_trajectory[-1]))
            cv2.circle(frame, last_point, 5, self.colors['predicted_gaze'], -1)
    
    def create_debug_window(self, debug_data):
        """Cria janela separada para informações de debug"""
        debug_frame = np.zeros((400, 600, 3), dtype=np.uint8)
        
        y_pos = 30
        line_height = 20
        
        # Título
        cv2.putText(debug_frame, "DEBUG INFORMATION", (20, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_pos += 40
        
        # Informações de debug
        debug_items = [
            f"Face Detection Interval: {debug_data.get('face_interval', 'N/A')}",
            f"Iris Cache Size: {debug_data.get('cache_size', 'N/A')}",
            f"Processing Time Avg: {debug_data.get('avg_time', 'N/A')}ms",
            f"Memory Usage: {debug_data.get('memory', 'N/A')}MB",
            f"Calibration Points: {debug_data.get('cal_points', 'N/A')}",
            f"SA Score History: {debug_data.get('sa_history', 'N/A')}",
        ]
        
        for item in debug_items:
            cv2.putText(debug_frame, item, (20, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            y_pos += line_height
        
        return debug_frame
    
    def toggle_visualization_elements(self, element):
        """Alterna elementos de visualização"""
        if element == 'landmarks':
            self.show_landmarks = not self.show_landmarks
            return f"Landmarks: {'ON' if self.show_landmarks else 'OFF'}"
        elif element == 'iris':
            self.show_iris_detection = not self.show_iris_detection
            return f"Detecção de Íris: {'ON' if self.show_iris_detection else 'OFF'}"
        elif element == 'gaze':
            self.show_gaze_vectors = not self.show_gaze_vectors
            return f"Vetores de Gaze: {'ON' if self.show_gaze_vectors else 'OFF'}"
        elif element == 'zones':
            self.show_attention_zones = not self.show_attention_zones
            return f"Zonas de Atenção: {'ON' if self.show_attention_zones else 'OFF'}"
        elif element == 'sa_metrics':
            self.show_sa_metrics = not self.show_sa_metrics
            return f"Métricas SA: {'ON' if self.show_sa_metrics else 'OFF'}"
        
        return "Elemento não reconhecido"
