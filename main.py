import cv2
import time
import numpy as np
import tkinter as tk
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
from src.utils.data_collector import ResearchDataCollector
from src.calibration.screen_calibration import ScreenCalibration, AdaptiveCalibration
from src.filters.kalman_filter import GazeKalmanFilter, AdaptiveKalmanFilter


class GazeTrackingSystem:
    """Sistema completo: 3 algoritmos + melhorias MonitorTracking"""
    
    def __init__(self, config=None):
        self.config = config or GazeTrackingConfig()
        
        # Componentes principais
        self.face_detector = RobustFaceDetector(self.config)
        self.iris_tracker = IrisTracker(self.config)
        self.gaze_calculator = GazeCalculator(self.config)
        
        # === ALGORITMO 1: SCREEN CALIBRATION ===
        root = tk.Tk()
        screen_w = root.winfo_screenwidth()
        screen_h = root.winfo_screenheight()
        root.destroy()
        
        self.screen_calibrator = ScreenCalibration(screen_w, screen_h)
        self.adaptive_screen_cal = AdaptiveCalibration(self.screen_calibrator)
        
        if not self.screen_calibrator.load_calibration():
            print("⚠️ Calibração não encontrada. Execute run_calibration().")
        
        # === ALGORITMO 2: KALMAN FILTER ===
        self.kalman_filter = AdaptiveKalmanFilter()
        self.use_kalman = True
        
        # === ALGORITMO 3: FATIGUE DETECTOR ===
        #self.fatigue_detector = AdvancedFatigueDetector(fps=30)
        self.enable_fatigue_detection = False
        
        # Sistemas de calibração (original)
        #self.adaptive_calibration = AdaptiveCalibrationSystem(self.config)
        #self.screen_calibration = ScreenCalibrationSystem(self.config)
        
        # Utilitários
        self.performance_optimizer = PerformanceOptimizer(self.config)
        self.visualization_manager = VisualizationManager(self.config)
        self.data_collector = ResearchDataCollector(self.config) if self.config.collect_research_data else None
        
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
        
        # Métricas
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        self.gaze_history = deque(maxlen=100)
        self._last_gaze = None
        
        self.cap = None
        
        print("🚀 Sistema Completo Inicializado")
        print("✅ 3 Algoritmos: Kalman, Screen Cal, Fatigue")
        print("✅ Melhorias 3D:", 
              f"MediaPipe={self.config.algorithm.use_mediapipe}",
              f"PCA={self.config.algorithm.use_pca_pose}",
              f"Esferas={self.config.algorithm.use_eye_spheres}")
    
    def initialize_camera(self) -> bool:
        """Inicializa câmera"""
        try:
            self.cap = cv2.VideoCapture(self.config.hardware.camera_id)
            if not self.cap.isOpened():
                print("❌ Erro ao abrir webcam")
                return False
            
            width, height = self.config.hardware.resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.cap.set(cv2.CAP_PROP_FPS, self.config.hardware.target_fps)
            
            print(f"✅ Câmera: {width}x{height} @ {self.config.hardware.target_fps}fps")
            return True
        except Exception as e:
            print(f"❌ Erro: {e}")
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
                        if self.iris_tracker.improved_detection.face_mesh_results:
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
                
                # === 6. SCREEN POINT (com calibração) ===
                if gaze_vectors and 'left' in gaze_vectors and 'right' in gaze_vectors:
                    if self.screen_calibrator.is_calibrated():
                        try:
                            avg_yaw = (gaze_vectors['left']['yaw'] + gaze_vectors['right']['yaw']) / 2
                            avg_pitch = (gaze_vectors['left']['pitch'] + gaze_vectors['right']['pitch']) / 2
                            screen_point = self.screen_calibrator.map_gaze_to_screen(avg_yaw, avg_pitch)
                            
                            # === 7. KALMAN FILTER ===
                            if self.use_kalman and screen_point:
                                filtered_screen_point = self.kalman_filter.update(screen_point)
                            else:
                                filtered_screen_point = screen_point
                        except Exception as e_map:
                            if self.config.debug_enabled:
                                print(f"Erro no mapeamento: {e_map}")
                            pass
                    
                    # === 8. FATIGUE DETECTION ===
                    if self.enable_fatigue_detection and iris_data:
                        # (Lógica de fadiga permanece desativada por enquanto)
                        pass 
                        # eye_metrics = { ... }
                        # fatigue_state = self.fatigue_detector.update(eye_metrics)
                        # ...
        
        except Exception as e:
            if self.config.debug_enabled:
                print(f"❌ Erro em process_frame: {e}")
        
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
    
    def run_calibration(self):
        """
        Executa a calibração de tela usando o loop e a lógica 
        da classe ScreenCalibration.
        """
        if not self.cap or not self.cap.isOpened():
            print("❌ Câmera não inicializada. Tentando inicializar...")
            if not self.initialize_camera():
                return False
        
        # Inicia o sistema de calibração
        self.screen_calibrator.start_calibration()
        print("Iniciando loop de calibração... Pressione 'q' para cancelar.")

        try:
            while not self.screen_calibrator.calibration_complete:
                # 1. Obter frame da câmera
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    print("❌ Erro ao ler frame da câmera na calibração, pulando...")
                    time.sleep(0.05)
                    continue # Pula para a próxima iteração
                
                # 2. Processar o frame para obter o gaze
                # (Usamos o mesmo pipeline do 'run' principal)
                result = self.process_frame(frame)
                
                gaze_data = None
                if result['gaze_vectors'] and 'left' in result['gaze_vectors']:
                    gaze = result['gaze_vectors']
                    avg_yaw = (gaze['left']['yaw'] + gaze['right']['yaw']) / 2
                    avg_pitch = (gaze['left']['pitch'] + gaze['right']['pitch']) / 2
                    gaze_data = {'yaw': avg_yaw, 'pitch': avg_pitch}
                    
                    # Debug para você:
                    print(f"DEBUG Calibração: Gaze detectado ({avg_yaw:.2f}, {avg_pitch:.2f})")
                else:
                    # Debug para você:
                    print("DEBUG Calibração: Gaze não detectado...")
                
                # 3. Coletar a amostra de gaze
                # A classe ScreenCalibration gerencia internamente se deve
                # coletar, quantas amostras faltam, etc.
                self.screen_calibrator.collect_gaze_sample(gaze_data)
                
                # 4. Obter e exibir a tela de calibração
                # A classe ScreenCalibration desenha o ponto correto,
                # a barra de progresso, e muda a cor.
                cal_frame = self.screen_calibrator.display_calibration_point()
                cv2.imshow('Calibration', cal_frame)
                
                # 5. Adicionar um pequeno feed da câmera (opcional, mas útil)
                # Isso ajuda a saber se a câmera está vendo você.
                vis_frame = frame.copy()
                if result.get('best_face') is not None:
                    (x, y, w, h) = result['best_face']
                    cv2.rectangle(vis_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Redimensiona para caber no canto
                vis_frame_small = cv2.resize(vis_frame, (320, 240))
                cv2.imshow('Camera Feed (Calibracao)', vis_frame_small)
                cv2.moveWindow('Camera Feed (Calibracao)', cal_frame.shape[1] - 330, 10)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27: # 'q' ou ESC
                    print("\n❌ Calibração cancelada pelo usuário.")
                    self.screen_calibrator.calibration_complete = False # Força a saída
                    cv2.destroyWindow('Calibration')
                    cv2.destroyWindow('Camera Feed (Calibracao)')
                    return False
            
            # Se o loop terminar (calibração completa)
            print("✅ Loop de calibração finalizado.")
            cv2.destroyWindow('Calibration')
            cv2.destroyWindow('Camera Feed (Calibracao)')
            
            # A classe já salvou os dados no _finalize_calibration()
            return self.screen_calibrator.calibration_complete
            
        except Exception as e:
            print(f"❌ Erro grave durante a calibração: {e}")
            cv2.destroyWindow('Calibration')
            cv2.destroyWindow('Camera Feed (Calibracao)')
            return False

    """
    def run_calibration(self):
        #Calibração de tela (9 pontos)
        if not self.cap:
            print("❌ Câmera não inicializada")
            return False
        
        print("\n" + "="*60)
        print("🎯 CALIBRAÇÃO DE TELA")
        print("="*60)
        print("- Olhe para cada ponto verde")
        print("- Mantenha cabeça estável")
        print("- ESC para cancelar")
        print("="*60 + "\n")
        
        cv2.namedWindow('Calibração', cv2.WINDOW_NORMAL)
        cv2.setWindowProperty('Calibração', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        try:
            for point_idx, (x, y) in enumerate(self.screen_calibrator.calibration_points):
                print(f"\n📍 Ponto {point_idx + 1}/9: ({x}, {y})")
                
                cal_frame = np.zeros((self.screen_calibrator.screen_height,
                                     self.screen_calibrator.screen_width, 3), dtype=np.uint8)
                
                cv2.circle(cal_frame, (x, y), 15, (0, 255, 0), -1)
                cv2.circle(cal_frame, (x, y), 20, (255, 255, 255), 2)
                cv2.putText(cal_frame, f"Ponto {point_idx + 1}/9", (50, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                cv2.imshow('Calibração', cal_frame)
                cv2.waitKey(1000)
                
                collected_samples = []
                for sample in range(30):
                    ret, frame = self.cap.read()
                    if not ret:
                        continue
                    
                    result = self.process_frame(frame)

                    # === ADICIONAR ===
                    print(f"DEBUG: gaze_vectors = {result.get('gaze_vectors')}")
                    print(f"DEBUG: iris_data = {result.get('iris_data')}")
                    print(f"DEBUG: best_face = {result.get('best_face')}")
                    # ===
                    
                    if result['gaze_vectors'] and 'left' in result['gaze_vectors']:
                        gaze = result['gaze_vectors']
                        avg_yaw = (gaze['left']['yaw'] + gaze['right']['yaw']) / 2
                        avg_pitch = (gaze['left']['pitch'] + gaze['right']['pitch']) / 2
                        collected_samples.append({'yaw': avg_yaw, 'pitch': avg_pitch})
                    
                    progress = int((sample / 30) * 100)
                    cv2.rectangle(cal_frame, (50, 100), (50 + progress*5, 130), (0, 255, 0), -1)
                    cv2.imshow('Calibração', cal_frame)
                    
                    if cv2.waitKey(30) & 0xFF == 27:
                        raise KeyboardInterrupt
                
                if len(collected_samples) >= 20:
                    self.screen_calibrator.add_calibration_point(x, y, collected_samples)
                    print(f"✅ Coletado: {len(collected_samples)} amostras")
                else:
                    print(f"⚠️ Poucas amostras: {len(collected_samples)}")
            
            print("\n🧠 Treinando modelo...")
            self.screen_calibrator.train_model()
            self.screen_calibrator.save_calibration()
            
            report = self.screen_calibrator.get_calibration_report()
            print("\n📊 RELATÓRIO")
            print(f"Status: {report['status']}")
            print(f"Pontos: {report['points_collected']}")
            print(f"Qualidade: {report['quality_score']:.1f}/100\n")
            
            return True
            
        except KeyboardInterrupt:
            print("\n❌ Cancelado")
            return False
        finally:
            cv2.destroyWindow('Calibração')
    """
    
    def calibrate_eye_spheres(self, iris_data):
        """Calibra esferas oculares 3D"""
        if not self.config.algorithm.use_eye_spheres:
            print("⚠️ Esferas 3D não habilitadas")
            return False
        
        if self.current_head_center is None or self.current_R_head is None:
            print("❌ Calcule pose primeiro")
            return False
        
        if 'left' not in iris_data or 'right' not in iris_data:
            print("❌ Ambos os olhos necessários")
            return False
        
        success = self.gaze_calculator.calibrate_eye_spheres(
            iris_data, self.current_head_center, self.current_R_head, self.current_scale
        )
        
        if success:
            if hasattr(self.face_detector, 'calibrate_pose_scale'):
                self.face_detector.calibrate_pose_scale()
            
            self.spheres_calibrated = True
            print("\n✅ ESFERAS CALIBRADAS! Sistema compensa distância automaticamente.\n")
            return True
        
        return False
    
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
            return
        
        if not self.screen_calibrator.is_calibrated():
            print("\n⚠️ Sistema não calibrado!")
            response = input("Calibrar agora? (s/n): ")
            if response.lower() == 's':
                self.run_calibration()
        
        self.is_running = True
        
        print("\n🚀 Sistema Rodando")
        print("\nControles:")
        print("  c - Calibrar tela")
        print("  e - Calibrar esferas 3D")
        print("  r - Reset calibração")
        print("  k - Toggle Kalman")
        print("  f - Toggle Fadiga")
        print("  i - Info sistema 3D")
        print("  m - Métricas Kalman")
        print("  d - Toggle debug")
        print("\n  Ajuste de sensibilidade:")
        print("  + - Aumentar ganho (+0.1)")
        print("  - - Diminuir ganho (-0.1)")
        print("  g - Mostrar configurações de gaze")
        print("\n  q - Sair\n")
        
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                print("⚠️ Erro ao ler frame da câmera, pulando...")
                time.sleep(0.05) # Evita loop 100% CPU se a câmera morrer
                continue # Pula para a próxima iteração
            
            result = self.process_frame(frame)
            
            """
            vis_frame = self.visualization_manager.render_frame(
                frame, result, self.is_calibrating, self.adaptive_calibration,
                self.screen_calibration, self.current_fps
            )
            """
            vis_frame = self.visualization_manager.render_frame(
                frame, result, self.is_calibrating, None,
                self.screen_calibrator, self.current_fps
            )
            
            self._draw_info_overlay(vis_frame, result)
            
            cv2.imshow('Gaze Tracking - Sistema Completo', vis_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('c'):
                self.run_calibration()
            elif key == ord('e'):
                print("\n⚙️ Calibrando esferas... Olhe para câmera!")
                time.sleep(2)
                ret, frame = self.cap.read()
                if ret:
                    result = self.process_frame(frame)
                    if result['iris_data']:
                        self.calibrate_eye_spheres(result['iris_data'])
            elif key == ord('r'):
                print("🔄 Reset")
                self.screen_calibrator.reset_calibration()
                self.spheres_calibrated = False
            elif key == ord('k'):
                self.use_kalman = not self.use_kalman
                print(f"Kalman: {'ON' if self.use_kalman else 'OFF'}")
            elif key == ord('f'):
                self.enable_fatigue_detection = not self.enable_fatigue_detection
                print(f"Fadiga: {'ON' if self.enable_fatigue_detection else 'OFF'}")
            elif key == ord('i'):
                self._print_3d_info()
            elif key == ord('m'):
                if self.use_kalman:
                    metrics = self.kalman_filter.get_smoothness_metrics()
                    print("\n📈 KALMAN:", metrics)
            elif key == ord('d'):
                self.config.debug_enabled = not self.config.debug_enabled
                status = "ATIVADO" if self.config.debug_enabled else "DESATIVADO"
                print(f"🐛 Debug: {status}")
            elif key == ord('+') or key == ord('='):
                self.config.algorithm.gaze_gain += 0.1
                print(f"🎚️ Ganho: {self.config.algorithm.gaze_gain:.2f}")
            elif key == ord('-') or key == ord('_'):
                self.config.algorithm.gaze_gain = max(0.1, self.config.algorithm.gaze_gain - 0.1)
                print(f"🎚️ Ganho: {self.config.algorithm.gaze_gain:.2f}")
            elif key == ord('g'):
                self._print_gaze_settings()

            self.fps_counter += 1
            if time.time() - self.fps_start_time >= 1.0:
                self.current_fps = self.fps_counter
                self.fps_counter = 0
                self.fps_start_time = time.time()
        
        self.cleanup()
    
    def _draw_info_overlay(self, frame, result):
        """Overlay com info dos algoritmos"""
        # Kalman
        if self.use_kalman:
            cv2.putText(frame, "KALMAN: ON", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
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
            
            cv2.putText(frame, f"FADIGA: {level.upper()}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Esferas calibradas?
        if self.spheres_calibrated:
            cv2.putText(frame, "ESFERAS: OK", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    def _print_3d_info(self):
        """Info do sistema 3D"""
        print("\n" + "="*50)
        print("📊 SISTEMA 3D")
        print("="*50)
        print(f"MediaPipe: {'✅' if self.config.algorithm.use_mediapipe else '❌'}")
        print(f"PCA: {'✅' if self.config.algorithm.use_pca_pose else '❌'}")
        print(f"Esferas: {'✅' if self.config.algorithm.use_eye_spheres else '❌'}")
        print(f"Calibradas: {'✅' if self.spheres_calibrated else '❌ (pressione E)'}")

        if self.current_head_center is not None:
            print(f"\nPose: ({self.current_head_center[0]:.1f}, {self.current_head_center[1]:.1f}, {self.current_head_center[2]:.1f})")
            scale = self.face_detector.get_scale_ratio()
            if scale:
                print(f"Distância: {scale:.2f}x")
        print("="*50 + "\n")

    def _print_gaze_settings(self):
        """Exibe configurações atuais de sensibilidade do gaze"""
        print("\n" + "="*60)
        print("🎚️ CONFIGURAÇÕES DE SENSIBILIDADE DO GAZE (NOVO MÉTODO)")
        print("="*60)
        print(f"Ganho: {self.config.algorithm.gaze_gain:.2f}x")
        print(f"  └─ Amplificação dos ângulos detectados")
        print(f"\nMétodo de normalização: Largura do olho (automático)")
        print(f"  └─ Calcula posição relativa da íris no olho")
        print(f"  └─ Valores típicos: dx/dy entre -0.5 e +0.5")
        print("\n💡 DICAS:")
        print("  • Se o gaze está muito 'parado': pressione + para aumentar ganho")
        print("  • Se o gaze está muito 'nervoso': pressione - para diminuir ganho")
        print("  • Valores recomendados: Ganho=0.8-1.5")
        print(f"  • Ganho atual: {self.config.algorithm.gaze_gain:.2f}")
        print("\n🐛 DEBUG:")
        print("  • Pressione 'd' para ativar debug e ver valores intermediários")
        print("  • Verifique se dx/dy estão variando durante movimentos oculares")
        print("="*60 + "\n")
    
    def cleanup(self):
        """Limpa recursos"""
        print("\n🧹 Finalizando...")
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        
        if self.data_collector:
            try:
                self.data_collector.save_session_data()
                print("💾 Dados salvos")
            except Exception as e:
                print(f"⚠️ Erro ao salvar: {e}")
        
        print("✅ Finalizado")


def main():
    """Função principal"""
    print("🚀 SISTEMA COMPLETO DE GAZE TRACKING")
    print("="*70)
    print("✅ 3 Algoritmos: Kalman + Screen Cal + Fadiga")
    print("✅ Melhorias 3D: MediaPipe + PCA + Esferas")
    print("="*70)
    
    try:
        choice = input("\n1. Padrão\n2. Personalizado\nEscolha (1-2) [1]: ").strip() or "1"
        
        if choice == "2":
            config = GazeTrackingConfig()
            
            fps = input(f"FPS [{config.hardware.target_fps}]: ").strip()
            if fps:
                config.hardware.target_fps = int(fps)
            
            mediapipe = input("MediaPipe? (s/n) [s]: ").strip().lower() or 's'
            config.algorithm.use_mediapipe = mediapipe == 's'
            
            pca = input("PCA pose? (s/n) [s]: ").strip().lower() or 's'
            config.algorithm.use_pca_pose = pca == 's'
            
            spheres = input("Esferas 3D? (s/n) [s]: ").strip().lower() or 's'
            config.algorithm.use_eye_spheres = spheres == 's'
            
            system = GazeTrackingSystem(config)
        else:
            system = GazeTrackingSystem()
        
        system.run()
        print("\n🎉 Sucesso!")
        
    except KeyboardInterrupt:
        print("\n⏹️ Cancelado")
    except Exception as e:
        print(f"\n❌ Erro: {e}")
        print("Instale: pip install opencv-python mediapipe scipy numpy scikit-learn")


if __name__ == "__main__":
    main()
