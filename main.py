import cv2
import time
import numpy as np
import tkinter as tk
from collections import deque

# Importações dos módulos locais
from src.config.settings import GazeTrackingConfig, DEFAULT_CONFIG
from src.core.face_detector import RobustFaceDetector
from src.core.iris_tracker import IrisTracker
from src.core.gaze_calculator import GazeCalculator
from src.core.calibration import AdaptiveCalibrationSystem, ScreenCalibrationSystem
from src.analysis.situational_awareness import SituationalAwarenessAnalyzer
from src.utils.performance import PerformanceOptimizer
from src.utils.visualization import VisualizationManager
from src.utils.data_collector import ResearchDataCollector

class GazeTrackingSystem:
    """Sistema principal de rastreamento ocular"""
    
    def __init__(self, config=None):
        # Configuração
        self.config = config or GazeTrackingConfig()
        
        # Componentes principais
        self.face_detector = RobustFaceDetector(self.config)
        self.iris_tracker = IrisTracker(self.config)
        #self.gaze_calculator = GazeCalculator(self.config)
        #from src.core.gaze_calculator import PersonalizedGazeCalculator
        #self.gaze_calculator = PersonalizedGazeCalculator(self.config)
        
        from src.core.gaze_calculator import GazeCalculator # GARANTIR IMPORT
        self.gaze_calculator = GazeCalculator(self.config) # USAR O CÁLCULO DE GAZE PADRÃO

        # Sistemas de calibração
        self.adaptive_calibration = AdaptiveCalibrationSystem(self.config)
        self.screen_calibration = ScreenCalibrationSystem(self.config)
        
        # Análise de consciência situacional
        self.sa_analyzer = SituationalAwarenessAnalyzer(self.config)
        
        # Utilitários
        self.performance_optimizer = PerformanceOptimizer(self.config)
        self.visualization_manager = VisualizationManager(self.config)
        self.data_collector = ResearchDataCollector(self.config) if self.config.collect_research_data else None
        
        # Estados do sistema
        self.is_calibrating = False
        self.use_adaptive_calibration = True
        self.is_running = False
        
        # Métricas de performance
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
        # Histórico para análise temporal
        self.gaze_history = deque(maxlen=100)
        
        # Captura de vídeo
        self.cap = None
        
        print("🚀 Sistema de Rastreamento Ocular inicializado")
        self.print_system_info()
    
    def print_system_info(self):
        """Imprime informações do sistema"""
        print("\n=== SISTEMA DE RASTREAMENTO OCULAR ===")
        print(f"📊 Método de calibração: {self.config.calibration.method}")
        print(f"👁️ Detecção facial: {self.config.algorithm.face_detection_method}")
        print(f"🎯 Detecção de íris: {self.config.algorithm.iris_detection_method}")
        print(f"📐 Resolução alvo: {self.config.hardware.resolution}")
        print(f"⚡ FPS alvo: {self.config.hardware.target_fps}")
        print(f"🔬 Coleta de dados: {'Ativada' if self.config.collect_research_data else 'Desativada'}")
        print("=======================================\n")
    
    def initialize_camera(self) -> bool:
        """Inicializa captura da câmera"""
        try:
            self.cap = cv2.VideoCapture(self.config.hardware.camera_id)
            
            if not self.cap.isOpened():
                print("❌ Erro: Não foi possível abrir a webcam")
                return False
            
            # Configurar resolução
            width, height = self.config.hardware.resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.cap.set(cv2.CAP_PROP_FPS, self.config.hardware.target_fps)
            
            cv2.namedWindow('Gaze Tracking - Consciência Situacional', cv2.WINDOW_NORMAL)
            cv2.setWindowProperty('Gaze Tracking - Consciência Situacional', 
                                cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

            print(f"✅ Câmera inicializada: {width}x{height} @ {self.config.hardware.target_fps}fps")
            print("🖥️ Modo tela cheia ativado")
            return True
            
        except Exception as e:
            print(f"❌ Erro ao inicializar câmera: {e}")
            return False
    
    def process_frame(self, frame):
        """Processa um frame completo - VERSÃO ROBUSTA"""
        frame_start_time = time.time()
        
        # ... (Mantém a lógica de Detecção Facial e Histórico) ...
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.performance_optimizer.should_detect_face():
            faces, confidences = self.face_detector.detect_faces(frame)
            self.face_detector.update_face_history(faces, confidences)
        
        best_face, face_confidence = self.face_detector.get_best_face()
        
        landmarks = None
        iris_data = {}
        gaze_vectors = {}
        screen_point = None
        
        if best_face is not None:
            try:
                landmarks = self.face_detector.get_landmarks(gray, best_face)
                
                if landmarks is not None:
                    # 1. Rastreamento de íris (usando a versão robusta que configuramos)
                    iris_data = self.iris_tracker.track_iris(frame, landmarks)

                    gaze_vectors = self.gaze_calculator.calculate_gaze(
                        iris_data, landmarks, frame, best_face # <--- NOVO FLUXO
                    )
                        
                    if gaze_vectors:
                            
                         # 4. Predição do ponto na tela (se calibrado)
                        if self.screen_calibration.is_calibrated() and not self.is_calibrating:
                            avg_gaze = self.gaze_calculator.calculate_average_gaze(gaze_vectors)
                                
                            if avg_gaze:
                                # Usa o modelo treinado (Híbrido/ML/Geométrico)
                                screen_point = self.screen_calibration.predict_screen_point(
                                        avg_gaze['yaw'], avg_gaze['pitch']
                                    )
                                    
            except Exception as e:
                if self.config.debug_enabled:
                    print(f"Erro no processamento robusto: {e}")
        
        # ... (Mantém a lógica de Coleta de Dados e Análise SA) ...

        # Processa calibração se ativa
        if self.is_calibrating and self.use_adaptive_calibration:
            if gaze_vectors:
                self.add_calibration_data_adaptive(gaze_vectors)
        
        # Análise de consciência situacional (se não calibrando)
        sa_analysis = {}
        if not self.is_calibrating and gaze_vectors:
            sa_analysis = self.sa_analyzer.analyze_gaze_data(
                gaze_vectors, screen_point, time.time()
            )
        
        # Coleta dados para pesquisa
        if self.data_collector and gaze_vectors:
            self.data_collector.collect_frame_data({
                'timestamp': time.time(),
                'gaze_vectors': gaze_vectors,
                'screen_point': screen_point,
                'iris_data': iris_data,
                'face_confidence': face_confidence,
                'sa_analysis': sa_analysis
            })
        
        # Calcula tempo de processamento
        processing_time = time.time() - frame_start_time
        self.performance_optimizer.track_processing_time(processing_time)
        
        return {
            'faces': [best_face] if best_face else [],
            'face_confidence': face_confidence,
            'landmarks': landmarks,
            'iris_data': iris_data,
            'gaze_vectors': gaze_vectors,
            'screen_point': screen_point,
            'sa_analysis': sa_analysis,
            'processing_time': processing_time
        }
    
    def add_calibration_data_adaptive(self, gaze_vectors):
        """Adiciona dados à calibração adaptativa com bias personalizado"""
        if not self.is_calibrating or not self.adaptive_calibration.is_active:
            return
            
        if not gaze_vectors or 'left' not in gaze_vectors or 'right' not in gaze_vectors:
            return
        
        # Adiciona amostra ao sistema adaptativo
        result = self.adaptive_calibration.add_gaze_sample(gaze_vectors)
        
        if result:
            # Se retornou dados de treinamento, finaliza calibração
            if isinstance(result, list):
                print("🧠 Treinando modelos de calibração...")
                
                # Treina modelo tradicional
                self.screen_calibration.train_calibration_models(result)
                
                # NOVO: Calibra bias personalizado
                # print("🎯 Calibrando bias personalizado...")
                # self.gaze_calculator.calibrate_personal_bias(result)
                
                # Mostra relatório
                # print(self.gaze_calculator.get_bias_quality_report())
                
                self.is_calibrating = False
                print("✅ Calibração adaptativa com bias personalizado concluída!")
    
    def start_calibration_adaptive(self):
        """Inicia calibração adaptativa"""
        if not self.cap:
            print("❌ Câmera não inicializada")
            return False
        
        self.is_calibrating = True
        
        # Obtém resolução atual
        root = tk.Tk()
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        root.destroy()
        
        # Inicia sistema adaptativo
        self.adaptive_calibration.start_adaptive_calibration(screen_width, screen_height)
        
        print("🎯 Calibração Adaptativa Iniciada!")
        print("👀 Mantenha o olhar fixo em cada ponto")
        print("⚡ A coleta é automática - não pressione nada")
        return True
    
    def update_fps_counter(self):
        """Atualiza contador de FPS"""
        self.fps_counter += 1
        elapsed = time.time() - self.fps_start_time
        
        if elapsed >= 1.0:
            self.current_fps = self.fps_counter / elapsed
            self.fps_counter = 0
            self.fps_start_time = time.time()
    
    def handle_key_input(self, key):
        """Processa entrada do teclado"""
        if key == ord('q'):
            return 'quit'
        elif key == ord('c'):
            if not self.is_calibrating:
                self.start_calibration_adaptive()
        elif key == ord('r'):
            self.screen_calibration.reset()
            self.adaptive_calibration = AdaptiveCalibrationSystem(self.config)
            print("🔄 Calibração resetada")
        elif key == ord('m'):
            self.use_adaptive_calibration = not self.use_adaptive_calibration
            mode = "Adaptativa" if self.use_adaptive_calibration else "Clássica"
            print(f"🔧 Modo de calibração: {mode}")
        elif key == ord('b'):
            # Mostra relatório de bias
            if hasattr(self.gaze_calculator, 'get_bias_quality_report'):
                print(self.gaze_calculator.get_bias_quality_report())
            else:
                print("Sistema de bias não disponível")
        elif key == ord('f'):  # Tecla 'f' para fullscreen
            # Toggle fullscreen
            if not hasattr(self, 'is_fullscreen'):
                self.is_fullscreen = False
            
            self.is_fullscreen = not self.is_fullscreen
            
            if self.is_fullscreen:
                cv2.setWindowProperty('Gaze Tracking - Consciência Situacional', 
                                    cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                print("🖥️ Modo tela cheia ativado")
            else:
                cv2.setWindowProperty('Gaze Tracking - Consciência Situacional', 
                                    cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                print("🪟 Modo janela ativado")
        elif key == 27:  # ESC
            if self.is_calibrating:
                self.is_calibrating = False
                self.adaptive_calibration = AdaptiveCalibrationSystem(self.config)
                print("❌ Calibração cancelada")
        elif key == ord('d'):
            self.config.debug_enabled = not self.config.debug_enabled
            status = "ativadas" if self.config.debug_enabled else "desativadas"
            print(f"🐛 Informações de debug {status}")
        elif key == ord('s'):
            # Salva relatório de sessão
            if hasattr(self.sa_analyzer, 'generate_session_report'):
                report = self.sa_analyzer.generate_session_report()
                print("\n📊 RELATÓRIO DE SESSÃO:")
                print(f"⏱️ Duração: {report['session_duration']:.1f}s")
                print(f"👁️ Fixações: {report['total_fixations']}")
                print(f"🔄 Sacadas: {report['total_saccades']}")
                print(f"🎯 Mudanças de atenção: {report['attention_switches']}")
        elif key == ord('v'):
            self.configure_calibration_intervals()
        
        return 'continue'
    
    def configure_calibration_intervals(self):
        """Menu para configurar intervalos de calibração"""
        print("\n=== CONFIGURAÇÃO DE INTERVALOS ===")
        print(f"Status atual: {'Ativados' if self.adaptive_calibration.interval_enabled else 'Desativados'}")
        print(f"Duração: {self.adaptive_calibration.interval_duration}s")
        print(f"Contagem regressiva: {self.adaptive_calibration.countdown_duration}s")
        print("\nPressione 'v' novamente durante a execução para acessar este menu")
    
    def run(self):
        """Executa o sistema principal"""
        if not self.initialize_camera():
            return False
        
        self.is_running = True
        
        print("\n🚀 Sistema iniciado! Controles:")
        print("  'c' - Iniciar calibração adaptativa")
        print("  'r' - Reset calibração")
        print("  'b' - Mostrar relatório de bias")
        print("  's' - Relatório de sessão")
        print("  'd' - Toggle debug")
        print("  'ESC' - Cancelar calibração")
        print("  'q' - Sair")
        
        try:
            while self.is_running:
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ Erro na captura do frame")
                    break
                
                # Processa frame
                results = self.process_frame(frame)
                
                # Visualização
                display_frame = self.visualization_manager.render_frame(
                    frame, results, self.is_calibrating, self.adaptive_calibration,
                    self.screen_calibration, self.current_fps
                )
                
                # Mostra frame
                cv2.imshow('Gaze Tracking - Consciência Situacional', display_frame)
                
                # Atualiza FPS
                self.update_fps_counter()
                
                # Processa entrada do teclado
                key = cv2.waitKey(1) & 0xFF
                action = self.handle_key_input(key)
                
                if action == 'quit':
                    break
        
        except KeyboardInterrupt:
            print("\n⏹️ Sistema interrompido pelo usuário")
        except Exception as e:
            print(f"❌ Erro durante execução: {e}")
        finally:
            self.cleanup()
        
        return True
    
    def cleanup(self):
        """Limpa recursos do sistema"""
        print("\n🧹 Limpando recursos...")
        
        if self.cap:
            self.cap.release()
        
        cv2.destroyAllWindows()
        
        # Salva dados de pesquisa se disponível
        if self.data_collector:
            try:
                self.data_collector.save_session_data()
                print("💾 Dados de pesquisa salvos")
            except Exception as e:
                print(f"⚠️ Erro ao salvar dados: {e}")
        
        # Gera relatório final
        if hasattr(self.sa_analyzer, 'generate_session_report'):
            try:
                report = self.sa_analyzer.generate_session_report()
                print("\n📊 RELATÓRIO FINAL DE SESSÃO:")
                print(f"⏱️ Duração total: {report.get('session_duration', 0):.1f}s")
                print(f"👁️ Total de fixações: {report.get('total_fixations', 0)}")
                print(f"🔄 Total de sacadas: {report.get('total_saccades', 0)}")
                print(f"🎯 Mudanças de atenção: {report.get('attention_switches', 0)}")
                
                performance = report.get('performance_summary', {})
                if performance.get('status') == 'complete':
                    print(f"📈 Score de consciência situacional: {performance.get('average_sa_score', 0):.2f}")
                    print(f"🏷️ Classificação: {performance.get('sa_classification', 'N/A')}")
                
            except Exception as e:
                print(f"⚠️ Erro ao gerar relatório: {e}")
        
        print("✅ Sistema finalizado")


def create_mining_simulation_system():
    """Cria sistema configurado para simulação de mineração"""
    return GazeTrackingSystem(DEFAULT_CONFIG)


def main():
    """Função principal"""
    print("🚛 Sistema de Rastreamento Ocular - Consciência Situacional em Mineração")
    print("=" * 70)
    
    # Opções de configuração
    print("\nEscolha a configuração:")
    print("1. Configuração padrão")
    print("2. Configuração para simulador de mineração")
    print("3. Configuração personalizada")
    
    try:
        choice = input("\nDigite sua escolha (1-3) [1]: ").strip() or "1"
        
        if choice == "2":
            system = create_mining_simulation_system()
            print("⛏️ Sistema configurado para simulação de mineração")
        elif choice == "3":
            # Configuração personalizada básica
            config = GazeTrackingConfig()
            
            # Permite algumas personalizações
            print("\n=== Configuração Personalizada ===")
            
            try:
                fps = input(f"FPS desejado [{config.hardware.target_fps}]: ").strip()
                if fps:
                    config.hardware.target_fps = int(fps)
                
                resolution = input("Resolução (width,height) [640,480]: ").strip()
                if resolution and ',' in resolution:
                    w, h = map(int, resolution.split(','))
                    config.hardware.resolution = (w, h)
                
                debug = input("Ativar debug? (s/n) [n]: ").strip().lower()
                config.debug_enabled = debug == 's'
                
            except ValueError:
                print("⚠️ Valor inválido, usando configuração padrão")
            
            system = GazeTrackingSystem(config)
            print("🔧 Sistema configurado com parâmetros personalizados")
        else:
            system = GazeTrackingSystem()
            print("⚙️ Sistema configurado com parâmetros padrão")
        
        # Executa sistema
        success = system.run()
        
        if success:
            print("\n🎉 Sistema executado com sucesso!")
        else:
            print("\n❌ Erro na execução do sistema")
            
    except KeyboardInterrupt:
        print("\n⏹️ Operação cancelada pelo usuário")
    except Exception as e:
        print(f"\n❌ Erro: {e}")
        print("Verifique se todas as dependências estão instaladas:")
        print("pip install opencv-python mediapipe dlib numpy scikit-learn")


if __name__ == "__main__":
    main()