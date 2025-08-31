import cv2
import numpy as np
import urllib.request
import os
import dlib
import math
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("MediaPipe não disponível. Instale com: pip install mediapipe")


class OptimizedGazeTracker:
    def __init__(self):
        if not MEDIAPIPE_AVAILABLE:
            raise ImportError("MediaPipe é necessário para este sistema.")
        
        # URLs dos arquivos do modelo DNN
        self.prototxt_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
        self.model_url = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
        self.landmarks_url = "https://github.com/italojs/facial-landmarks-recognition/raw/master/shape_predictor_68_face_landmarks.dat"
        
        # Nomes dos arquivos locais
        self.prototxt_path = "deploy.prototxt"
        self.model_path = "res10_300x300_ssd_iter_140000.caffemodel"
        self.landmarks_path = "shape_predictor_68_face_landmarks.dat"
        
        # Baixa os arquivos se necessário
        self.download_model_files()
        
        # Carrega o modelo DNN
        self.net = cv2.dnn.readNetFromCaffe(self.prototxt_path, self.model_path)
        
        # Carrega o preditor de landmarks faciais
        self.predictor = dlib.shape_predictor(self.landmarks_path)
        
        # Inicializa MediaPipe para detecção de íris
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # Configurações do modelo DNN
        self.confidence_threshold = 0.5
        self.input_size = (300, 300)
        
        # Índices dos landmarks para os olhos (padrão 68 pontos dlib)
        self.LEFT_EYE_POINTS = list(range(36, 42))
        self.RIGHT_EYE_POINTS = list(range(42, 48))
        
        # Índices dos landmarks da íris no MediaPipe
        self.LEFT_IRIS_LANDMARKS = [474, 475, 476, 477]
        self.RIGHT_IRIS_LANDMARKS = [469, 470, 471, 472]
        
        # Sistema de Calibração de Tela baseado no artigo
        self.calibration_system = ScreenCalibrationSystem()
        
        # Flags para controle de visualização
        self.show_landmarks = True
        self.show_iris_detection = True
        
        # Estado de calibração
        self.is_calibrating = False
        self.calibration_points = []
        self.current_calibration_point = None
        self.calibration_gaze_vectors = []
        
        # Histórico para suavização
        self.gaze_history = {'left': [], 'right': []}
        self.history_size = 5
        
        # Parâmetros de tela (serão configurados durante calibração)
        self.screen_width = 1920
        self.screen_height = 1080
        
        # Inicializa a captura da webcam
        self.cap = cv2.VideoCapture(0)
        
        if not self.cap.isOpened():
            print("Erro: Não foi possível abrir a webcam")
            return

    def download_model_files(self):
        # Baixa os arquivos do modelo DNN e landmarks se não existirem
        files_to_download = [
            (self.prototxt_url, self.prototxt_path),
            (self.model_url, self.model_path),
            (self.landmarks_url, self.landmarks_path)
        ]
        
        for url, filename in files_to_download:
            if not os.path.exists(filename):
                print(f"Baixando {filename}...")
                try:
                    urllib.request.urlretrieve(url, filename)
                    print(f"{filename} baixado com sucesso!")
                except Exception as e:
                    print(f"Erro ao baixar {filename}: {e}")
            else:
                print(f"{filename} já existe.")
    
    def detect_faces_dnn(self, frame):
        # Detecta rostos usando DNN otimizado
        (h, w) = frame.shape[:2]
        
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 
            1.0,
            (300, 300), 
            (104.0, 177.0, 123.0)
        )
        
        self.net.setInput(blob)
        detections = self.net.forward()
        
        faces = []
        confidences = []
        
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence > self.confidence_threshold:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x, y, x1, y1) = box.astype("int")
                
                x = max(0, x)
                y = max(0, y)
                x1 = min(w, x1)
                y1 = min(h, y1)
                
                width = x1 - x
                height = y1 - y
                
                if width > 0 and height > 0:
                    faces.append((x, y, width, height))
                    confidences.append(float(confidence))
        
        return faces, confidences
    
    def get_landmarks(self, gray_frame, face_rect):
        # Obtém os 68 landmarks faciais para um rosto detectado
        x, y, w, h = face_rect
        dlib_rect = dlib.rectangle(x, y, x + w, y + h)
        landmarks = self.predictor(gray_frame, dlib_rect)
        
        landmarks_points = []
        for i in range(68):
            x_point = landmarks.part(i).x
            y_point = landmarks.part(i).y
            landmarks_points.append((x_point, y_point))
        
        return np.array(landmarks_points)

    def detect_iris_mediapipe(self, full_frame):
        # Detecta íris usando MediaPipe
        try:
            rgb_frame = cv2.cvtColor(full_frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)
            
            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                h, w = full_frame.shape[:2]
                
                iris_data = {}
                
                left_iris_points = []
                for idx in self.LEFT_IRIS_LANDMARKS:
                    if idx < len(face_landmarks.landmark):
                        landmark = face_landmarks.landmark[idx]
                        x = int(landmark.x * w)
                        y = int(landmark.y * h)
                        left_iris_points.append((x, y))
                
                if len(left_iris_points) >= 3:
                    left_points = np.array(left_iris_points)
                    center_x = int(np.mean(left_points[:, 0]))
                    center_y = int(np.mean(left_points[:, 1]))
                    distances = np.sqrt(np.sum((left_points - [center_x, center_y])**2, axis=1))
                    radius = max(8, int(np.mean(distances) * 1.8))
                    
                    iris_data['left'] = {
                        'center': (center_x, center_y),
                        'radius': radius,
                        'points': left_iris_points
                    }
                
                right_iris_points = []
                for idx in self.RIGHT_IRIS_LANDMARKS:
                    if idx < len(face_landmarks.landmark):
                        landmark = face_landmarks.landmark[idx]
                        x = int(landmark.x * w)
                        y = int(landmark.y * h)
                        right_iris_points.append((x, y))
                
                if len(right_iris_points) >= 3:
                    right_points = np.array(right_iris_points)
                    center_x = int(np.mean(right_points[:, 0]))
                    center_y = int(np.mean(right_points[:, 1]))
                    distances = np.sqrt(np.sum((right_points - [center_x, center_y])**2, axis=1))
                    radius = max(8, int(np.mean(distances) * 1.8))
                    
                    iris_data['right'] = {
                        'center': (center_x, center_y),
                        'radius': radius,
                        'points': right_iris_points
                    }
                
                return iris_data
                    
        except Exception as e:
            print(f"Erro no MediaPipe: {e}")
        
        return {}
    
    def calculate_3d_gaze_vector_simple(self, iris_data, landmarks, frame):
        """
        [Método original]
        Calcula vetor de olhar 3D com base na posição da íris e do olho.
        """
        h, w = frame.shape[:2]
        focal_length = w
        
        gaze_vectors = {}
        
        for eye_side in ['left', 'right']:
            if eye_side in iris_data and landmarks is not None:
                if eye_side == 'left':
                    eye_points = self.LEFT_EYE_POINTS
                else:
                    eye_points = self.RIGHT_EYE_POINTS
                
                eye_landmarks = landmarks[eye_points]
                eye_center = np.mean(eye_landmarks, axis=0)
                
                iris_center = np.array(iris_data[eye_side]['center'])
                
                iris_displacement = iris_center - eye_center
                
                if np.linalg.norm(iris_displacement) > 0:
                    gaze_x = iris_displacement[0] / focal_length
                    gaze_y = iris_displacement[1] / focal_length
                    gaze_z = 1.0 
                    
                    gaze_vector_3d = np.array([gaze_x, gaze_y, gaze_z])
                    gaze_vector_3d = gaze_vector_3d / np.linalg.norm(gaze_vector_3d)
                    
                    yaw = np.arctan2(gaze_vector_3d[0], gaze_vector_3d[2])
                    pitch = np.arctan2(-gaze_vector_3d[1], np.sqrt(gaze_vector_3d[0]**2 + gaze_vector_3d[2]**2))
                    
                    gaze_vectors[eye_side] = {
                        'vector_3d': gaze_vector_3d,
                        'yaw': yaw,
                        'pitch': pitch,
                        'eye_center': eye_center.astype(int),
                        'iris_center': iris_center.astype(int)
                    }
        
        return gaze_vectors

    def start_calibration(self):
        # Inicia o processo de calibração da tela
        self.is_calibrating = True
        self.calibration_points = []
        self.calibration_gaze_vectors = []
        
        margin = 0.15
        grid_points = []
        for y in np.linspace(margin, 1-margin, 3):
            for x in np.linspace(margin, 1-margin, 3):
                screen_x = int(x * self.screen_width)
                screen_y = int(y * self.screen_height)
                grid_points.append((screen_x, screen_y))
        
        self.calibration_points = grid_points
        self.current_calibration_point = 0
        
        print(f"Iniciando calibração com {len(self.calibration_points)} pontos")
        print("Olhe para o ponto vermelho e pressione SPACE para confirmar")
    
    def add_calibration_data(self, gaze_vectors):
        # Adiciona dados de calibração
        if self.is_calibrating and self.current_calibration_point is not None:
            if self.current_calibration_point < len(self.calibration_points):
                if 'left' in gaze_vectors and 'right' in gaze_vectors:
                    avg_yaw = (gaze_vectors['left']['yaw'] + gaze_vectors['right']['yaw']) / 2
                    avg_pitch = (gaze_vectors['left']['pitch'] + gaze_vectors['right']['pitch']) / 2
                    
                    screen_point = self.calibration_points[self.current_calibration_point]
                    
                    self.calibration_gaze_vectors.append({
                        'yaw': avg_yaw,
                        'pitch': avg_pitch,
                        'screen_x': screen_point[0],
                        'screen_y': screen_point[1]
                    })
                    
                    print(f"Ponto de calibração {self.current_calibration_point + 1}/{len(self.calibration_points)} coletado")
                    self.current_calibration_point += 1
                    
                    if self.current_calibration_point >= len(self.calibration_points):
                        self.finish_calibration()
    
    def finish_calibration(self):
        # Finaliza a calibração e treina os modelos
        self.is_calibrating = False
        self.current_calibration_point = None
        
        if len(self.calibration_gaze_vectors) >= 4:
            print("Treinando modelos de calibração...")
            self.calibration_system.train_calibration_models(self.calibration_gaze_vectors)
            print("Calibração concluída!")
        else:
            print("Dados insuficientes para calibração")
    
    def predict_screen_point(self, gaze_vectors):
        # Prediz ponto na tela usando o sistema de calibração
        if not self.calibration_system.is_calibrated():
            return None
        
        if 'left' in gaze_vectors and 'right' in gaze_vectors:
            avg_yaw = (gaze_vectors['left']['yaw'] + gaze_vectors['right']['yaw']) / 2
            avg_pitch = (gaze_vectors['left']['pitch'] + gaze_vectors['right']['pitch']) / 2
            
            return self.calibration_system.predict_screen_point(avg_yaw, avg_pitch)
        
        return None
    
    def apply_temporal_smoothing(self, gaze_vectors):
        # Aplica suavização temporal aos vetores de olhar.
        for eye_side in ['left', 'right']:
            if eye_side in gaze_vectors:
                self.gaze_history[eye_side].append(gaze_vectors[eye_side])
                if len(self.gaze_history[eye_side]) > self.history_size:
                    self.gaze_history[eye_side].pop(0)

        smoothed_vectors = {}
        for eye_side in ['left', 'right']:
            if len(self.gaze_history[eye_side]) > 0:
                recent_data = self.gaze_history[eye_side]
                
                # Suavização usando média ponderada (mais peso para dados recentes)
                weights = np.exp(np.linspace(-1, 0, len(recent_data)))
                weights /= np.sum(weights)

                yaw_values = [data['yaw'] for data in recent_data]
                pitch_values = [data['pitch'] for data in recent_data]
                
                smoothed_yaw = np.average(yaw_values, weights=weights)
                smoothed_pitch = np.average(pitch_values, weights=weights)
                
                # Reconstrói vetor 3D a partir dos ângulos suavizados
                smoothed_vector_3d = np.array([
                    np.sin(smoothed_yaw) * np.cos(smoothed_pitch),
                    -np.sin(smoothed_pitch),
                    np.cos(smoothed_yaw) * np.cos(smoothed_pitch)
                ])

                # Retorna o resultado suavizado
                smoothed_vectors[eye_side] = {
                    'vector_3d': smoothed_vector_3d,
                    'yaw': float(smoothed_yaw),
                    'pitch': float(smoothed_pitch),
                    'eye_center': recent_data[-1]['eye_center'],
                    'iris_center': recent_data[-1]['iris_center']
                }

        return smoothed_vectors

    def draw_visualization(self, frame, faces, iris_data, gaze_vectors, landmarks=None):
        # Desenha visualizações na tela
        
        for i, (x, y, w, h) in enumerate(faces):
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"Face {i+1}", (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        if self.show_landmarks and landmarks is not None:
            for i, (x, y) in enumerate(landmarks):
                color = (0, 255, 255)
                if i in self.LEFT_EYE_POINTS:
                    color = (0, 255, 0)
                elif i in self.RIGHT_EYE_POINTS:
                    color = (255, 0, 0)
                cv2.circle(frame, (x, y), 1, color, -1)
        
        if self.show_iris_detection and iris_data:
            for eye_side, data in iris_data.items():
                center = data['center']
                radius = data['radius']
                color = (0, 255, 0) if eye_side == 'left' else (255, 0, 0)
                cv2.circle(frame, tuple(map(int, center)), int(radius), color, 2)
                cv2.circle(frame, tuple(map(int, center)), 3, color, -1)

        if gaze_vectors:
            for eye_side, data in gaze_vectors.items():
                eye_center = data['eye_center']
                iris_center = data['iris_center']
                color = (0, 255, 0) if eye_side == 'left' else (255, 0, 0)
                
                cv2.line(frame, tuple(map(int, eye_center)), tuple(map(int, iris_center)), color, 2)
                
                scale = 50
                end_point = (
                    iris_center[0] + int(data['vector_3d'][0] * scale),
                    iris_center[1] + int(data['vector_3d'][1] * scale)
                )
                cv2.arrowedLine(frame, tuple(map(int, iris_center)), tuple(map(int, end_point)), (0, 255, 255), 2)
        
        if self.is_calibrating and self.current_calibration_point is not None:
            if self.current_calibration_point < len(self.calibration_points):
                point = self.calibration_points[self.current_calibration_point]
                frame_x = int(point[0] * frame.shape[1] / self.screen_width)
                frame_y = int(point[1] * frame.shape[0] / self.screen_height)
                
                cv2.circle(frame, (frame_x, frame_y), 15, (0, 0, 255), -1)
                cv2.putText(frame, f"Olhe aqui e pressione SPACE", 
                           (frame_x - 100, frame_y - 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                cv2.putText(frame, f"Ponto {self.current_calibration_point + 1}/{len(self.calibration_points)}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        elif self.calibration_system.is_calibrated() and gaze_vectors:
            screen_point = self.predict_screen_point(gaze_vectors)
            if screen_point is not None:
                frame_x = int(screen_point[0] * frame.shape[1] / self.screen_width)
                frame_y = int(screen_point[1] * frame.shape[0] / self.screen_height)
                
                frame_x = np.clip(frame_x, 0, frame.shape[1] - 1)
                frame_y = np.clip(frame_y, 0, frame.shape[0] - 1)
                
                cv2.circle(frame, (frame_x, frame_y), 10, (255, 0, 255), -1)
                cv2.putText(frame, "Olhar Predito", (frame_x + 15, frame_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
        
        return frame
    
    def run(self):
        #Loop principal
        frame_count = 0
        
        while True:
            ret, frame = self.cap.read()
            
            if not ret:
                print("Erro: Não foi possível capturar o frame")
                break
            
            frame_count += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            faces, confidences = self.detect_faces_dnn(frame)
            
            landmarks = None
            gaze_vectors = {}
            
            if faces:
                face = faces[0]
                
                try:
                    landmarks = self.get_landmarks(gray, face)
                except:
                    landmarks = None

                if landmarks is not None:
                    iris_data = self.detect_iris_mediapipe(frame)
                    
                    if iris_data:
                        gaze_vectors = self.calculate_3d_gaze_vector_simple(iris_data, landmarks, frame)
                        gaze_vectors = self.apply_temporal_smoothing(gaze_vectors)

            frame = self.draw_visualization(frame, faces, iris_data, gaze_vectors, landmarks)
            
            info_lines = [
                f'Rostos: {len(faces)} | Frame: {frame_count}',
                f'Calibração: {"ON" if self.is_calibrating else "Concluída" if self.calibration_system.is_calibrated() else "Necessária"}',
                f'Método: {self.calibration_system.get_current_method()}'
            ]
            
            if self.calibration_system.is_calibrated():
                accuracy = self.calibration_system.evaluate_calibration_accuracy()
                if accuracy:
                    info_lines.append(f'Erro médio: {accuracy["mean_error"]:.1f}px')
            
            for i, line in enumerate(info_lines):
                cv2.putText(frame, line, (10, frame.shape[0] - 80 + i * 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow('Gaze Tracker Otimizado', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                self.start_calibration()
            elif key == ord(' '):
                if self.is_calibrating and gaze_vectors:
                    self.add_calibration_data(gaze_vectors)
            elif key == ord('r'):
                self.calibration_system.reset()
                print("Calibração resetada")
            elif key == ord('l'):
                self.show_landmarks = not self.show_landmarks
            elif key == ord('i'):
                self.show_iris_detection = not self.show_iris_detection
            
            self.cap.release()
            cv2.destroyAllWindows()


class ScreenCalibrationSystem:
    def __init__(self):
        self.calibration_method = "hybrid"
        self.is_trained = False
        self.ml_model_x = Ridge(alpha=1.0)
        self.ml_model_y = Ridge(alpha=1.0)
        self.scaler = StandardScaler()
        self.geometric_params = None
        self.calibration_data = []
    
    def train_calibration_models(self, calibration_data):
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
        X_scaled = self.scaler.fit_transform(X)
        self.ml_model_x.fit(X_scaled, y_screen_x)
        self.ml_model_y.fit(X_scaled, y_screen_y)
        print("Modelo ML treinado")
    
    def _train_geometric_model(self, calibration_data):
        X = np.array([[d['yaw'], d['pitch'], 1] for d in calibration_data])
        y_x = np.array([d['screen_x'] for d in calibration_data])
        y_y = np.array([d['screen_y'] for d in calibration_data])
        params_x = np.linalg.lstsq(X, y_x, rcond=None)[0]
        params_y = np.linalg.lstsq(X, y_y, rcond=None)[0]
        self.geometric_params = {'x_params': params_x, 'y_params': params_y}
        print("Modelo Geométrico treinado")
    
    def _train_hybrid_model(self, X, y_screen_x, y_screen_y):
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
    
    def predict_screen_point(self, yaw, pitch):
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
    
    def evaluate_calibration_accuracy(self):
        if not self.is_trained or not self.calibration_data:
            return None
        errors = []
        for data in self.calibration_data:
            pred = self.predict_screen_point(data['yaw'], data['pitch'])
            if pred:
                error = np.sqrt((pred[0] - data['screen_x'])**2 + (pred[1] - data['screen_y'])**2)
                errors.append(error)
        if errors:
            return {'mean_error': np.mean(errors), 'std_error': np.std(errors), 'max_error': np.max(errors), 'min_error': np.min(errors)}
        return None
    
    def switch_calibration_method(self, method):
        if method in ["geometric", "ml", "hybrid"]:
            self.calibration_method = method
            if self.calibration_data:
                self.train_calibration_models(self.calibration_data)
            print(f"Método alterado para: {method}")
            return True
        return False
    
    def get_current_method(self):
        return self.calibration_method.upper()
    
    def is_calibrated(self):
        return self.is_trained
    
    def reset(self):
        self.is_trained = False
        self.calibration_data = []
        self.geometric_params = None
        print("Sistema de calibração resetado")


if __name__ == "__main__":
    try:
        print("Sistema Gaze Tracking")
        
        tracker = OptimizedGazeTracker()
        
        print("=== Instruções de Uso ===")
        print("1. Pressione 'c' para iniciar calibração")
        print("2. Olhe para cada ponto vermelho e pressione SPACE")
        print("3. Após calibração, o sistema predirá onde você está olhando")
        print("4. Use 'r' para resetar e recalibrar se necessário")
        print()
        print("Controles adicionais:")
        print("- 'l': alternar landmarks")
        print("- 'i': alternar detecção de íris")
        print("- 'q': sair")
        print()
        
        def enhanced_run():
            frame_count = 0
            
            while True:
                ret, frame = tracker.cap.read()
                
                if not ret:
                    print("Erro: Não foi possível capturar o frame")
                    break
                
                frame_count += 1
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                faces, confidences = tracker.detect_faces_dnn(frame)
                
                landmarks = None
                gaze_vectors = {}
                
                if faces:
                    face = faces[0]
                    
                    try:
                        landmarks = tracker.get_landmarks(gray, face)
                    except:
                        landmarks = None

                    if landmarks is not None:
                        iris_data = tracker.detect_iris_mediapipe(frame)
                        
                        if iris_data:
                            gaze_vectors = tracker.calculate_3d_gaze_vector_simple(iris_data, landmarks, frame)
                            # Aplica a suavização temporal
                            gaze_vectors = tracker.apply_temporal_smoothing(gaze_vectors)

                frame = tracker.draw_visualization(frame, faces, iris_data, gaze_vectors, landmarks)
                
                info_lines = [
                    f'Rostos: {len(faces)} | Frame: {frame_count}',
                    f'Calibração: {"ON" if tracker.is_calibrating else "Concluída" if tracker.calibration_system.is_calibrated() else "Necessária"}',
                    f'Método: {tracker.calibration_system.get_current_method()}'
                ]
                
                if tracker.calibration_system.is_calibrated():
                    accuracy = tracker.calibration_system.evaluate_calibration_accuracy()
                    if accuracy:
                        info_lines.append(f'Erro médio: {accuracy["mean_error"]:.1f}px')
                
                for i, line in enumerate(info_lines):
                    cv2.putText(frame, line, (10, frame.shape[0] - 80 + i * 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                cv2.imshow('Gaze Tracker Otimizado', frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('c'):
                    tracker.start_calibration()
                elif key == ord(' '):
                    if tracker.is_calibrating and gaze_vectors:
                        tracker.add_calibration_data(gaze_vectors)
                elif key == ord('r'):
                    tracker.calibration_system.reset()
                    print("Calibração resetada")
                elif key == ord('l'):
                    tracker.show_landmarks = not tracker.show_landmarks
                elif key == ord('i'):
                    tracker.show_iris_detection = not tracker.show_iris_detection
            
            tracker.cap.release()
            cv2.destroyAllWindows()
            
        enhanced_run()
        
    except ImportError as e:
        if 'mediapipe' in str(e):
            print("Erro: MediaPipe não está instalado!")
            print("Instale com: pip install mediapipe")
        elif 'dlib' in str(e):
            print("Erro: dlib não está instalado!")
            print("Instale com: pip install dlib")
        elif 'sklearn' in str(e):
            print("Erro: scikit-learn não está instalado!")
            print("Instale com: pip install scikit-learn")
        else:
            print(f"Erro de importação: {e}")
    except Exception as e:
        print(f"Erro: {e}")
        print("\nVerifique se todas as dependências estão instaladas:")
        print("pip install opencv-python mediapipe dlib numpy scikit-learn")
