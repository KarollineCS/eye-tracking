import cv2
import numpy as np
import urllib.request
import os
import dlib
import math
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("MediaPipe não disponível. Instale com: pip install mediapipe")

class GazeTracker:
    def __init__(self):
        if not MEDIAPIPE_AVAILABLE:
            raise ImportError("MediaPipe é necessário para este sistema. Instale com: pip install mediapipe")
        
        # URLs dos arquivos do modelo DNN
        self.prototxt_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
        self.model_url = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
        
        # URL do modelo de landmarks faciais
        self.landmarks_url = "https://github.com/italojs/facial-landmarks-recognition/raw/master/shape_predictor_68_face_landmarks.dat"
        
        # Nomes dos arquivos locais
        self.prototxt_path = "deploy.prototxt"
        self.model_path = "res10_300x300_ssd_iter_140000.caffemodel"
        self.landmarks_path = "shape_predictor_68_face_landmarks.dat"
        
        # Baixa os arquivos se necessário
        self.download_model_files()
        
        # Carrega o modelo DNN
        print("Carregando modelo DNN...")
        self.net = cv2.dnn.readNetFromCaffe(self.prototxt_path, self.model_path)
        
        # Carrega o preditor de landmarks faciais
        print("Carregando preditor de landmarks...")
        self.predictor = dlib.shape_predictor(self.landmarks_path)
        
        # Inicializa MediaPipe para detecção de íris
        print("Inicializando MediaPipe...")
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
        self.LEFT_EYE_POINTS = list(range(36, 42))   # Olho esquerdo
        self.RIGHT_EYE_POINTS = list(range(42, 48))  # Olho direito
        
        # Pontos de referência para pose da cabeça (modelo 3D simplificado)
        self.model_points_3d = np.array([
            (0.0, 0.0, 0.0),             # Ponta do nariz
            (0.0, -330.0, -65.0),        # Queixo
            (-225.0, 170.0, -135.0),     # Canto esquerdo do olho esquerdo
            (225.0, 170.0, -135.0),      # Canto direito do olho direito
            (-150.0, -150.0, -125.0),    # Canto esquerdo da boca
            (150.0, -150.0, -125.0)      # Canto direito da boca
        ], dtype=np.float64)
        
        # Índices dos landmarks correspondentes no modelo 68 pontos
        self.pose_landmarks_2d = [30, 8, 36, 45, 48, 54]  # nariz, queixo, olhos, boca
        
        # Índices dos landmarks da íris no MediaPipe (468 pontos)
        self.LEFT_IRIS_LANDMARKS = [474, 475, 476, 477]   # Íris olho esquerdo
        self.RIGHT_IRIS_LANDMARKS = [469, 470, 471, 472]  # Íris olho direito
        
        # Flags para controle de visualização
        self.show_landmarks = True
        self.show_eye_regions = True
        self.show_iris_detection = True
        self.show_pose_estimation = True
        self.show_normalized_eyes = True
        
        # Parâmetros de normalização
        self.normalized_eye_size = (60, 30)  # Tamanho normalizado dos olhos
        self.pose_compensation_factor = 0.7   # Fator de compensação de pose
        
        # Histórico para suavização
        self.gaze_history = {'left': [], 'right': []}
        self.history_size = 5
        
        # Inicializa a captura da webcam
        self.cap = cv2.VideoCapture(0)
        
        if not self.cap.isOpened():
            print("Erro: Não foi possível abrir a webcam")
            return
            
        print("Sistema Gaze Tracker com Normalização inicializado!")
        print("Controles:")
        print("  q: sair")
        print("  +/-: ajustar sensibilidade de detecção de rosto")
        print("  l: alternar landmarks")
        print("  r: alternar regiões dos olhos")  
        print("  i: alternar detecção de íris")
        print("  p: alternar estimação de pose")
        print("  n: alternar olhos normalizados")
        
    def download_model_files(self):
        """
        Baixa os arquivos do modelo DNN e landmarks se não existirem
        """
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
                    print("Você pode baixar manualmente de:")
                    print(f"  {url}")
            else:
                print(f"{filename} já existe.")
    
    def detect_faces_dnn(self, frame):
        """
        Detecta rostos usando DNN
        """
        (h, w) = frame.shape[:2]
        
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, self.input_size), 
            1.0,
            self.input_size, 
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
        """
        Obtém os 68 landmarks faciais para um rosto detectado
        """
        x, y, w, h = face_rect
        dlib_rect = dlib.rectangle(x, y, x + w, y + h)
        landmarks = self.predictor(gray_frame, dlib_rect)
        
        landmarks_points = []
        for i in range(68):
            x_point = landmarks.part(i).x
            y_point = landmarks.part(i).y
            landmarks_points.append((x_point, y_point))
        
        return np.array(landmarks_points)
    
    def estimate_head_pose(self, landmarks, camera_matrix, dist_coeffs):
        """
        Estima a pose da cabeça usando solvePnP com modelo 3D melhorado
        """
        # Modelo 3D mais preciso da cabeça (em mm)
        model_points_3d = np.array([
            (0.0, 0.0, 0.0),             # Ponta do nariz (30)
            (0.0, -330.0, -65.0),        # Queixo (8)
            (-225.0, 170.0, -135.0),     # Canto externo olho esquerdo (36)
            (225.0, 170.0, -135.0),      # Canto externo olho direito (45)
            (-150.0, -150.0, -125.0),    # Canto esquerdo da boca (48)
            (150.0, -150.0, -125.0),     # Canto direito da boca (54)
            (0.0, 170.0, -135.0),        # Centro entre os olhos (27 - ponte do nariz)
        ], dtype=np.float64)
        
        # Índices dos landmarks correspondentes (68 pontos dlib)
        pose_landmarks_indices = [30, 8, 36, 45, 48, 54, 27]
        
        # Verifica se todos os landmarks necessários existem
        if len(landmarks) < max(pose_landmarks_indices) + 1:
            return None, None
        
        # Pontos 2D correspondentes
        image_points = np.array([
            landmarks[idx] for idx in pose_landmarks_indices
        ], dtype=np.float64)
        
        # Validação dos pontos
        for point in image_points:
            if point[0] < 0 or point[1] < 0:
                return None, None
        
        try:
            # Resolve o problema PnP
            success, rotation_vector, translation_vector = cv2.solvePnP(
                model_points_3d,
                image_points,
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE  # Método mais preciso
            )
            
            if success:
                return rotation_vector, translation_vector
            else:
                return None, None
                
        except Exception as e:
            print(f"Erro na estimação de pose: {e}")
            return None, None
    
    def normalize_illumination(self, eye_region):
        """
        Normaliza a iluminação da região do olho usando múltiplas técnicas
        """
        if eye_region.size == 0:
            return eye_region
            
        # Converte para escala de cinza se necessário
        if len(eye_region.shape) == 3:
            gray_eye = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
        else:
            gray_eye = eye_region.copy()
        
        # 1. Equalização de histograma adaptativa (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        normalized = clahe.apply(gray_eye)
        
        # 2. Correção gamma adaptativa
        mean_val = np.mean(normalized)
        target_mean = 128  # Valor alvo
        gamma = math.log(target_mean / 255.0) / math.log(mean_val / 255.0) if mean_val > 0 else 1.0
        gamma = np.clip(gamma, 0.5, 2.0)  # Limita gamma
        
        # Aplica correção gamma
        gamma_corrected = np.power(normalized / 255.0, gamma) * 255.0
        gamma_corrected = np.uint8(np.clip(gamma_corrected, 0, 255))
        
        # 3. Filtragem bilateral para reduzir ruído mantendo bordas
        bilateral = cv2.bilateralFilter(gamma_corrected, 5, 80, 80)
        
        # 4. Normalização final por desvio padrão local
        kernel = cv2.getGaussianKernel(9, 1.5)
        kernel = kernel @ kernel.T
        
        local_mean = cv2.filter2D(bilateral.astype(np.float32), -1, kernel)
        local_var = cv2.filter2D((bilateral.astype(np.float32) - local_mean) ** 2, -1, kernel)
        local_std = np.sqrt(local_var + 1e-6)
        
        # Normalização Z-score local
        normalized_final = (bilateral.astype(np.float32) - local_mean) / (local_std + 1e-6)
        normalized_final = np.clip(normalized_final * 50 + 128, 0, 255).astype(np.uint8)
        
        return normalized_final
    
    def normalize_eye_pose(self, eye_region, eye_landmarks, rotation_vector, translation_vector):
        """
        Normaliza a pose do olho compensando rotação da cabeça
        """
        if rotation_vector is None or eye_region.size == 0:
            return eye_region, eye_landmarks
        
        # Converte vetor de rotação para matriz de rotação
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        
        # Calcula ângulos de Euler
        angles = self.rotation_matrix_to_euler_angles(rotation_matrix)
        yaw, pitch, roll = angles
        
        # Cria matriz de transformação 2D para compensar rotação
        center = (eye_region.shape[1] // 2, eye_region.shape[0] // 2)
        
        # Compensa principalmente o roll (rotação no plano da imagem)
        compensation_angle = -roll * self.pose_compensation_factor
        transform_matrix = cv2.getRotationMatrix2D(center, np.degrees(compensation_angle), 1.0)
        
        # Aplica transformação
        normalized_eye = cv2.warpAffine(
            eye_region, 
            transform_matrix, 
            (eye_region.shape[1], eye_region.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT
        )
        
        # Transforma landmarks do olho
        if len(eye_landmarks) > 0:
            # Converte landmarks para coordenadas homogêneas
            landmarks_homo = np.hstack([eye_landmarks, np.ones((eye_landmarks.shape[0], 1))])
            
            # Aplica transformação
            transformed_landmarks = transform_matrix @ landmarks_homo.T
            normalized_landmarks = transformed_landmarks.T[:, :2].astype(np.int32)
        else:
            normalized_landmarks = eye_landmarks
        
        return normalized_eye, normalized_landmarks
    
    def rotation_matrix_to_euler_angles(self, R):
        """
        Converte matriz de rotação para ângulos de Euler (mais estável)
        """
        # Verifica se a matriz é válida
        if R.shape != (3, 3):
            return np.array([0, 0, 0])
        
        # Calcula os ângulos de Euler usando uma sequência XYZ mais estável
        sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        
        singular = sy < 1e-6
        
        if not singular:
            # Caso geral
            x = math.atan2(R[2, 1], R[2, 2])    # Roll
            y = math.atan2(-R[2, 0], sy)        # Pitch
            z = math.atan2(R[1, 0], R[0, 0])    # Yaw
        else:
            # Caso singular (gimbal lock)
            x = math.atan2(-R[1, 2], R[1, 1])   # Roll
            y = math.atan2(-R[2, 0], sy)        # Pitch
            z = 0                               # Yaw (indefinido)
        
        # Limita os ângulos para evitar saltos
        x = np.clip(x, -np.pi/2, np.pi/2)
        y = np.clip(y, -np.pi/2, np.pi/2)
        z = np.clip(z, -np.pi, np.pi)
        
        return np.array([z, y, x])  # Retorna [yaw, pitch, roll]
    
    def extract_eye_region(self, frame, landmarks, eye_points, padding=15):
        """
        Extrai a região do olho com base nos landmarks
        """
        eye_landmarks = landmarks[eye_points]
        
        x_min = np.min(eye_landmarks[:, 0]) - padding
        x_max = np.max(eye_landmarks[:, 0]) + padding
        y_min = np.min(eye_landmarks[:, 1]) - padding
        y_max = np.max(eye_landmarks[:, 1]) + padding
        
        # Garante que está dentro da imagem
        h, w = frame.shape[:2]
        x_min = max(0, x_min)
        x_max = min(w, x_max)
        y_min = max(0, y_min)
        y_max = min(h, y_max)
        
        eye_region = frame[y_min:y_max, x_min:x_max]
        
        # Ajusta coordenadas dos landmarks para a região extraída
        local_landmarks = eye_landmarks.copy()
        local_landmarks[:, 0] -= x_min
        local_landmarks[:, 1] -= y_min
        
        return eye_region, (x_min, y_min, x_max - x_min, y_max - y_min), local_landmarks
    
    def detect_iris_mediapipe(self, full_frame):
        """
        Detecta íris usando MediaPipe em todo o frame
        """
        try:
            rgb_frame = cv2.cvtColor(full_frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)
            
            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                h, w = full_frame.shape[:2]
                
                # Extrai coordenadas das íris
                left_iris_points = []
                right_iris_points = []
                
                # Íris esquerda
                for idx in self.LEFT_IRIS_LANDMARKS:
                    if idx < len(face_landmarks.landmark):
                        landmark = face_landmarks.landmark[idx]
                        x = int(landmark.x * w)
                        y = int(landmark.y * h)
                        left_iris_points.append((x, y))
                
                # Íris direita
                for idx in self.RIGHT_IRIS_LANDMARKS:
                    if idx < len(face_landmarks.landmark):
                        landmark = face_landmarks.landmark[idx]
                        x = int(landmark.x * w)
                        y = int(landmark.y * h)
                        right_iris_points.append((x, y))
                
                # Calcula centro e raio de cada íris
                iris_data = {}
                
                if len(left_iris_points) >= 3:
                    left_points = np.array(left_iris_points)
                    center_x = int(np.mean(left_points[:, 0]))
                    center_y = int(np.mean(left_points[:, 1]))
                    
                    # Calcula raio baseado na dispersão dos pontos
                    distances = np.sqrt(np.sum((left_points - [center_x, center_y])**2, axis=1))
                    radius = max(8, int(np.mean(distances) * 1.8))
                    
                    iris_data['left'] = {
                        'center': (center_x, center_y),
                        'radius': radius,
                        'points': left_iris_points
                    }
                
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
    
    def detect_pupil_from_normalized_image_advanced(self, normalized_eye_image):
        """
        Detecta a pupila em uma imagem de olho já normalizada para iluminação,
        usando uma análise avançada de contornos com múltiplos critérios.
        
        Parâmetros:
        - normalized_eye_image: Imagem do olho em escala de cinza já normalizada.
        
        Retorna:
        - tuple: (x, y, raio) da pupila, ou None se não for detectada.
        """
        if normalized_eye_image.size == 0 or normalized_eye_image.shape[0] < 10 or normalized_eye_image.shape[1] < 10:
            return None
        
        # 1. Limiarização binária adaptativa
        # A limiarização adaptativa ajuda a lidar com variações de luz que ainda possam existir.
        binary = cv2.adaptiveThreshold(
            normalized_eye_image, 
            255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 
            11, # Block size
            2   # C
        )
        
        # 2. Operações morfológicas para preencher buracos e remover ruído
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # 3. Análise de contornos
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
            
        # Variáveis de controle para encontrar o melhor contorno
        best_contour = None
        best_score = 0
        
        # Calcula área mínima e máxima esperada da pupila
        eye_area = normalized_eye_image.shape[0] * normalized_eye_image.shape[1]
        min_area = eye_area * 0.05
        max_area = eye_area * 0.5
        
        # 4. Avalia cada contorno com múltiplos critérios
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filtra por área (evita ruídos muito pequenos e grandes demais)
            if min_area < area < max_area:
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    # Calcula circularidade: 1.0 para um círculo perfeito
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    
                    # Calcula solidez: área do contorno / área do hull convexo
                    hull = cv2.convexHull(contour)
                    hull_area = cv2.contourArea(hull)
                    solidity = area / hull_area if hull_area > 0 else 0
                    
                    # Criterio de pontuação ponderado para encontrar a pupila
                    # A pupila é geralmente circular e sólida
                    score = circularity * 0.6 + solidity * 0.4
                    
                    if score > best_score:
                        best_score = score
                        best_contour = contour

        if best_contour is not None:
            # Encontra o círculo que melhor se ajusta ao melhor contorno
            (x, y), radius = cv2.minEnclosingCircle(best_contour)
            return (int(x), int(y), int(radius))
        
        return None

    def calculate_gaze_direction_normalized(self, iris_data, landmarks, rotation_vector, translation_vector, frame):
        """
        Calcula a direção do olhar com normalização de pose e um sistema de coordenadas relativo à anatomia do olho.
        """
        gaze_data = {}
        
        h, w = frame.shape[:2]
        focal_length = w
        camera_center = (w // 2, h // 2)
        camera_matrix = np.array([
            [focal_length, 0, camera_center[0]],
            [0, focal_length, camera_center[1]],
            [0, 0, 1]
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1))
        
        for eye_side in ['left', 'right']:
            # Verifica se os dados da íris e os landmarks do rosto estão disponíveis
            if eye_side in iris_data and landmarks is not None:
                # Seleciona os landmarks do olho e os cantos para o sistema de coordenadas
                if eye_side == 'left':
                    eye_points = self.LEFT_EYE_POINTS
                    # Canto externo e interno do olho esquerdo
                    eye_corner_outer = landmarks[36]
                    eye_corner_inner = landmarks[39]
                else:
                    eye_points = self.RIGHT_EYE_POINTS
                    # Canto interno e externo do olho direito
                    eye_corner_outer = landmarks[45]
                    eye_corner_inner = landmarks[42]

                # --- 1. DETECÇÃO DA ÍRIS ---
                # Posição da íris detectada pelo MediaPipe
                iris_center_global = iris_data[eye_side]['center']
                
                # --- 2. SISTEMA DE COORDENADAS ANATÔMICO ---
                # Cria um vetor que representa o eixo horizontal do olho
                eye_axis_vector = eye_corner_inner - eye_corner_outer
                eye_axis_length = np.linalg.norm(eye_axis_vector)
                
                # Projeta o centro da íris no eixo do olho
                iris_to_outer_corner = iris_center_global - eye_corner_outer
                
                # Calcula a posição relativa da íris no eixo horizontal (0 a 1)
                horizontal_position = np.dot(iris_to_outer_corner, eye_axis_vector) / (eye_axis_length**2)
                
                # Calcula a distância vertical da íris em relação ao eixo do olho
                vertical_position = np.linalg.norm(np.cross(iris_to_outer_corner, eye_axis_vector)) / eye_axis_length

                # Compensação de pose (adaptação da lógica existente)
                if rotation_vector is not None:
                    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
                    angles = self.rotation_matrix_to_euler_angles(rotation_matrix)
                    yaw, pitch, roll = angles
                    
                    # Compensa as posições calculadas com base na pose da cabeça
                    compensated_x = horizontal_position - (yaw * 0.1)  # Fator de ajuste
                    compensated_y = vertical_position - (pitch * 0.05) # Fator de ajuste
                    
                    final_vector = (compensated_x, compensated_y)
                else:
                    final_vector = (horizontal_position, vertical_position)
                    
                # Normaliza o vetor para suavização
                vector_length = np.sqrt(final_vector[0]**2 + final_vector[1]**2)
                if vector_length > 0:
                    normalized_vector = (final_vector[0] / vector_length, final_vector[1] / vector_length)
                else:
                    normalized_vector = (0, 0)
                
                # Suavização temporal
                self.gaze_history[eye_side].append(normalized_vector)
                if len(self.gaze_history[eye_side]) > self.history_size:
                    self.gaze_history[eye_side].pop(0)
                
                smoothed_vector = np.mean(self.gaze_history[eye_side], axis=0) if self.gaze_history[eye_side] else normalized_vector

                # Extrai dados adicionais para o dicionário
                eye_center_global = (int(np.mean(landmarks[eye_points][:,0])), int(np.mean(landmarks[eye_points][:,1])))
                
                gaze_data[eye_side] = {
                    'iris_center_global': iris_center_global,
                    'eye_center_global': eye_center_global,
                    'normalized_vector': tuple(smoothed_vector),
                    'gaze_vector': final_vector,
                    'pose_compensated': rotation_vector is not None
                }
        
        return gaze_data
    
    def get_gaze_direction_label(self, normalized_gaze_vector, threshold=0.3):
        """
        Traduz um vetor de olhar normalizado para uma direção textual.

        Parâmetros:
        - normalized_gaze_vector (tuple): O vetor de olhar (x, y) normalizado.
        - threshold (float): O limite para determinar se o olhar é significativo em uma direção.

        Retorna:
        - str: Um rótulo para a direção do olhar (e.g., 'Olhando para a Esquerda').
        """
        if normalized_gaze_vector is None:
            return "N/A"
        
        x, y = normalized_gaze_vector
        direction = ""

        # Determina a direção horizontal
        if x > threshold:
            direction += "Esquerda"
        elif x < -threshold:
            direction += "Direita"
        
        # Determina a direção vertical
        if y > threshold:
            direction += " Cima"
        elif y < -threshold:
            direction += " Baixo"
        
        # Se não houver direção significativa, o olhar é central
        if not direction:
            return "Centro"
        
        return f"Olhando para {direction.strip()}"

    def draw_landmarks(self, frame, landmarks):
        """
        Desenha landmarks faciais
        """
        for i, (x, y) in enumerate(landmarks):
            if i in self.LEFT_EYE_POINTS:
                color = (0, 255, 0)  # Verde para olho esquerdo
                radius = 2
            elif i in self.RIGHT_EYE_POINTS:
                color = (255, 0, 0)  # Azul para olho direito  
                radius = 2
            elif i in self.pose_landmarks_2d:
                color = (255, 255, 0)  # Ciano para pontos de pose
                radius = 3
            else:
                color = (0, 255, 255)  # Amarelo para outros pontos
                radius = 1
            
            cv2.circle(frame, (x, y), radius, color, -1)
        
        return frame
    
    def draw_head_pose(self, frame, landmarks, rotation_vector, translation_vector):
        """
        Desenha a estimação de pose da cabeça
        """
        if rotation_vector is None or translation_vector is None:
            return frame
        
        # Parâmetros da câmera melhorados
        h, w = frame.shape[:2]
        focal_length = max(w, h)  # Focal length mais realista
        camera_center = (w // 2, h // 2)
        camera_matrix = np.array([
            [focal_length, 0, camera_center[0]],
            [0, focal_length, camera_center[1]],
            [0, 0, 1]
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1))
        
        # Usa o ponto do nariz como origem dos eixos
        nose_tip = tuple(landmarks[30])  # Ponta do nariz
        
        # Pontos dos eixos 3D relativos à origem (nariz)
        axis_length = 80
        axis_points_3d = np.array([
            (0, 0, 0),              # Origem (nariz)
            (axis_length, 0, 0),    # Eixo X (direita da cabeça)
            (0, -axis_length, 0),   # Eixo Y (para cima da cabeça)
            (0, 0, -axis_length)    # Eixo Z (para frente da cabeça)
        ], dtype=np.float64)
        
        try:
            # Projeta pontos 3D para 2D
            projected_points, _ = cv2.projectPoints(
                axis_points_3d, rotation_vector, translation_vector, 
                camera_matrix, dist_coeffs
            )
            
            # Converte para inteiros
            projected_points = projected_points.reshape(-1, 2).astype(int)
            
            # Garante que os pontos estão dentro da tela
            for i, point in enumerate(projected_points):
                projected_points[i] = (
                    np.clip(point[0], 0, w-1),
                    np.clip(point[1], 0, h-1)
                )
            
            # Desenha eixos a partir do nariz
            origin = nose_tip
            x_axis = tuple(projected_points[1])
            y_axis = tuple(projected_points[2])
            z_axis = tuple(projected_points[3])
            
            # Calcula e mostra ângulos primeiro para debug
            rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
            angles = self.rotation_matrix_to_euler_angles(rotation_matrix)
            yaw, pitch, roll = np.degrees(angles)
            
            # Eixo X - Vermelho (esquerda/direita da cabeça)
            cv2.arrowedLine(frame, origin, x_axis, (0, 0, 255), 3, tipLength=0.2)
            cv2.putText(frame, 'X', x_axis, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # Eixo Y - Verde (cima/baixo da cabeça)
            cv2.arrowedLine(frame, origin, y_axis, (0, 255, 0), 3, tipLength=0.2)
            cv2.putText(frame, 'Y', y_axis, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Eixo Z - Azul (frente/trás da cabeça)
            cv2.arrowedLine(frame, origin, z_axis, (255, 0, 0), 3, tipLength=0.2)
            cv2.putText(frame, 'Z', z_axis, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            # Desenha ponto de origem
            cv2.circle(frame, origin, 5, (255, 255, 255), -1)
            cv2.circle(frame, origin, 5, (0, 0, 0), 2)
            
            # Mostra ângulos na tela com mais informações
            pose_text = f'Pose - Yaw: {yaw:.1f}° Pitch: {pitch:.1f}° Roll: {roll:.1f}°'
            cv2.putText(frame, pose_text, (10, 160), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Adiciona indicações de direção
            direction_text = ""
            if abs(yaw) > 10:
                direction_text += f"{'→' if yaw > 0 else '←'} "
            if abs(pitch) > 10:
                direction_text += f"{'↑' if pitch > 0 else '↓'} "
            if abs(roll) > 10:
                direction_text += f"{'↻' if roll > 0 else '↺'} "
            
            if direction_text:
                cv2.putText(frame, f'Movimento: {direction_text}', (10, 180), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
        except Exception as e:
            # Se houver erro na projeção, mostra apenas os ângulos
            rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
            angles = self.rotation_matrix_to_euler_angles(rotation_matrix)
            yaw, pitch, roll = np.degrees(angles)
            
            pose_text = f'Pose - Yaw: {yaw:.1f}° Pitch: {pitch:.1f}° Roll: {roll:.1f}° [Erro projeção]'
            cv2.putText(frame, pose_text, (10, 160), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        return frame
    
    def draw_eye_regions(self, frame, landmarks):
        """
        Desenha retângulos ao redor das regiões dos olhos
        """
        for eye_points, color, label in [
            (self.LEFT_EYE_POINTS, (0, 255, 0), "L"),
            (self.RIGHT_EYE_POINTS, (255, 0, 0), "R")
        ]:
            eye_landmarks = landmarks[eye_points]
            
            x_min = np.min(eye_landmarks[:, 0]) - 15
            x_max = np.max(eye_landmarks[:, 0]) + 15
            y_min = np.min(eye_landmarks[:, 1]) - 15
            y_max = np.max(eye_landmarks[:, 1]) + 15
            
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
            cv2.putText(frame, label, (x_min, y_min - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return frame
    
    def draw_iris(self, frame, iris_data):
        """
        Desenha íris detectadas
        """
        for eye_side, data in iris_data.items():
            center = data['center']
            radius = data['radius']
            points = data['points']
            
            # Cor baseada no lado do olho
            color = (0, 255, 0) if eye_side == 'left' else (255, 0, 0)
            
            # Desenha círculo da íris
            cv2.circle(frame, center, radius, color, 2)
            
            # Desenha centro da íris
            cv2.circle(frame, center, 3, color, -1)
            
            # Desenha pontos de landmark da íris
            for point in points:
                cv2.circle(frame, point, 1, (255, 0, 255), -1)  # Magenta
            
            # Adiciona texto com coordenadas
            text = f'{eye_side[0].upper()}: {center}'
            text_pos = (center[0] - 40, center[1] - radius - 10)
            cv2.putText(frame, text, text_pos, 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        return frame
    
    def draw_gaze_direction_normalized(self, frame, gaze_data):
        """
        Desenha vetores de direção do olhar normalizados.
        """
        for eye_side, data in gaze_data.items():
            # AQUI FOI FEITA A CORREÇÃO:
            # Agora, a função de desenho acessa 'iris_center_global',
            # que é a chave correta retornada pelo novo método de cálculo.
            iris_center_global = data['iris_center_global']
            eye_center_global = data['eye_center_global']
            gaze_vector = data['gaze_vector']
            normalized_vector = data['normalized_vector']
            magnitude = np.linalg.norm(gaze_vector) # Recalcula a magnitude para maior precisão
            pose_compensated = data['pose_compensated']
            
            color = (0, 255, 0) if eye_side == 'left' else (255, 0, 0)
            
            # Desenha linha do centro do olho para a íris
            cv2.line(frame, eye_center_global, iris_center_global, color, 2)
            
            # Estende a linha para mostrar direção normalizada
            # A condição de magnitude > 2 é mantida para filtrar movimentos pequenos
            if magnitude > 2: 
                scale_factor = 40
                end_point = (
                    iris_center_global[0] + int(normalized_vector[0] * scale_factor),
                    iris_center_global[1] + int(normalized_vector[1] * scale_factor)
                )
                
                arrow_color = (0, 255, 255) if pose_compensated else color
                cv2.arrowedLine(frame, iris_center_global, end_point, arrow_color, 3, tipLength=0.3)
            
            # Mostra coordenadas do vetor normalizado
            text = f'{eye_side[0].upper()}: ({normalized_vector[0]:.2f}, {normalized_vector[1]:.2f})'
            if pose_compensated:
                text += ' [PC]'
            
            text_pos = (10, 140 + (20 if eye_side == 'right' else 0))
            cv2.putText(frame, text, text_pos, 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
        return frame
    
    def draw_normalized_eyes(self, frame, gaze_data):
        """
        Mostra janelas com os olhos normalizados
        """
        window_height = 80
        window_width = 120
        start_y = 10
        
        for i, (eye_side, data) in enumerate(gaze_data.items()):
            if 'normalized_eye' in data:
                normalized_eye = data['normalized_eye']
                eye_center = data['eye_center_local']
                
                # Redimensiona olho normalizado para visualização
                if normalized_eye.size > 0:
                    resized_eye = cv2.resize(normalized_eye, (window_width, window_height))
                    
                    # Posição da janela na tela
                    start_x = 10 + i * (window_width + 10)
                    end_x = start_x + window_width
                    end_y = start_y + window_height
                    
                    # Verifica se cabe na tela
                    if end_x < frame.shape[1] and end_y < frame.shape[0]:
                        # Converte para BGR se necessário
                        if len(resized_eye.shape) == 2:
                            resized_eye_bgr = cv2.cvtColor(resized_eye, cv2.COLOR_GRAY2BGR)
                        else:
                            resized_eye_bgr = resized_eye
                        
                        # Sobrepõe na imagem principal
                        frame[start_y:end_y, start_x:end_x] = resized_eye_bgr
                        
                        # Desenha centro do olho na janela normalizada
                        center_in_window = (
                            start_x + int(eye_center[0] * window_width / normalized_eye.shape[1]),
                            start_y + int(eye_center[1] * window_height / normalized_eye.shape[0])
                        )
                        color = (0, 255, 0) if eye_side == 'left' else (255, 0, 0)
                        cv2.circle(frame, center_in_window, 2, color, -1)
                        
                        # Label
                        label = f'Norm {eye_side[0].upper()}'
                        cv2.putText(frame, label, (start_x, start_y - 5), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
                        
                        # Borda
                        cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), color, 1)
        
        return frame
    
    def run(self):
        """
        Loop principal do gaze tracking com normalização
        """
        # Parâmetros da câmera (estimativa)
        frame_count = 0
        
        while True:
            ret, frame = self.cap.read()
            
            if not ret:
                print("Erro: Não foi possível capturar o frame")
                break
            
            frame_count += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detecta rostos
            faces, confidences = self.detect_faces_dnn(frame)
            
            # Detecta íris usando MediaPipe
            iris_data = self.detect_iris_mediapipe(frame)
            
            # Parâmetros da câmera
            h, w = frame.shape[:2]
            focal_length = w
            camera_center = (w // 2, h // 2)
            camera_matrix = np.array([
                [focal_length, 0, camera_center[0]],
                [0, focal_length, camera_center[1]],
                [0, 0, 1]
            ], dtype=np.float64)
            dist_coeffs = np.zeros((4, 1))
            
            # Inicializa as labels fora do loop de rostos
            left_gaze_label = "N/A"
            right_gaze_label = "N/A"

            # Processa cada rosto detectado
            for i, (x, y, w_face, h_face) in enumerate(faces):
                confidence = confidences[i]
                
                # Desenha retângulo do rosto
                if confidence > 0.8:
                    color = (0, 255, 0)
                elif confidence > 0.6:
                    color = (0, 255, 255)
                else:
                    color = (0, 165, 255)
                
                cv2.rectangle(frame, (x, y), (x + w_face, y + h_face), color, 2)
                cv2.putText(frame, f'Face: {confidence:.1%}', (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Obtém landmarks
                try:
                    landmarks = self.get_landmarks(gray, (x, y, w_face, h_face))
                    
                    # Estima pose da cabeça
                    rotation_vector, translation_vector = self.estimate_head_pose(
                        landmarks, camera_matrix, dist_coeffs
                    )
                    
                    # Desenha landmarks se habilitado
                    if self.show_landmarks:
                        frame = self.draw_landmarks(frame, landmarks)
                    
                    # Desenha regiões dos olhos se habilitado
                    if self.show_eye_regions:
                        frame = self.draw_eye_regions(frame, landmarks)
                    
                    # Desenha estimação de pose se habilitado
                    if self.show_pose_estimation:
                        frame = self.draw_head_pose(frame, landmarks, rotation_vector, translation_vector)
                    
                    # Desenha íris e calcula direção do olhar normalizada
                    if self.show_iris_detection and iris_data:
                        frame = self.draw_iris(frame, iris_data)
                        
                        # Calcula direção do olhar com normalização
                        gaze_data = self.calculate_gaze_direction_normalized(
                            iris_data, landmarks, rotation_vector, translation_vector, frame
                        )
                        
                        # Atualiza as labels de direção do olhar
                        if 'left' in gaze_data:
                            left_gaze_label = self.get_gaze_direction_label(gaze_data['left']['normalized_vector'])
                        if 'right' in gaze_data:
                            right_gaze_label = self.get_gaze_direction_label(gaze_data['right']['normalized_vector'])

                        # Desenha direção do olhar normalizada
                        frame = self.draw_gaze_direction_normalized(frame, gaze_data)
                        
                        # Mostra olhos normalizados se habilitado
                        if self.show_normalized_eyes:
                            frame = self.draw_normalized_eyes(frame, gaze_data)
                
                except Exception as e:
                    print(f"Erro ao processar landmarks: {e}")
            
            # Mostra informações na tela
            info_lines = [
                f'Rostos: {len(faces)} | Threshold: {self.confidence_threshold:.1f}',
                f'Landmarks: {"ON" if self.show_landmarks else "OFF"} | '
                f'Regiões: {"ON" if self.show_eye_regions else "OFF"} | '
                f'Íris: {"ON" if self.show_iris_detection else "OFF"}',
                f'Pose: {"ON" if self.show_pose_estimation else "OFF"} | '
                f'Norm Eyes: {"ON" if self.show_normalized_eyes else "OFF"}',
                f'Íris detectadas: {len(iris_data)} | Frame: {frame_count}',
                f'Dir. Olho Esq: {left_gaze_label}',
                f'Dir. Olho Dir: {right_gaze_label}'
            ]
            
            for i, line in enumerate(info_lines):
                cv2.putText(frame, line, (10, 200 + i * 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Instruções
            instructions = [
                'q: sair | +/-: sens | l: landmarks | r: regioes | i: iris',
                'p: pose | n: norm eyes | Seta amarela: compensada'
            ]
            
            for i, instruction in enumerate(instructions):
                cv2.putText(frame, instruction, (10, frame.shape[0] - 30 + i * 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            cv2.imshow('Gaze Tracker - Normalizado', frame)
            
            # Verifica teclas pressionadas
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('+') or key == ord('='):
                self.confidence_threshold = max(0.1, self.confidence_threshold - 0.1)
                print(f"Threshold: {self.confidence_threshold:.1f}")
            elif key == ord('-'):
                self.confidence_threshold = min(0.9, self.confidence_threshold + 0.1)
                print(f"Threshold: {self.confidence_threshold:.1f}")
            elif key == ord('l'):
                self.show_landmarks = not self.show_landmarks
                print(f"Landmarks: {'ON' if self.show_landmarks else 'OFF'}")
            elif key == ord('r'):
                self.show_eye_regions = not self.show_eye_regions
                print(f"Regiões dos olhos: {'ON' if self.show_eye_regions else 'OFF'}")
            elif key == ord('i'):
                self.show_iris_detection = not self.show_iris_detection
                print(f"Detecção de íris: {'ON' if self.show_iris_detection else 'OFF'}")
            elif key == ord('p'):
                self.show_pose_estimation = not self.show_pose_estimation
                print(f"Estimação de pose: {'ON' if self.show_pose_estimation else 'OFF'}")
            elif key == ord('n'):
                self.show_normalized_eyes = not self.show_normalized_eyes
                print(f"Olhos normalizados: {'ON' if self.show_normalized_eyes else 'OFF'}")
        
        self.cap.release()
        cv2.destroyAllWindows()

# Uso do código
if __name__ == "__main__":
    try:
        tracker = GazeTracker()
        tracker.run()
    except ImportError as e:
        if 'mediapipe' in str(e):
            print("Erro: MediaPipe não está instalado!")
            print("Instale com: pip install mediapipe")
        elif 'dlib' in str(e):
            print("Erro: dlib não está instalado!")
            print("Instale com: pip install dlib")
        else:
            print(f"Erro de importação: {e}")
    except Exception as e:
        print(f"Erro: {e}")
