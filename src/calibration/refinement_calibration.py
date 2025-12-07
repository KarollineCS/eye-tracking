import cv2
import numpy as np
import time
from collections import deque

try:
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import Ridge
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️ sklearn não disponível. Usando homografia como fallback.")


class RefinementCalibration:
    """
    Calibração de refinamento com 9 pontos.
    Corrige erros sistemáticos do sistema 3D.
    """
    
    def __init__(self, screen_width: int, screen_height: int):
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        # Pontos de calibração (9 pontos em grade 3x3)
        self.calibration_points = self._generate_calibration_points()
        self.current_point_idx = 0
        
        # Dados coletados
        self.collected_data = []
        self.current_samples = deque(maxlen=50)
        
        # Modelos de correção
        self.model_x = None
        self.model_y = None
        self.poly_features = None
        self.homography_matrix = None
        self.use_polynomial = SKLEARN_AVAILABLE
        
        # Estado
        self.calibration_complete = False
        self.calibration_active = False
        self.is_collecting = False
        
        # Parâmetros
        self.min_samples_per_point = 20
        self.collection_time = 2.0  # segundos por ponto
        self.point_radius = 30
        
    def _generate_calibration_points(self):
        """Gera 9 pontos em grade 3x3."""
        points = []
        margin = 0.1  # 10% de margem
        
        for row, y_ratio in enumerate([margin, 0.5, 1 - margin]):
            for col, x_ratio in enumerate([margin, 0.5, 1 - margin]):
                px = int(x_ratio * self.screen_width)
                py = int(y_ratio * self.screen_height)
                points.append((px, py))
        
        return points
    
    def is_calibrated(self) -> bool:
        """Verifica se está calibrado."""
        if self.use_polynomial:
            return self.model_x is not None and self.model_y is not None
        else:
            return self.homography_matrix is not None
    
    def start_calibration(self):
        """Inicia processo de calibração."""
        self.current_point_idx = 0
        self.collected_data = []
        self.current_samples.clear()
        self.calibration_complete = False
        self.calibration_active = True
        self.is_collecting = False
        self.collection_start_time = None
        
        print("\n" + "=" * 50)
        print("CALIBRAÇÃO DE REFINAMENTO - 9 PONTOS")
        print("=" * 50)
        print("Olhe para cada ponto e pressione ESPAÇO para coletar")
        print("ESC para cancelar")
        print("=" * 50 + "\n")
    
    def add_gaze_sample(self, predicted_x: float, predicted_y: float):
        """
        Adiciona amostra de gaze predito pelo sistema 3D.
        Chamar continuamente durante a coleta.
        """
        if not self.calibration_active or not self.is_collecting:
            return
        
        self.current_samples.append((predicted_x, predicted_y))
    
    def start_collection(self):
        """Inicia coleta para o ponto atual."""
        self.current_samples.clear()
        self.is_collecting = True
        self.collection_start_time = time.time()
    
    def stop_collection(self):
        """Para coleta e salva dados do ponto atual."""
        if not self.is_collecting:
            return False
        
        self.is_collecting = False
        
        if len(self.current_samples) < self.min_samples_per_point:
            print(f"⚠️ Poucas amostras ({len(self.current_samples)}). Tente novamente.")
            return False
        
        # Calcular mediana das predições (mais robusto que média)
        samples = np.array(list(self.current_samples))
        median_x = np.median(samples[:, 0])
        median_y = np.median(samples[:, 1])
        
        # Ponto alvo
        target_x, target_y = self.calibration_points[self.current_point_idx]
        
        # Salvar par (predição, alvo)
        self.collected_data.append({
            'predicted': (median_x, median_y),
            'target': (target_x, target_y),
            'n_samples': len(self.current_samples),
            'std_x': np.std(samples[:, 0]),
            'std_y': np.std(samples[:, 1])
        })
        
        print(f"✓ Ponto {self.current_point_idx + 1}/9: "
              f"Predito ({median_x:.0f}, {median_y:.0f}) → "
              f"Alvo ({target_x}, {target_y}) "
              f"[{len(self.current_samples)} amostras]")
        
        # Avançar para próximo ponto
        self.current_point_idx += 1
        self.current_samples.clear()
        
        # Verificar se terminou
        if self.current_point_idx >= len(self.calibration_points):
            return self._finalize_calibration()
        
        return True
    
    def _finalize_calibration(self):
        """Finaliza calibração e treina modelos."""
        print("\n" + "-" * 40)
        print("Treinando modelo de correção...")
        
        if len(self.collected_data) < 4:
            print("❌ Dados insuficientes para calibração")
            self.calibration_active = False
            return False
        
        # Preparar dados
        X = np.array([d['predicted'] for d in self.collected_data])
        y_x = np.array([d['target'][0] for d in self.collected_data])
        y_y = np.array([d['target'][1] for d in self.collected_data])
        
        success = False
        
        # Tentar modelo polinomial primeiro
        if self.use_polynomial:
            try:
                success = self._train_polynomial_model(X, y_x, y_y)
            except Exception as e:
                print(f"⚠️ Erro no modelo polinomial: {e}")
                success = False
        
        # Fallback para homografia
        if not success:
            success = self._train_homography_model(X, y_x, y_y)
        
        if success:
            self._calculate_calibration_error()
            self.calibration_complete = True
            print("\n✅ CALIBRAÇÃO DE REFINAMENTO CONCLUÍDA!")
        else:
            print("\n❌ Falha na calibração")
        
        self.calibration_active = False
        return success
    
    def _train_polynomial_model(self, X, y_x, y_y):
        """Treina modelo polinomial de grau 2."""
        self.poly_features = PolynomialFeatures(degree=2, include_bias=True)
        X_poly = self.poly_features.fit_transform(X)
        
        self.model_x = Ridge(alpha=0.1)
        self.model_y = Ridge(alpha=0.1)
        
        self.model_x.fit(X_poly, y_x)
        self.model_y.fit(X_poly, y_y)
        
        print("✓ Modelo polinomial treinado")
        return True
    
    def _train_homography_model(self, X, y_x, y_y):
        """Treina modelo de homografia."""
        src_points = X.astype(np.float32)
        dst_points = np.column_stack([y_x, y_y]).astype(np.float32)
        
        self.homography_matrix, _ = cv2.findHomography(
            src_points, dst_points, cv2.RANSAC, 5.0
        )
        
        if self.homography_matrix is not None:
            print("✓ Modelo homografia treinado")
            return True
        return False
    
    def _calculate_calibration_error(self):
        """Calcula erro de calibração."""
        errors = []
        
        for data in self.collected_data:
            pred_x, pred_y = data['predicted']
            target_x, target_y = data['target']
            
            # Aplicar correção
            corrected = self.correct_gaze(pred_x, pred_y)
            if corrected:
                corr_x, corr_y = corrected
                error = np.sqrt((corr_x - target_x)**2 + (corr_y - target_y)**2)
                errors.append(error)
        
        if errors:
            mean_error = np.mean(errors)
            std_error = np.std(errors)
            print(f"Erro médio após correção: {mean_error:.1f} ± {std_error:.1f} pixels")
    
    def correct_gaze(self, predicted_x: float, predicted_y: float):
        """
        Aplica correção ao ponto de gaze predito.
        
        Args:
            predicted_x, predicted_y: Coordenadas preditas pelo sistema 3D
            
        Returns:
            (corrected_x, corrected_y) ou None se não calibrado
        """
        if not self.is_calibrated():
            return None
        
        if self.use_polynomial and self.model_x is not None:
            # Modelo polinomial
            X = np.array([[predicted_x, predicted_y]])
            X_poly = self.poly_features.transform(X)
            
            corrected_x = float(self.model_x.predict(X_poly)[0])
            corrected_y = float(self.model_y.predict(X_poly)[0])
            
        elif self.homography_matrix is not None:
            # Homografia
            point = np.array([[[predicted_x, predicted_y]]], dtype=np.float32)
            transformed = cv2.perspectiveTransform(point, self.homography_matrix)
            
            corrected_x = float(transformed[0, 0, 0])
            corrected_y = float(transformed[0, 0, 1])
        else:
            return None
        
        # Clamp às bordas da tela
        corrected_x = max(0, min(self.screen_width - 1, corrected_x))
        corrected_y = max(0, min(self.screen_height - 1, corrected_y))
        
        return (int(corrected_x), int(corrected_y))
    
    def display_calibration_point(self) -> np.ndarray:
        """Renderiza tela de calibração."""
        screen = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)
        
        if self.current_point_idx >= len(self.calibration_points):
            # Calibração finalizada
            cv2.putText(screen, "Calibracao Concluida!", 
                       (self.screen_width // 2 - 200, self.screen_height // 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            return screen
        
        # Ponto atual
        px, py = self.calibration_points[self.current_point_idx]
        
        # Cor do ponto
        if self.is_collecting:
            # Verde durante coleta
            color = (0, 255, 0)
            # Barra de progresso
            elapsed = time.time() - self.collection_start_time
            progress = min(1.0, elapsed / self.collection_time)
            bar_width = int(200 * progress)
            cv2.rectangle(screen, 
                         (self.screen_width // 2 - 100, self.screen_height - 50),
                         (self.screen_width // 2 - 100 + bar_width, self.screen_height - 30),
                         (0, 255, 0), -1)
            cv2.rectangle(screen,
                         (self.screen_width // 2 - 100, self.screen_height - 50),
                         (self.screen_width // 2 + 100, self.screen_height - 30),
                         (255, 255, 255), 2)
        else:
            # Amarelo aguardando
            color = (0, 255, 255)
        
        # Desenhar ponto
        cv2.circle(screen, (px, py), self.point_radius, color, -1)
        cv2.circle(screen, (px, py), self.point_radius + 5, (255, 255, 255), 2)
        cv2.circle(screen, (px, py), 5, (0, 0, 0), -1)  # Centro
        
        # Instruções
        point_num = self.current_point_idx + 1
        if self.is_collecting:
            text = f"Ponto {point_num}/9 - Coletando... ({len(self.current_samples)} amostras)"
        else:
            text = f"Ponto {point_num}/9 - Olhe para o ponto e pressione ESPACO"
        
        cv2.putText(screen, text,
                   (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        cv2.putText(screen, "ESC para cancelar",
                   (20, self.screen_height - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
        
        return screen
    
    def get_calibration_stats(self):
        """Retorna estatísticas da calibração."""
        if not self.collected_data:
            return None
        
        return {
            'n_points': len(self.collected_data),
            'avg_samples': np.mean([d['n_samples'] for d in self.collected_data]),
            'avg_std_x': np.mean([d['std_x'] for d in self.collected_data]),
            'avg_std_y': np.mean([d['std_y'] for d in self.collected_data]),
            'model_type': 'polynomial' if self.use_polynomial else 'homography'
        }
