import numpy as np
from collections import deque
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass
from enum import Enum


class FusionMethod(Enum):
    """Método usado para fusão dos olhos"""
    VERGENCE = "vergence"       # Ponto de vergência encontrado
    WEIGHTED = "weighted"       # Média ponderada por confiança
    SINGLE_LEFT = "single_left"   # Apenas olho esquerdo
    SINGLE_RIGHT = "single_right"  # Apenas olho direito
    SIMPLE_AVERAGE = "simple"   # Média simples (fallback)


@dataclass
class FusionResult:
    """Resultado da fusão binocular"""
    combined_direction: np.ndarray  # Direção combinada normalizada
    combined_origin: np.ndarray     # Origem do raio combinado
    method: FusionMethod            # Método usado
    vergence_point: Optional[np.ndarray]  # Ponto de vergência 3D (se calculado)
    vergence_distance: float        # Distância até o ponto de vergência
    vergence_quality: float         # Qualidade da vergência (menor = melhor)
    discrepancy_angle: float        # Ângulo entre os dois olhos (graus)
    confidence: float               # Confiança geral (0-1)


class BinocularGazeFilter:
    """
    Filtro de consistência temporal para gaze binocular.
    
    Aplica smoothing individual a cada olho ANTES da fusão,
    o que produz resultados mais estáveis do que filtrar
    apenas o vetor combinado final.
    """
    
    def __init__(self, 
                 window_size: int = 5,
                 outlier_threshold_deg: float = 15.0,
                 outlier_attenuation: float = 0.2):
        """
        Args:
            window_size: Tamanho da janela de média móvel
            outlier_threshold_deg: Limite em graus para considerar outlier
            outlier_attenuation: Peso dado a outliers (0 = rejeitar, 1 = aceitar)
        """
        self.window_size = window_size
        self.outlier_threshold = outlier_threshold_deg
        self.outlier_attenuation = outlier_attenuation
        
        # Históricos separados para cada olho
        self.left_history: deque = deque(maxlen=window_size)
        self.right_history: deque = deque(maxlen=window_size)
        
        # Últimos valores válidos
        self.last_valid_left: Optional[np.ndarray] = None
        self.last_valid_right: Optional[np.ndarray] = None
        
        # Contadores de outliers (para diagnóstico)
        self.left_outlier_count = 0
        self.right_outlier_count = 0
    
    def reset(self):
        """Limpa os históricos"""
        self.left_history.clear()
        self.right_history.clear()
        self.last_valid_left = None
        self.last_valid_right = None
        self.left_outlier_count = 0
        self.right_outlier_count = 0
    
    def update(self, 
               left_dir: Optional[np.ndarray], 
               right_dir: Optional[np.ndarray]
              ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Atualiza filtro com novas direções e retorna direções filtradas.
        
        Args:
            left_dir: Direção do olho esquerdo (normalizada)
            right_dir: Direção do olho direito (normalizada)
            
        Returns:
            Tupla (direção_esquerda_filtrada, direção_direita_filtrada)
        """
        filtered_left = self._filter_eye(
            left_dir, 
            self.left_history, 
            self.last_valid_left,
            is_left=True
        )
        if filtered_left is not None:
            self.last_valid_left = filtered_left.copy()
        
        filtered_right = self._filter_eye(
            right_dir,
            self.right_history,
            self.last_valid_right,
            is_left=False
        )
        if filtered_right is not None:
            self.last_valid_right = filtered_right.copy()
        
        return filtered_left, filtered_right
    
    def _filter_eye(self,
                    current_dir: Optional[np.ndarray],
                    history: deque,
                    last_valid: Optional[np.ndarray],
                    is_left: bool = True
                   ) -> Optional[np.ndarray]:
        """Filtra direção de um olho individual"""
        if current_dir is None:
            return last_valid
        
        current_dir = np.array(current_dir, dtype=float)
        norm = np.linalg.norm(current_dir)
        if norm < 1e-9:
            return last_valid
        current_dir = current_dir / norm
        
        # Detecção e tratamento de outlier
        if last_valid is not None:
            dot = np.clip(np.dot(current_dir, last_valid), -1.0, 1.0)
            angle_deg = np.degrees(np.arccos(dot))
            
            if angle_deg > self.outlier_threshold:
                # Outlier detectado
                if is_left:
                    self.left_outlier_count += 1
                else:
                    self.right_outlier_count += 1
                
                # Atenuar o salto (interpolar parcialmente)
                alpha = self.outlier_attenuation
                current_dir = (1 - alpha) * last_valid + alpha * current_dir
                current_dir = current_dir / np.linalg.norm(current_dir)
        
        # Adicionar ao histórico
        history.append(current_dir.copy())
        
        # Calcular média móvel
        if len(history) >= 2:
            # Média vetorial (mais estável que média de ângulos)
            avg_dir = np.mean(list(history), axis=0)
            avg_norm = np.linalg.norm(avg_dir)
            if avg_norm > 1e-9:
                return avg_dir / avg_norm
        
        return current_dir
    
    def get_diagnostics(self) -> Dict:
        """Retorna estatísticas para diagnóstico"""
        return {
            'left_history_size': len(self.left_history),
            'right_history_size': len(self.right_history),
            'left_outliers': self.left_outlier_count,
            'right_outliers': self.right_outlier_count,
            'has_left_reference': self.last_valid_left is not None,
            'has_right_reference': self.last_valid_right is not None
        }


class BinocularFusion:
    """
    Sistema de fusão binocular para combinar gaze dos dois olhos.
    
    Implementa três estratégias:
    1. Vergência 3D: Encontra o ponto onde os raios se cruzam
    2. Ponderação por Confiança: Dá mais peso ao olho mais estável
    3. Fallback: Usa apenas um olho quando o outro é muito ruidoso
    """
    
    def __init__(self,
                 vergence_quality_threshold: float = 30.0,
                 discrepancy_threshold_deg: float = 20.0,
                 use_temporal_filter: bool = True,
                 filter_window_size: int = 5):
        """
        Args:
            vergence_quality_threshold: Distância máxima entre raios para 
                                        considerar vergência válida (mm)
            discrepancy_threshold_deg: Ângulo máximo entre olhos antes de
                                       usar apenas um (graus)
            use_temporal_filter: Se True, aplica filtro temporal individual
            filter_window_size: Tamanho da janela do filtro
        """
        self.vergence_threshold = vergence_quality_threshold
        self.discrepancy_threshold = discrepancy_threshold_deg
        self.use_temporal_filter = use_temporal_filter
        
        # Filtro temporal
        if use_temporal_filter:
            self.temporal_filter = BinocularGazeFilter(
                window_size=filter_window_size
            )
        else:
            self.temporal_filter = None
        
        # Histórico de métodos usados (para diagnóstico)
        self.method_history: deque = deque(maxlen=100)
        
        # Confiança por olho (atualizada dinamicamente)
        self.left_confidence = 0.5
        self.right_confidence = 0.5
    
    def reset(self):
        """Limpa estados internos"""
        if self.temporal_filter:
            self.temporal_filter.reset()
        self.method_history.clear()
        self.left_confidence = 0.5
        self.right_confidence = 0.5
    
    def fuse(self,
             origin_left: np.ndarray,
             direction_left: np.ndarray,
             origin_right: np.ndarray,
             direction_right: np.ndarray
            ) -> FusionResult:
        """
        Combina gaze dos dois olhos.
        
        Args:
            origin_left: Origem do raio esquerdo (centro da esfera ocular)
            direction_left: Direção normalizada do olho esquerdo
            origin_right: Origem do raio direito
            direction_right: Direção normalizada do olho direito
            
        Returns:
            FusionResult com todos os dados da fusão
        """
        # Converter para arrays
        origin_l = np.array(origin_left, dtype=float)
        origin_r = np.array(origin_right, dtype=float)
        dir_l = np.array(direction_left, dtype=float)
        dir_r = np.array(direction_right, dtype=float)
        
        # Normalizar direções
        dir_l = self._normalize(dir_l)
        dir_r = self._normalize(dir_r)
        
        # Aplicar filtro temporal se habilitado
        if self.temporal_filter:
            dir_l_filtered, dir_r_filtered = self.temporal_filter.update(
                dir_l, dir_r
            )
            if dir_l_filtered is not None:
                dir_l = dir_l_filtered
            if dir_r_filtered is not None:
                dir_r = dir_r_filtered
        
        # Calcular discrepância angular
        discrepancy = self._compute_discrepancy(dir_l, dir_r)
        
        # Origem combinada (sempre o ponto médio entre os olhos)
        origin_combined = (origin_l + origin_r) / 2.0
        
        # Estratégia 1: Tentar encontrar ponto de vergência
        vergence_point, vergence_quality = self._compute_vergence_point(
            origin_l, dir_l, origin_r, dir_r
        )
        
        # Decidir método baseado na qualidade
        if (vergence_point is not None and 
            vergence_quality < self.vergence_threshold and
            discrepancy < self.discrepancy_threshold):
            # Boa vergência - usar ponto de vergência
            combined_dir = vergence_point - origin_combined
            combined_dir = self._normalize(combined_dir)
            method = FusionMethod.VERGENCE
            confidence = 1.0 - (vergence_quality / self.vergence_threshold)
            
        elif discrepancy > self.discrepancy_threshold:
            # Discrepância muito alta - usar apenas um olho
            combined_dir, method = self._select_best_eye(
                dir_l, dir_r, origin_combined
            )
            confidence = 0.5  # Confiança reduzida
            vergence_point = None
            
        else:
            # Usar média ponderada por confiança
            combined_dir = self._weighted_average(dir_l, dir_r)
            method = FusionMethod.WEIGHTED
            confidence = 1.0 - (discrepancy / 90.0)  # 0-90 graus -> 1-0
            vergence_point = None
        
        # Calcular distância de vergência
        if vergence_point is not None:
            vergence_distance = np.linalg.norm(vergence_point - origin_combined)
        else:
            # Estimar distância baseada na direção
            vergence_distance = 500.0  # Valor padrão em mm
        
        # Registrar método usado
        self.method_history.append(method)
        
        return FusionResult(
            combined_direction=combined_dir,
            combined_origin=origin_combined,
            method=method,
            vergence_point=vergence_point,
            vergence_distance=vergence_distance,
            vergence_quality=vergence_quality if vergence_quality != float('inf') else -1,
            discrepancy_angle=discrepancy,
            confidence=confidence
        )
    
    def _normalize(self, v: np.ndarray) -> np.ndarray:
        """Normaliza vetor"""
        n = np.linalg.norm(v)
        return v / n if n > 1e-9 else v
    
    def _compute_discrepancy(self, 
                              dir_l: np.ndarray, 
                              dir_r: np.ndarray) -> float:
        """Calcula ângulo entre os dois vetores em graus"""
        dot = np.clip(np.dot(dir_l, dir_r), -1.0, 1.0)
        return np.degrees(np.arccos(dot))
    
    def _compute_vergence_point(self,
                                 origin_l: np.ndarray,
                                 dir_l: np.ndarray,
                                 origin_r: np.ndarray,
                                 dir_r: np.ndarray
                                ) -> Tuple[Optional[np.ndarray], float]:
        """
        Calcula o ponto de vergência (onde os raios se cruzam ou passam mais próximo).
        
        Usa o método de menor distância entre duas retas 3D.
        
        Returns:
            (ponto_vergência, distância_mínima_entre_raios)
        """
        # Vetor entre origens
        w0 = origin_l - origin_r
        
        # Produtos escalares
        a = np.dot(dir_l, dir_l)  # |dir_l|² = 1
        b = np.dot(dir_l, dir_r)
        c = np.dot(dir_r, dir_r)  # |dir_r|² = 1
        d = np.dot(dir_l, w0)
        e = np.dot(dir_r, w0)
        
        denom = a * c - b * b
        
        if abs(denom) < 1e-9:
            # Raios paralelos - sem ponto de vergência definido
            return None, float('inf')
        
        # Parâmetros dos pontos mais próximos em cada raio
        s = (b * e - c * d) / denom
        t = (a * e - b * d) / denom
        
        # Apenas considerar pontos à frente dos olhos (s, t > 0)
        if s < 0 or t < 0:
            # Ponto de cruzamento está atrás de um dos olhos
            # Tentar projetar para frente
            s = max(0, s)
            t = max(0, t)
        
        # Pontos mais próximos em cada raio
        point_l = origin_l + s * dir_l
        point_r = origin_r + t * dir_r
        
        # Ponto de vergência = média dos pontos mais próximos
        vergence_point = (point_l + point_r) / 2.0
        
        # Distância mínima entre os raios (qualidade)
        min_distance = np.linalg.norm(point_l - point_r)
        
        return vergence_point, min_distance
    
    def _select_best_eye(self,
                          dir_l: np.ndarray,
                          dir_r: np.ndarray,
                          origin: np.ndarray
                         ) -> Tuple[np.ndarray, FusionMethod]:
        """
        Seleciona o olho mais confiável quando discrepância é muito alta.
        
        Critérios:
        - Preferência pelo olho com menor variância histórica
        - Preferência pelo olho mais alinhado ao eixo frontal
        """
        forward = np.array([0, 0, 1], dtype=float)
        
        # Confiança baseada em alinhamento frontal
        conf_l = abs(np.dot(dir_l, forward))
        conf_r = abs(np.dot(dir_r, forward))
        
        # Atualizar confiâncias históricas (média móvel exponencial)
        alpha = 0.3
        self.left_confidence = (1 - alpha) * self.left_confidence + alpha * conf_l
        self.right_confidence = (1 - alpha) * self.right_confidence + alpha * conf_r
        
        # Decidir qual usar
        if self.right_confidence > self.left_confidence * 1.2:
            return dir_r, FusionMethod.SINGLE_RIGHT
        elif self.left_confidence > self.right_confidence * 1.2:
            return dir_l, FusionMethod.SINGLE_LEFT
        else:
            # Sem diferença clara - usar média simples
            avg = (dir_l + dir_r) / 2.0
            return self._normalize(avg), FusionMethod.SIMPLE_AVERAGE
    
    def _weighted_average(self,
                           dir_l: np.ndarray,
                           dir_r: np.ndarray
                          ) -> np.ndarray:
        """
        Calcula média ponderada baseada nas confiâncias históricas.
        """
        total = self.left_confidence + self.right_confidence
        if total < 1e-9:
            weight_l = 0.5
        else:
            weight_l = self.left_confidence / total
        
        combined = weight_l * dir_l + (1 - weight_l) * dir_r
        return self._normalize(combined)
    
    def get_method_statistics(self) -> Dict:
        """Retorna estatísticas dos métodos usados"""
        if not self.method_history:
            return {}
        
        counts = {}
        for method in self.method_history:
            counts[method.value] = counts.get(method.value, 0) + 1
        
        total = len(self.method_history)
        return {
            'counts': counts,
            'percentages': {k: v/total*100 for k, v in counts.items()},
            'most_common': max(counts.keys(), key=lambda k: counts[k]),
            'left_confidence': self.left_confidence,
            'right_confidence': self.right_confidence
        }


# =============================================================================
# Funções auxiliares para integração com seu código existente
# =============================================================================

def create_binocular_fusion_system(
    use_temporal_filter: bool = True,
    vergence_threshold: float = 30.0,
    discrepancy_threshold: float = 20.0,
    filter_window: int = 5
) -> BinocularFusion:
    """
    Cria e configura o sistema de fusão binocular.
    """
    return BinocularFusion(
        vergence_quality_threshold=vergence_threshold,
        discrepancy_threshold_deg=discrepancy_threshold,
        use_temporal_filter=use_temporal_filter,
        filter_window_size=filter_window
    )


def integrate_with_gaze_vectors(
    fusion_system: BinocularFusion,
    gaze_vectors_3d: Dict
) -> Optional[FusionResult]:
    """
    Integra o sistema de fusão com o formato de dados do seu gaze_calculator.
    
    Args:
        fusion_system: Instância de BinocularFusion
        gaze_vectors_3d: Dicionário retornado por EyeSphereSystem.compute_gaze_3d()
        
    Returns:
        FusionResult ou None se dados insuficientes
    """
    if 'left' not in gaze_vectors_3d or 'right' not in gaze_vectors_3d:
        return None
    
    left_data = gaze_vectors_3d['left']
    right_data = gaze_vectors_3d['right']
    
    # Extrair origens e direções
    origin_l = left_data.get('origin')
    dir_l = left_data.get('direction')
    origin_r = right_data.get('origin')
    dir_r = right_data.get('direction')
    
    if any(x is None for x in [origin_l, dir_l, origin_r, dir_r]):
        return None
    
    return fusion_system.fuse(origin_l, dir_l, origin_r, dir_r)
